import os
import time
import uuid
import json
import base64
import mimetypes
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
PATIENT_UPLOAD_DIR = UPLOADS_DIR / "patient"
DOCTOR_UPLOAD_DIR = UPLOADS_DIR / "doctor"
SQLITE_DB_PATH = BASE_DIR / "curecraft.sqlite3"
MAX_FILE_SIZE_BYTES = int(os.getenv("CURECRAFT_MAX_FILE_MB", "25")) * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = int(os.getenv("CURECRAFT_MAX_TOTAL_UPLOAD_MB", "60")) * 1024 * 1024
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("CURECRAFT_AI_MODEL", "gpt-5.6").strip() or "gpt-5.6"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCTOR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CureCraft AI - Advanced Triage & EHR")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# =====================================================================
# CENTRALIZED DATABASE
# =====================================================================



def _db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_persistent_db() -> None:
    with _db_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS doctor_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL UNIQUE,
            patient_id TEXT NOT NULL,
            doctor_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            feedback TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_doctor_reviews_doctor ON doctor_reviews(doctor_id);
        CREATE INDEX IF NOT EXISTS idx_doctor_reviews_patient ON doctor_reviews(patient_id);
        """)


def get_doctor_rating(doctor_id: str) -> Dict[str, Any]:
    with _db_connection() as conn:
        row = conn.execute(
            "SELECT ROUND(AVG(rating), 2) AS average, COUNT(*) AS count FROM doctor_reviews WHERE doctor_id = ?",
            (doctor_id,),
        ).fetchone()
    return {
        "average": float(row["average"] or 0),
        "count": int(row["count"] or 0),
    }


def get_case_review(case_id: str) -> Optional[Dict[str, Any]]:
    with _db_connection() as conn:
        row = conn.execute(
            "SELECT id, case_id, patient_id, doctor_id, rating, feedback, created_at "
            "FROM doctor_reviews WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    return dict(row) if row else None


def list_doctor_reviews() -> List[Dict[str, Any]]:
    with _db_connection() as conn:
        rows = conn.execute(
            "SELECT id, case_id, patient_id, doctor_id, rating, feedback, created_at "
            "FROM doctor_reviews ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def save_doctor_review(case_id: str, patient_id: str, doctor_id: str, rating: int, feedback: str) -> Dict[str, Any]:
    try:
        with _db_connection() as conn:
            cur = conn.execute(
                "INSERT INTO doctor_reviews(case_id, patient_id, doctor_id, rating, feedback, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, patient_id, doctor_id, rating, feedback.strip()[:2000], time.time()),
            )
            review_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("A review has already been submitted for this completed case.")
    review = get_case_review(case_id) or {
        "id": review_id,
        "case_id": case_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "rating": rating,
        "feedback": feedback.strip()[:2000],
        "created_at": time.time(),
    }
    return review


init_persistent_db()

CENTRAL_DB = {
    "hospitals": {
        "H-01": {"id": "H-01", "name": "AIIMS Central", "location": "New Delhi"},
        "H-02": {"id": "H-02", "name": "Apollo Specialty", "location": "Mumbai"}
    },
    "users": {
        "admin_chief": {"role": "admin", "password": "admin", "name": "Dr. A. Gupta", "title": "Chief Medical Officer", "profile_pic": None},
        "DOC-8831": {"role": "doctor", "password": "doc", "name": "Dr. R. Sharma", "dept": "Cardiology", "hospital_id": "H-01", "consults": 142, "profile_pic": None},
        "DOC-9924": {"role": "doctor", "password": "doc", "name": "Dr. M. Patel", "dept": "Neurology", "hospital_id": "H-02", "consults": 89, "profile_pic": None},
        "ABHA-9844": {"role": "patient", "password": "pass", "name": "Ramesh Chandra", "gender": "Male", "age": 54, "blood_group": "O+", "allergies": "None", "profile_pic": None}
    },
    "medicines": {
        "M-101": {"code": "M-101", "name": "Paracetamol 500mg"},
        "M-102": {"code": "M-102", "name": "Amoxicillin 250mg"},
        "M-103": {"code": "M-103", "name": "Lisinopril 10mg"},
        "M-104": {"code": "M-104", "name": "Metformin 500mg"}
    },
    "cases": {} # Central Case IDs storage
}

# =====================================================================
# PYDANTIC MODELS
# =====================================================================

class AuthReq(BaseModel): username: str; password: str
class PatRegReq(BaseModel): username: str; password: str; name: str; age: int; gender: str; blood_group: str
class ProfileUpdateReq(BaseModel): user_id: str; new_password: Optional[str] = None; profile_pic: Optional[str] = None; updates: Dict[str, str]

class HospReq(BaseModel): admin_id: str; id: Optional[str] = None; name: str; location: str; action: str
class MedReq(BaseModel): admin_id: str; code: str; name: str; action: str
class StaffReq(BaseModel): admin_id: str; user_id: Optional[str] = None; new_password: Optional[str] = None; name: str; role: str; dept: Optional[str] = None; hospital_id: Optional[str] = None; action: str

class IntakeReq(BaseModel):
    abha_id: str
    is_followup: bool
    transcript: str
    photos: List[Any]
    hospital_id: Optional[str] = None
    dept: Optional[str] = None
    doc_id: Optional[str] = None
    ref_case_id: Optional[str] = None

class PrescribeReq(BaseModel):
    case_id: str; abha_id: str; doc_id: str; clinical_notes: str; medicine_codes: List[str]; attachments: List[Any]

class AIAnalysisReq(BaseModel):
    transcript: str
    age: Optional[int] = None
    gender: Optional[str] = None

class ReviewReq(BaseModel):
    patient_id: str
    case_id: str
    doctor_id: str
    rating: int
    feedback: str = ""

# =====================================================================
# FILE STORAGE HELPERS
# =====================================================================

def _safe_filename(filename: str) -> str:
    filename = os.path.basename(filename or "attachment")
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip() or "attachment"
    return filename[:150]


def save_attachments(items: List[Any], category: str, case_id: str) -> List[Dict[str, Any]]:
    """Save browser-uploaded files locally. File type is unrestricted; size is bounded."""
    saved: List[Dict[str, Any]] = []
    if not items:
        return saved

    folder = UPLOADS_DIR / category
    folder.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    for item in items:
        try:
            original_name = "attachment"
            mime = "application/octet-stream"

            if isinstance(item, dict):
                original_name = item.get("name") or original_name
                mime = item.get("type") or mime
                data = item.get("data") or item.get("content") or ""
            else:
                data = str(item)

            if data.startswith("data:") and "," in data:
                header, encoded = data.split(",", 1)
                header_mime = header[5:].split(";", 1)[0]
                if header_mime:
                    mime = header_mime
            else:
                encoded = data

            raw = base64.b64decode(encoded, validate=False)
            if len(raw) > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit")
            if total_bytes + len(raw) > MAX_TOTAL_UPLOAD_BYTES:
                raise ValueError("Total upload batch exceeds the allowed size limit")

            total_bytes += len(raw)
            ext = Path(original_name).suffix
            if not ext:
                ext = mimetypes.guess_extension(mime) or ""

            stored_name = f"{case_id}_{uuid.uuid4().hex}{ext}"
            path = folder / stored_name
            path.write_bytes(raw)

            saved.append({
                "id": uuid.uuid4().hex,
                "name": _safe_filename(original_name),
                "type": mime,
                "size": len(raw),
                "url": f"/uploads/{category}/{stored_name}"
            })
        except Exception as exc:
            print(f"Attachment save failed: {exc}")

    return saved


# =====================================================================
# AI ANALYSIS ENGINE
# =====================================================================

RED_FLAG_RULES = [
    (r"\b(severe|crushing|pressure|tight)\b.*\b(chest|heart)\b|\b(chest|heart)\b.*\b(severe|crushing|pressure|tight)\b", "Possible cardiac emergency"),
    (r"\b(shortness of breath|can't breathe|cannot breathe|difficulty breathing|breathless)\b", "Breathing emergency"),
    (r"\b(face droop|facial droop|slurred speech|can't speak|cannot speak|one[- ]sided weakness|one sided weakness)\b", "Possible stroke warning sign"),
    (r"\b(unconscious|passed out|fainted and not recovering|seizure)\b", "Reduced consciousness / seizure"),
    (r"\b(heavy bleeding|uncontrolled bleeding|vomiting blood|coughing blood)\b", "Potentially significant bleeding"),
    (r"\b(suicidal|want to die|kill myself|self harm)\b", "Immediate mental-health safety concern"),
]

DEPARTMENT_RULES = [
    ("Neurology / Stroke", ["headache", "migraine", "seizure", "weakness", "numbness", "vertigo", "stroke", "slurred speech"]),
    ("Cardiology / Emergency", ["chest", "palpitation", "heart", "angina"]),
    ("Pulmonology / Respiratory", ["cough", "wheeze", "breathing", "breathless", "shortness of breath", "asthma"]),
    ("Gastroenterology", ["stomach", "abdomen", "abdominal", "diarrhea", "vomit", "nausea", "acidity", "constipation"]),
    ("ENT", ["ear pain", "sore throat", "sinus", "hearing", "tonsil"]),
    ("Dermatology", ["rash", "itch", "skin", "acne", "eczema"]),
    ("Orthopedics", ["joint", "bone", "knee", "shoulder", "back pain", "fracture"]),
    ("General Medicine", ["fever", "cold", "fatigue", "body ache", "pain", "infection"]),
]


def deterministic_safety_screen(text: str, age: Optional[int] = None) -> Dict[str, Any]:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    flags: List[str] = []
    for pattern, label in RED_FLAG_RULES:
        if re.search(pattern, normalized):
            flags.append(label)

    if age is not None and age < 18:
        default_department = "Pediatrics"
    else:
        default_department = "General Medicine"

    department = default_department
    for dep, keywords in DEPARTMENT_RULES:
        if any(k in normalized for k in keywords):
            department = dep
            break

    if flags:
        priority = "Level 1 - Emergency"
        risk = "High"
        status = "Immediate emergency assessment"
        action = "Seek emergency medical care now; do not rely on this software to rule out a serious condition."
    elif not normalized:
        priority = "Level 4 - Insufficient data"
        risk = "Unknown"
        status = "More information needed"
        action = "Provide the main symptom, when it started, severity, and associated symptoms."
    else:
        priority = "Level 3 - Routine clinical review"
        risk = "Moderate / undetermined"
        status = "Clinical assessment recommended"
        action = "Arrange an appropriate clinician review, sooner if symptoms worsen or new red flags appear."

    return {
        "status": status,
        "priority": priority,
        "priority_code": 1 if flags else 3,
        "dept": department,
        "risk": risk,
        "red_flags": flags,
        "safety_action": action,
    }


def _extract_response_text(response_json: Dict[str, Any]) -> tuple[str, List[Dict[str, str]]]:
    sources: List[Dict[str, str]] = []
    direct = response_json.get("output_text")
    chunks: List[str] = []

    for item in response_json.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in ("output_text", "text") and isinstance(content.get("text"), str):
                chunks.append(content["text"])
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") == "url_citation":
                    url = annotation.get("url") or annotation.get("source_url")
                    title = annotation.get("title") or url or "Web source"
                    if url:
                        sources.append({"title": title, "url": url})

    raw_text = direct.strip() if isinstance(direct, str) and direct.strip() else "\n".join(chunks).strip()

    unique_sources: List[Dict[str, str]] = []
    seen = set()
    for source in sources:
        key = source.get("url")
        if key and key not in seen:
            seen.add(key)
            unique_sources.append(source)
    return raw_text, unique_sources


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    candidate = (raw_text or "").strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start_idx = candidate.find("{")
    end_idx = candidate.rfind("}")
    if start_idx >= 0 and end_idx > start_idx:
        try:
            obj = json.loads(candidate[start_idx:end_idx + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _sanitize_ai_report(report: Dict[str, Any], safety: Dict[str, Any], sources: List[Dict[str, str]], mode: str, raw_text: str = "") -> Dict[str, Any]:
    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()][:8]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    urgency = str(report.get("urgency") or report.get("priority") or safety["priority"]).strip()
    if safety["red_flags"]:
        urgency = "Level 1 - Emergency"

    dept = str(report.get("recommended_department") or report.get("department") or safety["dept"]).strip()
    if safety["red_flags"] and "Emergency" not in dept:
        dept = safety["dept"]

    medication_guidance = str(
        report.get("medication_guidance")
        or "No individualized prescription is generated by the AI. Medication decisions should be made by a licensed clinician using the patient's full history, examination, allergies, and current medicines."
    ).strip()

    result = {
        "status": str(report.get("status") or safety["status"]).strip(),
        "priority": urgency,
        "priority_code": 1 if safety["red_flags"] else int(report.get("priority_code") or safety["priority_code"]),
        "dept": dept,
        "risk": str(report.get("risk") or safety["risk"]).strip(),
        "clinical_summary": str(report.get("clinical_summary") or report.get("summary") or "Clinical interpretation unavailable.").strip(),
        "red_flags": _as_list(report.get("red_flags")) or safety["red_flags"],
        "what_to_do_now": _as_list(report.get("what_to_do_now") or report.get("recommended_action")) or [safety["safety_action"]],
        "questions_for_doctor": _as_list(report.get("questions_for_doctor"))[:6],
        "medication_guidance": medication_guidance,
        "ayush_dosha_profile": str(report.get("ayush_dosha_profile") or "Not clinically assessed"),
        "ai_mode": mode,
        "sources": sources[:8],
    }

    if not result["clinical_summary"] and raw_text:
        result["clinical_summary"] = raw_text[:1200]
    result["fhir_resource"] = {
        "resourceType": "Observation",
        "status": "final",
        "category": "symptom-intake",
        "code": {"text": "Patient-reported symptom narrative"},
        "valueString": result["clinical_summary"][:1200],
        "interpretation": result["priority"],
        "note": [{"text": "Decision-support only; not a diagnosis or prescription."}]
    }
    return result


def _call_openai_web_search(transcript: str, age: Optional[int], gender: Optional[str]) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    prompt = f"""
You are the clinical decision-support layer of CureCraft, a patient case-taking prototype.
Analyze the symptom narrative conservatively. This is not a diagnosis and must never replace emergency care or an in-person clinician.
Use current web search to verify time-sensitive clinical facts and prefer high-quality medical sources (government, major hospitals, professional societies, peer-reviewed sources).
Do not reveal chain-of-thought. Return ONLY a JSON object with these keys:
clinical_summary, urgency, priority_code, recommended_department, risk, red_flags, what_to_do_now, questions_for_doctor, medication_guidance.
Rules:
- urgency must be one of: "Level 1 - Emergency", "Level 2 - Urgent", "Level 3 - Routine clinical review", "Level 4 - Insufficient data".
- priority_code must be 1, 2, 3, or 4.
- red_flags and what_to_do_now and questions_for_doctor are arrays of short strings.
- medication_guidance must NOT prescribe, dose, or start individualized prescription medicines. It should instead state safe next-step guidance and that prescription decisions belong to a clinician.
- If emergency warning signs are present, say so clearly and prioritize emergency evaluation.
- Distinguish symptoms from diagnoses and state uncertainty where appropriate.
Patient age: {age if age is not None else 'not provided'}
Patient gender: {gender or 'not provided'}
Symptom narrative: {transcript[:8000]}
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "tools": [{"type": "web_search"}],
        "input": prompt,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))

    raw_text, sources = _extract_response_text(data)
    report = _parse_json_object(raw_text)
    return report, sources


async def analyze_symptoms(text: str, age: Optional[int] = None, gender: Optional[str] = None) -> Dict[str, Any]:
    safety = deterministic_safety_screen(text, age)

    if not OPENAI_API_KEY:
        return _sanitize_ai_report(
            {
                "clinical_summary": "AI web research is not connected yet. A deterministic safety screen is shown so the demo remains functional.",
            },
            safety,
            [],
            "Local safety fallback",
        )

    try:
        report, sources = _call_openai_web_search(text, age, gender)
        return _sanitize_ai_report(report, safety, sources, "Live web research")
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"OpenAI web-search analysis failed: {exc}")
        return _sanitize_ai_report(
            {
                "clinical_summary": "Live AI research was temporarily unavailable. The emergency safety screen below is shown instead.",
            },
            safety,
            [],
            "Safety fallback after AI error",
        )
    except Exception as exc:
        print(f"Unexpected AI analysis error: {exc}")
        return _sanitize_ai_report(
            {
                "clinical_summary": "The AI service could not be reached. The deterministic safety screen is shown instead.",
            },
            safety,
            [],
            "Safety fallback after AI error",
        )

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/api/sys/meta")
async def get_sys_meta():
    doctors = []
    for k, v in CENTRAL_DB["users"].items():
        if v["role"] == "doctor":
            rating = get_doctor_rating(k)
            doctors.append({
                "id": k,
                "name": v["name"],
                "dept": v.get("dept"),
                "hospital_id": v.get("hospital_id"),
                "rating_average": rating["average"],
                "rating_count": rating["count"],
            })
    hospitals = list(CENTRAL_DB["hospitals"].values())
    meds = list(CENTRAL_DB["medicines"].values())
    return {"doctors": doctors, "hospitals": hospitals, "medicines": meds, "ai_live": bool(OPENAI_API_KEY)}

@app.post("/api/auth/login")
async def login(req: AuthReq):
    user = CENTRAL_DB["users"].get(req.username)
    if not user or user["password"] != req.password:
        return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid credentials"})
    user_data = {k: v for k, v in user.items() if k != "password"}
    user_data["username"] = req.username
    return {"status": "success", "user": user_data}

@app.post("/api/auth/register-patient")
async def register_patient(req: PatRegReq):
    if req.username in CENTRAL_DB["users"]:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Username/ID already exists."})
    CENTRAL_DB["users"][req.username] = {
        "role": "patient", "password": req.password, "name": req.name, 
        "age": req.age, "gender": req.gender, "blood_group": req.blood_group, "profile_pic": None
    }
    return {"status": "success", "message": "Patient registered successfully. You can now login."}

@app.post("/api/user/update")
async def update_profile(req: ProfileUpdateReq):
    if req.user_id not in CENTRAL_DB["users"]:
        return JSONResponse(status_code=404, content={"status": "error", "message": "User not found."})
    
    user = CENTRAL_DB["users"][req.user_id]
    if req.new_password:
        user["password"] = req.new_password
    if req.profile_pic is not None:
        user["profile_pic"] = req.profile_pic if req.profile_pic else None
        
    for k, v in req.updates.items():
        if k in ["name", "age", "gender", "blood_group", "allergies"]:
            user[k] = v
            
    user_data = {k: v for k, v in user.items() if k != "password"}
    user_data["username"] = req.user_id
    return {"status": "success", "message": "Profile updated successfully.", "user": user_data}

def _is_admin(admin_id: Optional[str]) -> bool:
    user = CENTRAL_DB["users"].get(admin_id or "")
    return bool(user and user.get("role") == "admin")


@app.post("/api/admin/manage-hospital")
async def manage_hosp(req: HospReq):
    if not _is_admin(req.admin_id):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})
    if req.action == "create":
        h_id = f"H-{len(CENTRAL_DB['hospitals']) + 1:02d}"
        CENTRAL_DB["hospitals"][h_id] = {"id": h_id, "name": req.name, "location": req.location}
        return {"status": "success", "message": f"Hospital '{req.name}' added."}
    elif req.action == "edit" and req.id in CENTRAL_DB["hospitals"]:
        CENTRAL_DB["hospitals"][req.id]["name"] = req.name
        CENTRAL_DB["hospitals"][req.id]["location"] = req.location
        return {"status": "success", "message": "Hospital updated."}
    elif req.action == "delete" and req.id in CENTRAL_DB["hospitals"]:
        del CENTRAL_DB["hospitals"][req.id]
        return {"status": "success", "message": "Hospital deleted."}
    return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid operation."})

@app.post("/api/admin/manage-medicine")
async def manage_med(req: MedReq):
    if not _is_admin(req.admin_id):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})
    if req.action == "create":
        CENTRAL_DB["medicines"][req.code] = {"code": req.code, "name": req.name}
        return {"status": "success", "message": f"Medicine '{req.name}' added."}
    elif req.action == "delete" and req.code in CENTRAL_DB["medicines"]:
        del CENTRAL_DB["medicines"][req.code]
        return {"status": "success", "message": "Medicine deleted."}
    return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid operation."})

@app.post("/api/admin/manage-staff")
async def manage_staff(req: StaffReq):
    if not _is_admin(req.admin_id):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})
    if req.action == "create":
        u_id = f"DOC-{str(uuid.uuid4().int)[:4]}" if req.role == "doctor" else f"ADM-{str(uuid.uuid4().int)[:4]}"
        CENTRAL_DB["users"][u_id] = {"role": req.role, "password": req.new_password, "name": req.name, "profile_pic": None}
        if req.role == "doctor":
            CENTRAL_DB["users"][u_id].update({"dept": req.dept, "hospital_id": req.hospital_id, "consults": 0})
        return {"status": "success", "message": f"Staff created. ID: {u_id}"}
    elif req.action == "delete" and req.user_id in CENTRAL_DB["users"]:
        del CENTRAL_DB["users"][req.user_id]
        return {"status": "success", "message": "Staff deleted."}
    return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid operation."})

@app.post("/api/admin/search")
async def admin_search(filters: dict):
    if not _is_admin(filters.get("admin_id")):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})
    results = []
    for case in CENTRAL_DB["cases"].values():
        if filters.get("date") and case["date_str"] != filters["date"]: continue
        if filters.get("hospital_id") and case["hospital_id"] != filters["hospital_id"]: continue
        if filters.get("doc_id") and case["doc_id"] != filters["doc_id"]: continue
        if filters.get("abha_id") and case["abha_id"] != filters["abha_id"]: continue
        if filters.get("dept") and case["dept"].lower() != filters["dept"].lower(): continue
        results.append(case)
    return {"status": "success", "data": results}

@app.get("/api/admin/stats/{stype}/{sid}")
async def get_stats(stype: str, sid: str, admin_id: Optional[str] = None):
    if not _is_admin(admin_id):
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})
    count = 0
    for c in CENTRAL_DB["cases"].values():
        if stype == "doc" and c["doc_id"] == sid: count += 1
        elif stype == "hosp" and c["hospital_id"] == sid: count += 1
    return {"status": "success", "count": count}


@app.post("/api/ai/analyze")
async def api_ai_analyze(req: AIAnalysisReq):
    if not req.transcript.strip():
        return JSONResponse(status_code=400, content={"status": "error", "message": "Symptoms / patient intake cannot be empty."})
    report = await analyze_symptoms(req.transcript.strip(), req.age, req.gender)
    return {"status": "success", "report": report}


@app.get("/api/doctor/{doctor_id}/rating")
async def doctor_rating(doctor_id: str):
    doctor = CENTRAL_DB["users"].get(doctor_id)
    if not doctor or doctor.get("role") != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found")
    return {"status": "success", "doctor_id": doctor_id, **get_doctor_rating(doctor_id)}


@app.get("/api/admin/ratings")
async def admin_ratings(admin_id: str):
    admin = CENTRAL_DB["users"].get(admin_id)
    if not admin or admin.get("role") != "admin":
        return JSONResponse(status_code=403, content={"status": "error", "message": "Admin access required."})

    aggregates = []
    for doctor_id, user in CENTRAL_DB["users"].items():
        if user.get("role") == "doctor":
            rating = get_doctor_rating(doctor_id)
            aggregates.append({
                "doctor_id": doctor_id,
                "doctor_name": user.get("name"),
                "dept": user.get("dept"),
                "hospital_id": user.get("hospital_id"),
                **rating,
            })
    aggregates.sort(key=lambda x: (-x["average"], -x["count"], x["doctor_name"] or ""))
    return {"status": "success", "aggregates": aggregates, "reviews": list_doctor_reviews()}


@app.post("/api/patient/review")
async def patient_review(req: ReviewReq):
    if req.rating < 1 or req.rating > 5:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Rating must be between 1 and 5."})
    patient = CENTRAL_DB["users"].get(req.patient_id)
    if not patient or patient.get("role") != "patient":
        return JSONResponse(status_code=403, content={"status": "error", "message": "Valid patient account required."})
    case = CENTRAL_DB["cases"].get(req.case_id)
    if not case or case.get("abha_id") != req.patient_id:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Completed patient case not found."})
    if case.get("status") != "completed":
        return JSONResponse(status_code=400, content={"status": "error", "message": "Reviews are available after the doctor completes the case."})
    if case.get("doc_id") != req.doctor_id:
        return JSONResponse(status_code=403, content={"status": "error", "message": "Doctor does not match this case."})

    try:
        review = save_doctor_review(req.case_id, req.patient_id, req.doctor_id, req.rating, req.feedback)
    except ValueError as exc:
        return JSONResponse(status_code=409, content={"status": "error", "message": str(exc)})

    return {"status": "success", "message": "Thank you. Your review has been saved.", "review": review, "doctor_rating": get_doctor_rating(req.doctor_id)}

@app.post("/api/patient/intake")
async def patient_intake(req: IntakeReq):
    case_id = f"CASE-{str(uuid.uuid4())[:8].upper()}"
    patient = CENTRAL_DB["users"].get(req.abha_id)

    if not patient or patient.get("role") != "patient":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Patient not found"}
        )

    if not req.transcript.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Symptoms / patient intake cannot be empty."}
        )

    if req.is_followup:
        ref_case = CENTRAL_DB["cases"].get(req.ref_case_id)
        if not ref_case or ref_case.get("abha_id") != req.abha_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid Central Case ID for follow-up."}
            )
        hospital_id = ref_case["hospital_id"]
        dept = ref_case["dept"]
        doc_id = ref_case["doc_id"]
    else:
        hospital_id = req.hospital_id
        dept = req.dept
        doc_id = req.doc_id

    # Validate selected doctor/hospital before dictionary access.
    if not hospital_id or hospital_id not in CENTRAL_DB["hospitals"]:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid or missing hospital selection."}
        )

    if not doc_id or doc_id not in CENTRAL_DB["users"]:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid or missing doctor selection."}
        )

    doc = CENTRAL_DB["users"][doc_id]
    if doc.get("role") != "doctor":
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Selected account is not a doctor."}
        )

    if not dept:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Department is required."}
        )

    hosp = CENTRAL_DB["hospitals"][hospital_id]

    # FIX: save patient files before putting them into the case.
    # The old code used `patient_files` without defining it, causing the
    # NameError reported by VS Code and stopping intake submission.
    patient_files = save_attachments(req.photos, "patient", case_id)

    ai_report = await analyze_symptoms(req.transcript, patient.get("age"), patient.get("gender"))

    case = {
        "case_id": case_id,
        "abha_id": req.abha_id,
        "patient_name": patient["name"],
        "doc_id": doc_id,
        "doc_name": doc["name"],
        "hospital_id": hospital_id,
        "hospital_name": hosp["name"],
        "dept": dept,
        "transcript": req.transcript,
        "patient_photos": patient_files,
        "is_followup": req.is_followup,
        "ref_case_id": req.ref_case_id,
        "ai_report": ai_report,
        "emergency": ai_report.get("priority_code") == 1 or ai_report.get("risk") == "High",
        "review": None,
        "timestamp": time.time(),
        "date_str": time.strftime("%Y-%m-%d"),
        "status": "waiting",
        "prescription": None
    }

    CENTRAL_DB["cases"][case_id] = case
    return {"status": "success", "case_id": case_id}

@app.get("/api/cases/queue/{doc_id}")
async def get_queue(doc_id: str):
    queue = [c for c in CENTRAL_DB["cases"].values() if c["doc_id"] == doc_id and c["status"] == "waiting"]
    return {"queue": queue}

@app.get("/api/cases/history/{user_id}")
async def get_history(user_id: str, role: str):
    if role == "patient":
        records = [c for c in CENTRAL_DB["cases"].values() if c["abha_id"] == user_id]
    elif role == "doctor":
        records = [c for c in CENTRAL_DB["cases"].values() if c["doc_id"] == user_id and c["status"] == "completed" and c["date_str"] == time.strftime("%Y-%m-%d")]
    else:
        records = list(CENTRAL_DB["cases"].values())
    
    enriched = []
    for record in records:
        item = dict(record)
        item["review"] = get_case_review(item.get("case_id", ""))
        enriched.append(item)
    enriched.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"records": enriched}

@app.post("/api/doctor/prescribe")
async def prescribe(req: PrescribeReq):
    case = CENTRAL_DB["cases"].get(req.case_id)
    if not case:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Case not found"}
        )

    doctor = CENTRAL_DB["users"].get(req.doc_id)
    if not doctor or doctor.get("role") != "doctor":
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid doctor account."}
        )

    if case.get("doc_id") != req.doc_id or case.get("abha_id") != req.abha_id:
        return JSONResponse(
            status_code=403,
            content={"status": "error", "message": "This case is not assigned to this doctor/patient."}
        )

    med_details = [
        f"[{code}] {CENTRAL_DB['medicines'][code]['name']}"
        for code in req.medicine_codes
        if code in CENTRAL_DB["medicines"]
    ]

    doctor_files = save_attachments(req.attachments, "doctor", req.case_id)

    case["status"] = "completed"
    case["prescription"] = {
        "notes": req.clinical_notes,
        "medicines": med_details,
        "attachments": doctor_files,
        "date": time.strftime("%Y-%m-%d %H:%M")
    }

    doctor["consults"] = int(doctor.get("consults", 0)) + 1
    return {"status": "success", "message": "Prescription secured."}

# =====================================================================
# FRONTEND HTML
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en" class="light scroll-smooth">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CureCraft - Encrypted AI Medical Ecosystem</title>
  <link rel="icon" type="image/png" href="/static/logo.png">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: { extend: { colors: { brand: { 50: '#ecfdf5', 100: '#d1fae5', 500: '#10b981', 600: '#059669', 700: '#047857', dark: '#0b2e59' } } } }
    }
  </script>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; cursor: default; }
    .glass { background: rgba(255,255,255,0.85); backdrop-filter: blur(16px); border: 1px solid rgba(226,232,240,0.8); }
    .dark .glass { background: rgba(15,23,42,0.85); border: 1px solid rgba(51,65,85,0.5); }
    .page-view { display: none; opacity: 0; transition: opacity 0.3s ease-in-out; }
    .page-view.active { display: block; opacity: 1; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 10px; }
    .dark ::-webkit-scrollbar-thumb { background: #475569; }
    ::selection { background: #10b981; color: white; }

    /* DNA Animation */
    .dna-bg { position: relative; overflow: hidden; background: linear-gradient(135deg, #0b2e59, #0f172a, #042f2e); }
    .dna-strand {
        position: absolute; top: -50%; left: 50%; width: 200%; height: 200%;
        background: repeating-linear-gradient(45deg, transparent, transparent 40px, rgba(16, 185, 129, 0.04) 40px, rgba(16, 185, 129, 0.04) 80px);
        animation: rollingDna 45s linear infinite; transform-origin: center; pointer-events: none;
    }
    .dna-strand:nth-child(2) {
        background: repeating-linear-gradient(-45deg, transparent, transparent 50px, rgba(56, 189, 248, 0.03) 50px, rgba(56, 189, 248, 0.03) 100px);
        animation: rollingDnaReverse 55s linear infinite;
    }
    @keyframes rollingDna { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
    @keyframes rollingDnaReverse { 0% { transform: translate(-50%, -50%) rotate(360deg); } 100% { transform: translate(-50%, -50%) rotate(0deg); } }
    .login-container { min-height: calc(100vh - 80px); display: flex; align-items: center; justify-content: center; }
    #scroll-progress { position: fixed; top: 0; left: 0; height: 3px; width: 0; background: linear-gradient(90deg,#10b981,#38bdf8,#8b5cf6); z-index: 9999; box-shadow: 0 0 18px rgba(16,185,129,.55); }
    #cursor-glow { position: fixed; width: 260px; height: 260px; border-radius: 9999px; pointer-events: none; z-index: 1; transform: translate(-50%,-50%); background: radial-gradient(circle, rgba(16,185,129,.10), transparent 65%); mix-blend-mode: screen; }
    #ambient-canvas { position: fixed; inset: 0; width: 100%; height: 100%; pointer-events: none; opacity: .30; z-index: 0; }
    body > *:not(#ambient-canvas):not(#cursor-glow):not(#scroll-progress) { position: relative; z-index: 2; }
    .glass { box-shadow: 0 20px 45px rgba(15,23,42,.06); transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease; }
    .glass:hover { transform: translateY(-2px); box-shadow: 0 26px 65px rgba(15,23,42,.10); }
    .reveal { opacity: 0; transform: translateY(18px); transition: opacity .7s ease, transform .7s ease; }
    .reveal.revealed { opacity: 1; transform: translateY(0); }
    .ai-pulse { box-shadow: 0 0 0 0 rgba(16,185,129,.28); animation: aiPulse 2s infinite; }
    @keyframes aiPulse { 0% { box-shadow: 0 0 0 0 rgba(16,185,129,.26) } 70% { box-shadow: 0 0 0 12px rgba(16,185,129,0) } 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0) } }
    .star-btn { font-size: 1.7rem; line-height: 1; transition: transform .16s ease, color .16s ease; color: #cbd5e1; }
    .star-btn:hover { transform: translateY(-2px) scale(1.08); color: #f59e0b; }
    .star-btn.active { color: #f59e0b; }
    .file-chip { animation: chipIn .22s ease both; }
    @keyframes chipIn { from { opacity: 0; transform: scale(.96) } to { opacity: 1; transform: scale(1) } }
    .shimmer { position: relative; overflow: hidden; }
    .shimmer::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%); background: linear-gradient(90deg, transparent, rgba(255,255,255,.15), transparent); animation: shimmer 2.4s infinite; pointer-events:none; }
    @keyframes shimmer { to { transform: translateX(120%); } }
    @media (pointer: coarse) { #cursor-glow { display:none; } }
  </style>
</head>
<body class="bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 min-h-screen flex flex-col transition-colors duration-300">
  <div id="scroll-progress"></div>
  <div id="cursor-glow"></div>
  <canvas id="ambient-canvas" aria-hidden="true"></canvas>

  <!-- HEADER NAVIGATION -->
  <header class="glass sticky top-0 z-50 px-6 py-3 flex items-center justify-between shadow-sm">
    <div class="flex items-center space-x-3 cursor-pointer" onclick="handleLogoClick()">
      <img src="/static/logo.png" alt="CureCraft Logo" class="h-10 w-auto object-contain" onerror="this.onerror=null; this.src='https://via.placeholder.com/120x40?text=CureCraft';">
      <div class="border-l-2 border-slate-300 dark:border-slate-700 pl-3">
        <h1 class="text-lg font-black text-brand-dark dark:text-white leading-none">CureCraft</h1>
        <p class="text-[10px] font-bold text-brand-600 uppercase tracking-wider flex items-center gap-1"><i data-lucide="lock" class="w-3 h-3"></i> Secure prototype</p>
      </div>
    </div>
    
    <div class="flex items-center space-x-4">
      <button onclick="toggleTheme()" class="p-2 rounded-xl bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-yellow-400 hover:bg-slate-300 transition">
        <i data-lucide="sun" class="w-4 h-4 hidden dark:block"></i>
        <i data-lucide="moon" class="w-4 h-4 block dark:hidden"></i>
      </button>
      
      <div id="user-badge" class="hidden items-center space-x-3 cursor-pointer bg-slate-100 dark:bg-slate-800 p-1.5 pr-4 rounded-full border border-slate-200 dark:border-slate-700 hover:shadow-md transition" onclick="openProfileModal()">
        <div class="w-8 h-8 rounded-full bg-brand-500 text-white flex items-center justify-center font-bold text-sm shadow-inner overflow-hidden" id="nav-initial-container">
           <span id="nav-initial">U</span>
        </div>
        <div class="text-right">
          <p id="nav-user-name" class="text-xs font-bold leading-tight">User</p>
          <p id="nav-user-role" class="text-[10px] text-brand-600 dark:text-brand-400 font-mono uppercase font-semibold">Role</p>
        </div>
      </div>
      
      <button id="logout-btn" onclick="logout()" class="hidden p-2 rounded-xl bg-rose-100 text-rose-600 hover:bg-rose-200 transition" title="Logout">
        <i data-lucide="log-out" class="w-4 h-4"></i>
      </button>
    </div>
  </header>

  <!-- MAIN VIEWPORT CONTAINER -->
  <main class="flex-1 w-full max-w-7xl mx-auto px-4 py-6">

    <!-- ========================================== -->
    <!-- VIEW: HOME PAGE -->
    <!-- ========================================== -->
    <section id="view-home" class="page-view active space-y-12 pt-4">
      <div class="dna-bg relative overflow-hidden rounded-3xl text-white p-8 md:p-14 shadow-2xl border border-slate-800">
        <div class="dna-strand"></div><div class="dna-strand"></div>
        <div class="max-w-3xl relative z-10 space-y-6">
          <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/20 text-brand-300 text-xs font-semibold border border-brand-500/30 backdrop-blur-md">
            <i data-lucide="network" class="w-3.5 h-3.5"></i> Centralized Hospital Network Live
          </div>
          <h1 class="text-4xl md:text-6xl font-black leading-tight drop-shadow-lg">Your History.<br><span class="text-brand-400">Our AI. Better Care.</span></h1>
          <p class="text-slate-300 text-sm md:text-base drop-shadow max-w-2xl">Real-time multilingual voice triage, instant doctor-patient queuing, strict admin-controlled medicine protocols, and automated e-prescriptions integrated into one secure ecosystem.</p>
          <div class="flex flex-wrap gap-3 pt-4">
            <button onclick="openLogin('patient')" class="px-6 py-3 bg-brand-600 hover:bg-brand-500 text-white font-bold rounded-xl shadow-lg flex items-center gap-2 transition transform hover:scale-105"><i data-lucide="user"></i> Patient Portal</button>
            <button onclick="openLogin('doctor')" class="px-6 py-3 bg-white/10 hover:bg-white/20 text-white border border-white/20 font-bold rounded-xl flex items-center gap-2 transition backdrop-blur-md hover:scale-105"><i data-lucide="stethoscope"></i> Doctor Station</button>
            <button onclick="openLogin('admin')" class="px-6 py-3 bg-white/10 hover:bg-white/20 text-white border border-white/20 font-bold rounded-xl flex items-center gap-2 transition backdrop-blur-md hover:scale-105"><i data-lucide="shield"></i> Admin Portal</button>
          </div>
        </div>
      </div>

      <div class="glass p-8 rounded-3xl shadow-lg border border-brand-500/30 relative">
        <div class="flex items-center gap-3 mb-6 relative z-10">
          <div class="w-10 h-10 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center font-bold shadow-inner"><i data-lucide="bot"></i></div>
          <div><h3 class="text-xl font-bold">Interactive AI Symptom Simulator</h3><p class="text-xs text-slate-500">Live clinical-triage prototype with web-grounded research when an API key is configured.</p></div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
          <div class="flex flex-col">
            <label class="block text-xs font-bold uppercase text-slate-500 mb-2">Simulate Patient Input</label>
            <textarea id="demo-input" rows="4" class="w-full p-4 rounded-2xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none focus:ring-2 focus:ring-brand-500 transition flex-1" placeholder="Type here (e.g., Severe chest pain vs I am feeling fine)..."></textarea>
            <button onclick="runDemoAi()" class="ai-pulse mt-4 w-full px-5 py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm rounded-xl transition shadow flex justify-center items-center gap-2"><i data-lucide="sparkles" class="w-4 h-4"></i> Run Live AI Triage</button>
            <div class="mt-3 flex items-center gap-2 text-[10px] text-slate-500"><span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span><span id="ai-connection-label">Checking AI connection…</span></div>
          </div>
          <div class="bg-slate-900 rounded-2xl border border-slate-700 flex flex-col overflow-hidden shadow-inner h-full min-h-[200px]">
            <div class="bg-slate-800 p-3 border-b border-slate-700 flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-rose-500"></div><div class="w-2 h-2 rounded-full bg-yellow-500"></div><div class="w-2 h-2 rounded-full bg-green-500"></div>
              <span class="text-[10px] font-bold text-slate-400 ml-2 uppercase">AI Diagnostic Engine</span>
            </div>
            <div class="p-5 overflow-y-auto flex-1 flex flex-col gap-3" id="demo-output">
              <div class="text-xs text-slate-500 italic text-center mt-10">Describe symptoms to start the analysis engine.</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- VIEW: AUTHENTICATION -->
    <!-- ========================================== -->
    <section id="view-auth" class="page-view login-container">
      <div class="glass p-8 rounded-3xl shadow-xl w-full max-w-md relative border-t-4 border-t-brand-500">
        <button onclick="navigate('home')" class="absolute top-4 left-4 p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 transition"><i data-lucide="arrow-left" class="w-5 h-5"></i></button>
        <h2 id="auth-title" class="text-2xl font-black text-center mb-1 mt-2">Secure Login</h2>
        <p class="text-center text-xs text-slate-500 mb-6 flex justify-center items-center gap-1"><i data-lucide="shield-check" class="w-3 h-3 text-brand-500"></i> Secure prototype</p>
        
        <div id="login-form" class="space-y-4">
          <input type="text" id="auth-user" placeholder="Network ID / Username" onkeypress="if(event.key === 'Enter') document.getElementById('auth-pass').focus()" class="w-full p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none focus:ring-2 focus:ring-brand-500 transition shadow-inner">
          <input type="password" id="auth-pass" placeholder="Passkey" onkeypress="if(event.key === 'Enter') executeLogin()" class="w-full p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none focus:ring-2 focus:ring-brand-500 transition shadow-inner">
          <button onclick="executeLogin()" class="w-full py-3.5 mt-2 bg-brand-600 hover:bg-brand-700 text-white font-black rounded-xl shadow-lg transition transform hover:-translate-y-0.5">Authenticate</button>
          <div class="text-center mt-3" id="register-link-container">
            <a href="#" onclick="toggleAuthMode(true)" class="text-xs text-brand-600 font-bold hover:underline">New Patient? Register Here</a>
          </div>
        </div>

        <div id="register-form" class="space-y-3 hidden">
          <input type="text" id="reg-id" placeholder="Create Patient ID (e.g. ABHA-1234)" class="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none">
          <input type="text" id="reg-name" placeholder="Full Name" class="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none">
          <div class="flex gap-2">
            <input type="number" id="reg-age" placeholder="Age" class="w-1/3 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none">
            <select id="reg-gender" class="w-1/3 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none"><option value="Male">Male</option><option value="Female">Female</option><option value="Other">Other</option></select>
            <select id="reg-bg" class="w-1/3 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none"><option value="O+">O+</option><option value="A+">A+</option><option value="B+">B+</option><option value="AB+">AB+</option><option value="O-">O-</option></select>
          </div>
          <input type="password" id="reg-pass" placeholder="Create Passkey" class="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none">
          <button onclick="executeRegister()" class="w-full py-3 mt-2 bg-slate-800 dark:bg-white text-white dark:text-slate-900 font-black rounded-xl shadow-lg transition">Register</button>
          <div class="text-center mt-2">
            <a href="#" onclick="toggleAuthMode(false)" class="text-xs text-slate-500 font-bold hover:underline">Back to Login</a>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- VIEW: PATIENT -->
    <!-- ========================================== -->
    <section id="view-patient" class="page-view space-y-6 pt-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="md:col-span-2 space-y-6">
          <div class="grid grid-cols-2 gap-4">
            <button onclick="toggleIntake(false)" class="glass p-5 rounded-2xl shadow-sm flex flex-col md:flex-row items-center gap-4 hover:border-brand-500 transition group">
              <div class="w-12 h-12 rounded-xl bg-brand-100 text-brand-600 flex items-center justify-center font-bold group-hover:scale-110 transition"><i data-lucide="plus" class="w-6 h-6"></i></div>
              <div class="text-left"><p class="font-bold text-sm">New Consultation</p><p class="text-xs text-slate-500">Book AI triage</p></div>
            </button>
            <button onclick="toggleIntake(true)" class="glass p-5 rounded-2xl shadow-sm flex flex-col md:flex-row items-center gap-4 hover:border-indigo-500 transition group">
              <div class="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold group-hover:scale-110 transition"><i data-lucide="refresh-cw" class="w-6 h-6"></i></div>
              <div class="text-left"><p class="font-bold text-sm">Follow-up Care</p><p class="text-xs text-slate-500">Provide Case ID</p></div>
            </button>
          </div>

          <!-- INTAKE FORM -->
          <div id="patient-intake-form" class="hidden glass p-6 rounded-3xl shadow-lg border border-brand-500/30 space-y-4">
            <h3 id="intake-title" class="font-bold text-lg text-brand-600 flex items-center gap-2"><i data-lucide="activity"></i> New Consultation</h3>
            
            <div id="new-consult-fields" class="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Select Hospital</label>
                <select id="intake-hosp" onchange="filterDepts()" class="w-full p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner"></select>
              </div>
              <div>
                <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Department</label>
                <select id="intake-dept" onchange="filterDoctors()" class="w-full p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner"></select>
              </div>
              <div>
                <label class="block text-[10px] font-bold text-slate-500 uppercase mb-1">Specialist</label>
                <select id="intake-doc" onchange="showSelectedDoctorRating()" class="w-full p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner"></select>
                <div id="doctor-rating-preview" class="mt-2 hidden rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-[10px]"></div>
              </div>
            </div>
            
            <div id="followup-fields" class="hidden">
               <label class="block text-[10px] font-bold text-indigo-500 uppercase mb-1">Central Case ID</label>
               <input type="text" id="intake-ref-case" placeholder="Enter valid Case ID (e.g. CASE-ABC12345)" class="w-full p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner font-mono">
            </div>

            <div>
              <div class="flex justify-between items-end mb-1">
                 <label class="block text-xs font-bold text-slate-500">Symptoms / Voice Intake</label>
                 <select id="voice-lang" class="text-[10px] p-1 rounded border border-slate-200 outline-none"><option value="en-IN">English</option><option value="hi-IN">Hindi</option></select>
              </div>
              <div class="relative">
                <textarea id="intake-transcript" rows="3" class="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none pr-12 shadow-inner" placeholder="Describe symptoms or click the microphone..."></textarea>
                <button onclick="startVoiceRecognition()" id="mic-btn" class="absolute right-3 top-3 p-2 rounded-xl bg-brand-100 text-brand-600 hover:bg-brand-600 hover:text-white transition shadow"><i data-lucide="mic" class="w-4 h-4"></i></button>
              </div>
            </div>

            <div>
              <label class="block text-xs font-bold text-slate-500 mb-1">Upload Medical Reports / Files</label>
              <input type="file" id="intake-photos" multiple class="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:font-bold file:bg-slate-200 file:text-slate-700" onchange="syncPatientFiles(this)">
              <div class="flex justify-between items-center mt-2"><p class="text-[10px] text-slate-500">Any file format • up to 25 MB each • multiple files supported</p><button type="button" onclick="clearPatientFiles()" class="text-[10px] font-bold text-rose-500 hover:underline">Remove all</button></div>
              <div id="intake-file-list" class="mt-2 flex flex-wrap gap-2"></div>
            </div>

            <div class="flex gap-3">
              <button onclick="submitIntake()" class="flex-1 py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-xl shadow-lg transition">Submit to Network</button>
              <button onclick="document.getElementById('patient-intake-form').classList.add('hidden')" class="px-5 py-3 bg-slate-200 dark:bg-slate-800 font-bold rounded-xl text-sm transition">Cancel</button>
            </div>
          </div>

          <!-- RECORDS -->
          <div class="glass p-6 rounded-3xl shadow-sm border-t-4 border-t-brand-500">
            <div class="flex justify-between items-center mb-4">
              <h3 class="font-bold text-base flex items-center gap-2"><i data-lucide="folder-open" class="text-brand-500"></i> Centralized Records</h3>
              <button onclick="loadPatientHistory()" class="px-3 py-1.5 bg-brand-100 text-brand-700 hover:bg-brand-200 rounded-lg text-xs font-bold flex items-center gap-1 transition"><i data-lucide="refresh-cw" class="w-3 h-3"></i> Refresh</button>
            </div>
            <div id="patient-history-list" class="space-y-4"></div>
          </div>
        </div>

        <div class="md:col-span-1 space-y-6">
           <div class="glass p-6 rounded-3xl shadow-sm text-center border border-brand-200">
             <div class="w-16 h-16 mx-auto bg-green-100 text-green-600 rounded-full flex items-center justify-center mb-3 shadow-inner"><i data-lucide="shield-check" class="w-8 h-8"></i></div>
             <h4 class="font-black text-brand-600">Privacy-first demo mode</h4>
             <p class="text-xs text-slate-500 mt-1">Prototype data is stored locally for the hackathon demo.</p>
           </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- VIEW: DOCTOR -->
    <!-- ========================================== -->
    <section id="view-doctor" class="page-view space-y-6 pt-4">
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div class="lg:col-span-1 space-y-4">
          <div class="glass p-5 rounded-3xl shadow-sm border-t-4 border-t-indigo-500 relative overflow-hidden">
            <h2 id="doc-prof-name" class="font-bold text-lg relative z-10">Dr. Name</h2>
            <p class="text-xs text-indigo-600 dark:text-indigo-400 font-bold mb-1 relative z-10"><span id="doc-prof-dept"></span> &bull; <span id="doc-prof-hosp"></span></p><div id="doc-prof-rating" class="text-sm mb-3 relative z-10"></div>
            <div class="bg-slate-100 dark:bg-slate-900 p-3 rounded-xl text-center shadow-inner relative z-10 flex gap-2">
              <div class="flex-1 cursor-pointer hover:bg-slate-200 dark:hover:bg-slate-800 rounded p-1 transition" onclick="showDocHistory()">
                <span class="text-[9px] text-slate-500 uppercase font-bold block">Today's Consults</span>
                <span id="doc-prof-consults" class="text-xl font-black text-slate-800 dark:text-white">0</span>
              </div>
            </div>
          </div>

          <div class="glass p-4 rounded-3xl shadow-sm flex flex-col h-[500px]">
            <div class="flex justify-between items-center mb-3 px-1">
              <h3 class="font-bold text-xs uppercase tracking-wider text-slate-500 flex items-center gap-1"><i data-lucide="users" class="w-3 h-3"></i> Live Queue</h3>
              <button onclick="loadDoctorQueue()" class="text-indigo-500 hover:rotate-180 transition duration-500"><i data-lucide="refresh-cw" class="w-4 h-4"></i></button>
            </div>
            <div id="doc-queue-list" class="space-y-2 flex-1 overflow-y-auto pr-1"></div>
          </div>
        </div>

        <div class="lg:col-span-3 h-full flex flex-col">
          <div id="doc-active-case" class="hidden glass p-6 md:p-8 rounded-3xl shadow-lg border border-slate-200 dark:border-slate-800 flex-1 relative flex flex-col">
            <div class="flex justify-between items-start border-b border-slate-200 dark:border-slate-800 pb-4 relative z-10">
              <div>
                <div class="flex items-center gap-2 mb-1" id="case-badges"></div>
                <h2 id="case-pat-name" class="text-2xl font-black text-indigo-600 dark:text-indigo-400">Patient Name</h2>
                <p id="case-pat-id" class="text-xs text-slate-500 font-mono">Patient ID</p>
              </div>
              <span id="case-id" class="px-3 py-1.5 bg-slate-100 dark:bg-slate-900 rounded-lg text-xs font-mono font-bold border border-slate-200 dark:border-slate-700">CASE-ID</span>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 relative z-10 mt-4">
              <div class="p-4 bg-slate-50 dark:bg-slate-900/80 rounded-2xl shadow-inner border border-slate-200 dark:border-slate-700">
                <h4 class="text-[10px] font-bold text-slate-500 uppercase mb-2">Patient Intake</h4>
                <p id="case-transcript" class="text-sm italic"></p>
              </div>
              <div class="p-4 bg-slate-50 dark:bg-slate-900/80 rounded-2xl shadow-inner border border-slate-200 dark:border-slate-700">
                <h4 class="text-[10px] font-bold text-slate-500 uppercase mb-2">AI Findings</h4>
                <div id="case-hpi" class="space-y-1 text-xs"></div>
              </div>
            </div>

            <div id="case-photos-container" class="hidden relative z-10 mt-4">
              <h4 class="text-[10px] font-bold text-slate-500 uppercase mb-2">Patient Attachments</h4>
              <div id="case-photos" class="flex gap-3 overflow-x-auto pb-2"></div>
            </div>

            <div class="mt-auto pt-5 space-y-4 relative z-10">
              <h3 class="font-bold text-base flex items-center gap-2"><i data-lucide="file-signature" class="text-teal-500"></i> Issue Encrypted E-Prescription</h3>
              <textarea id="presc-notes" rows="2" class="w-full p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner" placeholder="Clinical notes..."></textarea>
              
              <div class="grid grid-cols-2 gap-4">
                 <div class="p-3 rounded-xl border border-slate-200 bg-white dark:bg-slate-950">
                    <label class="block text-[10px] font-bold text-rose-500 uppercase mb-2">Strict DB Medicines</label>
                    <div class="flex gap-2">
                      <select id="presc-med-select" class="flex-1 p-2 rounded-lg bg-slate-50 dark:bg-slate-900 text-xs outline-none border border-slate-200 dark:border-slate-700"></select>
                      <button onclick="addMedicine()" class="px-3 bg-slate-800 text-white font-bold rounded-lg text-xs">Add</button>
                    </div>
                    <ul id="presc-med-list" class="space-y-1 mt-2"></ul>
                 </div>
                 <div class="p-3 rounded-xl border border-slate-200 bg-white dark:bg-slate-950">
                    <label class="block text-[10px] font-bold text-indigo-500 uppercase mb-2">Doctor Attachments</label>
                    <input type="file" id="presc-attach" multiple class="block w-full text-[10px] text-slate-500 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:font-bold file:bg-slate-200 file:text-slate-700" onchange="syncDoctorFiles(this)">
                     <div class="flex justify-between items-center mt-1"><span class="text-[9px] text-slate-500">Any format • 25 MB each</span><button type="button" onclick="clearDoctorFiles()" class="text-[9px] text-rose-500 font-bold hover:underline">Remove all</button></div>
                     <div id="doctor-file-list" class="mt-2 flex flex-wrap gap-2"></div>
                 </div>
              </div>

              <button onclick="submitPrescription()" class="w-full py-3 bg-teal-600 hover:bg-teal-700 text-white font-bold rounded-xl shadow-lg transition">Sign & Forward to Central DB</button>
            </div>
          </div>

          <div id="doc-idle" class="glass h-[500px] flex flex-col items-center justify-center rounded-3xl text-slate-400 p-8 text-center border-dashed border-2 border-slate-300 transition">
            <i data-lucide="inbox" class="w-16 h-16 mb-4 opacity-30"></i>
            <h3 class="text-lg font-bold text-slate-600 dark:text-slate-300 mb-1">Waiting for Patient</h3>
            <p class="text-sm">Select a patient from the queue.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- VIEW: ADMIN DASHBOARD -->
    <!-- ========================================== -->
    <section id="view-admin" class="page-view space-y-6 pt-4">
      <div class="glass p-6 rounded-3xl shadow-sm border-l-4 border-l-rose-500 flex justify-between items-center bg-gradient-to-r from-transparent to-rose-500/5">
        <div>
          <h2 class="text-xl font-black">System Administration Console</h2>
          <p class="text-xs text-slate-500 font-mono mt-1">Global DB Management</p>
        </div>
        <i data-lucide="shield-alert" class="w-8 h-8 text-rose-500"></i>
      </div>
      
      <!-- ADMIN TABS -->
      <div class="flex gap-2 overflow-x-auto pb-2 border-b border-slate-200 dark:border-slate-800">
         <button onclick="switchAdminTab('hospital')" id="atab-hospital" class="admin-tab px-4 py-2 font-bold text-sm border-b-2 border-brand-500 text-brand-600">Hospitals</button>
         <button onclick="switchAdminTab('staff')" id="atab-staff" class="admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500">Staff</button>
         <button onclick="switchAdminTab('medicine')" id="atab-medicine" class="admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500">Medicines</button>
         <button onclick="switchAdminTab('analytics')" id="atab-analytics" class="admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500">Global DB Search</button>
         <button onclick="switchAdminTab('ratings')" id="atab-ratings" class="admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500">Doctor Ratings</button>
      </div>

      <div class="glass p-6 rounded-3xl shadow-sm min-h-[400px]">
         <!-- HOSPITAL MGMT -->
         <div id="apanel-hospital" class="space-y-4">
            <h3 class="font-bold border-b border-slate-200 dark:border-slate-800 pb-2 mb-4"><i data-lucide="building" class="inline w-4 h-4 mr-2"></i> Manage Hospitals</h3>
            <div class="flex gap-2">
                <input type="text" id="ahosp-id" placeholder="Hosp ID (Leave blank to Create)" class="w-1/4 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 text-sm outline-none">
                <input type="text" id="ahosp-name" placeholder="Hospital Name" class="w-1/2 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 text-sm outline-none">
                <input type="text" id="ahosp-loc" placeholder="Location" class="w-1/4 p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 text-sm outline-none">
            </div>
            <div class="flex gap-2">
                <button onclick="manageHospital('create')" class="px-4 py-2 bg-brand-600 text-white font-bold rounded-lg text-sm">Create</button>
                <button onclick="manageHospital('edit')" class="px-4 py-2 bg-yellow-500 text-white font-bold rounded-lg text-sm">Edit</button>
                <button onclick="manageHospital('delete')" class="px-4 py-2 bg-rose-600 text-white font-bold rounded-lg text-sm">Delete</button>
            </div>
         </div>

         <!-- STAFF MGMT -->
         <div id="apanel-staff" class="space-y-4 hidden">
            <h3 class="font-bold border-b border-slate-200 dark:border-slate-800 pb-2 mb-4"><i data-lucide="user-plus" class="inline w-4 h-4 mr-2"></i> Manage Staff</h3>
            <div class="grid grid-cols-2 gap-4">
                <input type="text" id="astaff-id" placeholder="User ID (Leave blank to Create)" class="w-full p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                <select id="astaff-role" onchange="toggleAdminDept()" class="w-full p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                   <option value="doctor">Doctor</option>
                   <option value="admin">Admin</option>
                </select>
                <input type="text" id="astaff-name" placeholder="Full Name" class="w-full p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                <input type="password" id="astaff-pass" placeholder="Password" class="w-full p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                <div id="astaff-doc-fields" class="col-span-2 flex gap-4">
                   <input type="text" id="astaff-dept" placeholder="Department" class="w-1/2 p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                   <select id="astaff-hosp" class="w-1/2 p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none"></select>
                </div>
            </div>
            <div class="flex gap-2">
                <button onclick="manageStaff('create')" class="px-4 py-2 bg-brand-600 text-white font-bold rounded-lg text-sm">Create</button>
                <button onclick="manageStaff('delete')" class="px-4 py-2 bg-rose-600 text-white font-bold rounded-lg text-sm">Delete (Requires User ID)</button>
            </div>
         </div>

         <!-- MEDICINE MGMT -->
         <div id="apanel-medicine" class="space-y-4 hidden">
            <h3 class="font-bold border-b border-slate-200 dark:border-slate-800 pb-2 mb-4"><i data-lucide="pill" class="inline w-4 h-4 mr-2"></i> Manage Medicines</h3>
            <div class="flex gap-2">
                <input type="text" id="amed-code" placeholder="Code (e.g. M-109)" class="w-1/3 p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
                <input type="text" id="amed-name" placeholder="Medicine Name & Dosage" class="w-2/3 p-3 rounded-xl bg-slate-50 border border-slate-200 text-sm outline-none">
            </div>
            <div class="flex gap-2">
                <button onclick="manageMedicine('create')" class="px-4 py-2 bg-brand-600 text-white font-bold rounded-lg text-sm">Create</button>
                <button onclick="manageMedicine('delete')" class="px-4 py-2 bg-rose-600 text-white font-bold rounded-lg text-sm">Delete (By Code)</button>
            </div>
         </div>

         <!-- ANALYTICS / SEARCH ENGINE -->
         <div id="apanel-analytics" class="hidden flex flex-col h-full space-y-4">
            <div class="flex justify-between items-center border-b border-slate-200 dark:border-slate-800 pb-2">
               <h3 class="font-bold"><i data-lucide="search" class="inline w-4 h-4 mr-2 text-indigo-500"></i> Central Search Engine</h3>
               <button onclick="exportToCSV()" class="text-xs bg-green-100 text-green-700 px-3 py-1.5 rounded-lg font-bold hover:bg-green-200 transition flex items-center gap-1"><i data-lucide="download" class="w-3 h-3"></i> Export Excel/CSV</button>
            </div>
            <div class="grid grid-cols-5 gap-2">
                <input type="date" id="search-date" class="p-2 border border-slate-200 rounded text-xs outline-none">
                <input type="text" id="search-hosp" placeholder="Hospital ID" class="p-2 border border-slate-200 rounded text-xs outline-none">
                <input type="text" id="search-doc" placeholder="Doctor ID" class="p-2 border border-slate-200 rounded text-xs outline-none">
                <input type="text" id="search-pat" placeholder="Patient ID" class="p-2 border border-slate-200 rounded text-xs outline-none">
                <input type="text" id="search-dept" placeholder="Department" class="p-2 border border-slate-200 rounded text-xs outline-none">
            </div>
            <button onclick="searchDB()" class="w-full py-2 bg-indigo-600 text-white font-bold rounded-lg shadow text-sm">Search Records</button>
            <div class="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-4 overflow-y-auto max-h-[300px]" id="admin-search-results">
               <p class="text-xs text-slate-500 text-center mt-4">Enter filters and click search.</p>
            </div>
         </div>

         <!-- DOCTOR RATINGS -->
         <div id="apanel-ratings" class="hidden flex flex-col h-full space-y-4">
            <div class="flex justify-between items-center border-b border-slate-200 dark:border-slate-800 pb-2">
               <div><h3 class="font-bold"><i data-lucide="star" class="inline w-4 h-4 mr-2 text-amber-500"></i> Doctor Experience & Ratings</h3><p class="text-[10px] text-slate-500 mt-1">Live aggregate of submitted patient reviews stored in the local SQLite database.</p></div>
               <button onclick="loadAdminRatings()" class="text-xs bg-amber-100 text-amber-700 px-3 py-1.5 rounded-lg font-bold hover:bg-amber-200 transition">Refresh</button>
            </div>
            <div id="admin-rating-aggregates" class="grid grid-cols-1 md:grid-cols-2 gap-3"></div>
            <div class="border-t border-slate-200 dark:border-slate-800 pt-4">
              <h4 class="font-bold text-sm mb-3">Recent Patient Reviews</h4>
              <div id="admin-review-list" class="space-y-3 max-h-[300px] overflow-y-auto"></div>
            </div>
         </div>
      </div>
    </section>

  </main>

  <!-- ========================================== -->
  <!-- MODALS -->
  <!-- ========================================== -->
  
  <!-- Profile Modal -->
  <div id="profile-modal" class="hidden fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
    <div class="bg-white dark:bg-slate-900 rounded-3xl p-6 w-full max-w-sm shadow-2xl relative max-h-[90vh] overflow-y-auto">
      <button onclick="closeProfileModal()" class="absolute top-4 right-4 p-2 text-slate-500"><i data-lucide="x" class="w-5 h-5"></i></button>
      <div class="text-center border-b border-slate-200 pb-4 mb-4 mt-2">
        <div class="w-20 h-20 bg-brand-500 text-white rounded-2xl mx-auto flex items-center justify-center text-3xl font-black mb-3 overflow-hidden shadow-inner" id="modal-initial-container">
           <span id="modal-initial">U</span>
        </div>
        <h3 class="text-xl font-black" id="modal-name">Name</h3>
        <p class="text-xs font-mono text-brand-600 bg-brand-50 inline-block px-3 py-1 rounded-full mt-2" id="modal-id">ID</p>
      </div>
      <div id="modal-details" class="space-y-3 text-sm mb-4"></div>
      
      <div class="border-t border-slate-200 pt-4 space-y-3">
         <h4 class="font-bold text-xs uppercase text-slate-500">Edit Profile</h4>
         <input type="text" id="edit-name" placeholder="Full Name" class="w-full p-2 border border-slate-200 rounded text-sm outline-none">
         <input type="password" id="edit-pass" placeholder="New Password (Optional)" class="w-full p-2 border border-slate-200 rounded text-sm outline-none">
         <label class="block text-[10px] font-bold text-slate-500 uppercase">Profile Picture</label>
         <input type="file" id="edit-profile-pic" accept="image/*" class="w-full text-xs file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:bg-slate-200">
         <button onclick="updateProfile()" class="w-full py-2 bg-brand-600 text-white font-bold rounded-lg text-sm shadow">Save Changes</button>
      </div>
    </div>
  </div>

  <!-- Detailed Patient Case Modal -->
  <div id="case-detail-modal" class="hidden fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
    <div class="bg-white dark:bg-slate-900 rounded-3xl w-full max-w-3xl shadow-2xl relative max-h-[90vh] flex flex-col">
      <div class="p-4 border-b border-slate-200 flex justify-between items-center">
         <h3 class="text-lg font-black flex items-center gap-2"><i data-lucide="file-text" class="text-brand-500"></i> Case Report</h3>
         <div class="flex gap-2">
            <button onclick="downloadPDF()" class="bg-indigo-600 text-white px-3 py-1.5 rounded-lg text-xs font-bold shadow flex items-center gap-1"><i data-lucide="download" class="w-3 h-3"></i> Download PDF</button>
            <button onclick="document.getElementById('case-detail-modal').classList.add('hidden')" class="p-1.5 bg-slate-100 rounded-lg"><i data-lucide="x" class="w-4 h-4"></i></button>
         </div>
      </div>
      <div id="printable-report" class="p-6 overflow-y-auto flex-1 bg-white text-slate-800">
         <div class="flex items-center gap-4 border-b pb-4 mb-4">
            <img src="/static/logo.png" style="height:40px;">
            <div><h2 class="text-xl font-black text-brand-600">CureCraft Official Report</h2><p class="text-[10px] font-mono" id="pd-case-id"></p></div>
         </div>
         <div class="grid grid-cols-2 gap-4 mb-6 text-sm border border-slate-200 p-4 rounded-xl bg-slate-50">
            <div><span class="text-slate-500">Doctor:</span> <b id="pd-doc"></b></div>
            <div><span class="text-slate-500">Hospital:</span> <b id="pd-hosp"></b></div>
            <div><span class="text-slate-500">Patient:</span> <b id="pd-pat"></b></div>
            <div><span class="text-slate-500">Date:</span> <b id="pd-date"></b></div>
         </div>
         <div class="mb-4"><h4 class="font-bold border-b pb-1 mb-2 text-brand-600">1. Patient Elicitation</h4><p id="pd-trans" class="text-sm italic"></p></div>
         <div class="mb-4 bg-slate-50 p-3 rounded-lg border border-slate-200"><h4 class="font-bold border-b pb-1 mb-2 text-indigo-600">2. AI Diagnosis Engine</h4><div id="pd-ai" class="text-sm"></div></div>
         <div class="mb-4"><h4 class="font-bold border-b pb-1 mb-2 text-brand-600">3. Doctor's Clinical Notes</h4><p id="pd-notes" class="text-sm"></p></div>
         <div class="mb-4"><h4 class="font-bold border-b pb-1 mb-2 text-brand-600">4. Prescribed Medication</h4><ul id="pd-meds" class="text-sm list-disc pl-5"></ul></div>
         <div id="pd-doc-files-container" class="mb-4 hidden"><h4 class="font-bold border-b pb-1 mb-2 text-brand-600">5. Doctor Attachments</h4><div id="pd-doc-files" class="flex gap-2 flex-wrap"></div></div>
         <div class="mb-4"><h4 class="font-bold border-b pb-1 mb-2 text-brand-600">6. Patient Experience Review</h4><p id="pd-review-print" class="text-sm">Not submitted.</p></div>
         <div id="pd-review-section" class="mt-6 p-4 rounded-2xl border border-amber-200 bg-amber-50/70 hidden">
           <div class="flex items-center justify-between gap-3">
             <div><h4 class="font-black text-slate-800">How was your doctor visit?</h4><p class="text-[10px] text-slate-500">Your review becomes part of the doctor's live rating shown before future appointments.</p></div>
             <div id="pd-review-stars" class="flex gap-1" role="radiogroup" aria-label="Doctor rating"></div>
           </div>
           <textarea id="pd-review-feedback" rows="3" maxlength="2000" class="mt-3 w-full p-3 rounded-xl bg-white border border-amber-200 text-sm outline-none focus:ring-2 focus:ring-amber-400" placeholder="Optional feedback about your consultation..."></textarea>
           <div class="flex justify-end mt-3"><button onclick="submitDoctorReview()" class="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white font-black rounded-xl text-xs shadow">Submit Review</button></div>
         </div>
         <div id="pd-review-existing" class="mt-6 p-4 rounded-2xl border border-emerald-200 bg-emerald-50 hidden"></div>
      </div>
    </div>
  </div>

  <div id="doc-history-modal" class="hidden fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
    <div class="bg-white dark:bg-slate-900 rounded-3xl p-6 w-full max-w-2xl shadow-2xl relative max-h-[80vh] flex flex-col">
      <div class="flex justify-between items-center border-b border-slate-200 pb-4 mb-4">
        <h3 class="text-lg font-black flex items-center gap-2"><i data-lucide="calendar"></i> Today's Consultations</h3>
        <button onclick="document.getElementById('doc-history-modal').classList.add('hidden')" class="p-1 text-slate-500"><i data-lucide="x" class="w-5 h-5"></i></button>
      </div>
      <div id="doc-hist-list" class="flex-1 overflow-y-auto space-y-3 pr-2"></div>
    </div>
  </div>

  <script>
    lucide.createIcons();
    let currentUser = null;
    let systemMeta = { doctors: [], hospitals: [], medicines: [] };
    let currentAuthTarget = 'patient';
    let activeDoctorCase = null;
    let isFollowUpMode = false;
    let selectedMedicineCodes = [];
    let lastSearchResults = [];
    let selectedPatientFiles = [];
    let selectedDoctorFiles = [];
    let selectedReviewRating = 0;
    let activePatientCase = null;

    // ==========================================
    // INITIALIZATION & SESSION
    // ==========================================
    async function init() {
      try {
        await fetchSysMeta();
      } catch(err) {
        console.error("System metadata error:", err);
        alert("The server is running, but system data could not be loaded. Please refresh the page.");
      }
      checkSession();
      initModernUI();
    }

    async function fetchSysMeta() {
      const res = await fetch('/api/sys/meta');
      if(!res.ok) throw new Error(`Metadata request failed: ${res.status}`);

      const data = await res.json();
      systemMeta = {
        doctors: Array.isArray(data.doctors) ? data.doctors : [],
        hospitals: Array.isArray(data.hospitals) ? data.hospitals : [],
        medicines: Array.isArray(data.medicines) ? data.medicines : [],
        ai_live: Boolean(data.ai_live)
      };
      const connectionLabel = document.getElementById('ai-connection-label');
      if(connectionLabel) connectionLabel.innerText = systemMeta.ai_live ? 'Live AI + web grounding connected' : 'Demo mode • set OPENAI_API_KEY for live web grounding';

      const hospSelect = document.getElementById('intake-hosp');
      const staffHospSelect = document.getElementById('astaff-hosp');
      const medSelect = document.getElementById('presc-med-select');

      if(hospSelect) {
        hospSelect.innerHTML = `<option value="">Select Hospital...</option>` +
          systemMeta.hospitals.map(h => `<option value="${h.id}">${h.name} (${h.location})</option>`).join('');
      }
      if(staffHospSelect) {
        staffHospSelect.innerHTML =
          systemMeta.hospitals.map(h => `<option value="${h.id}">${h.name}</option>`).join('');
      }
      if(medSelect) {
        medSelect.innerHTML = `<option value="">-- DB Medicine --</option>` +
          systemMeta.medicines.map(m => `<option value="${m.code}">[${m.code}] ${m.name}</option>`).join('');
      }
    }

    function checkSession() {
      const savedStr = sessionStorage.getItem('curecraft_session');
      if(savedStr) { try { currentUser = JSON.parse(savedStr); applyUserState(); } catch(e) { logout(); } } 
      else navigate('home');
    }

    function saveSession(user) { sessionStorage.setItem('curecraft_session', JSON.stringify(user)); }
    function logout() { currentUser = null; sessionStorage.removeItem('curecraft_session'); document.getElementById('user-badge').classList.add('hidden'); document.getElementById('logout-btn').classList.add('hidden'); navigate('home'); }

    function renderAvatar(elementId, initialId, user) {
        const container = document.getElementById(elementId);
        if(!container || !user) return;
        if(user.profile_pic) {
            container.innerHTML = `<img src="${user.profile_pic}" class="w-full h-full object-cover">`;
        } else {
            container.innerHTML = `<span id="${initialId}">${(user.name || 'U').charAt(0).toUpperCase()}</span>`;
        }
    }

    function applyUserState() {
      document.getElementById('user-badge').classList.remove('hidden'); document.getElementById('user-badge').classList.add('flex'); document.getElementById('logout-btn').classList.remove('hidden');
      document.getElementById('nav-user-name').innerText = currentUser.name; document.getElementById('nav-user-role').innerText = currentUser.role; 
      renderAvatar('nav-initial-container', 'nav-initial', currentUser);
      
      if(currentUser.role === 'patient') {
        // These profile-card elements are optional in the current patient UI.
        // Guard them so successful patient authentication always reaches the dashboard.
        renderAvatar('pat-prof-initial', 'pat-prof-initial-txt', currentUser);
        const patName = document.getElementById('pat-prof-name');
        const patAbha = document.getElementById('pat-prof-abha');
        const patDemo = document.getElementById('pat-prof-demographics');
        const patBlood = document.getElementById('pat-prof-blood');
        if(patName) patName.innerText = currentUser.name || '';
        if(patAbha) patAbha.innerText = currentUser.username || '';
        if(patDemo) patDemo.innerText = `${currentUser.age || ''} / ${currentUser.gender || ''}`;
        if(patBlood) patBlood.innerText = currentUser.blood_group || '';
        loadPatientHistory(); navigate('patient'); 
      }
      if(currentUser.role === 'doctor') {
        document.getElementById('doc-prof-name').innerText = currentUser.name; document.getElementById('doc-prof-dept').innerText = currentUser.dept; document.getElementById('doc-prof-consults').innerText = currentUser.consults;
        const hosp = systemMeta.hospitals.find(h => h.id === currentUser.hospital_id); document.getElementById('doc-prof-hosp').innerText = hosp ? hosp.name : '';
        const myRating = systemMeta.doctors.find(d => d.id === currentUser.username);
        const ratingEl = document.getElementById('doc-prof-rating'); if(ratingEl && myRating) ratingEl.innerHTML = renderStars(myRating.rating_average, myRating.rating_count);
        loadDoctorQueue(); navigate('doctor'); 
      }
      if(currentUser.role === 'admin') navigate('admin');
    }

    // ==========================================
    // UI NAVIGATION
    // ==========================================
    function toggleTheme() { document.documentElement.classList.toggle('dark'); }
    function navigate(viewId) { document.querySelectorAll('.page-view').forEach(v => { v.classList.remove('active'); v.style.display = 'none'; }); const target = document.getElementById('view-' + viewId); if(target) { target.style.display = 'block'; setTimeout(() => target.classList.add('active'), 10); } window.scrollTo(0, 0); }
    function handleLogoClick() { if (currentUser) navigate(currentUser.role); else navigate('home'); }

    // ==========================================
    // AUTHENTICATION
    // ==========================================
    function openLogin(role) {
      document.getElementById('auth-title').innerText = `${role.charAt(0).toUpperCase() + role.slice(1)} Login`;
      document.getElementById('register-link-container').style.display = role === 'patient' ? 'block' : 'none';
      toggleAuthMode(false); currentAuthTarget = role; navigate('auth'); setTimeout(() => document.getElementById('auth-user').focus(), 100);
    }
    function toggleAuthMode(isReg) {
        document.getElementById('login-form').style.display = isReg ? 'none' : 'block';
        document.getElementById('register-form').style.display = isReg ? 'block' : 'none';
        document.getElementById('auth-title').innerText = isReg ? "Patient Registration" : "Secure Login";
    }

    async function executeLogin() {
      const u = document.getElementById('auth-user').value.trim();
      const p = document.getElementById('auth-pass').value;
      if(!u || !p) return alert("Please enter your ID and passkey.");

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username: u, password: p})
        });

        let data = {};
        try { data = await res.json(); } catch (_) {}

        if(!res.ok) {
          return alert(data.message || "Invalid credentials.");
        }

        if(!data.user || data.user.role !== currentAuthTarget) {
          return alert(`Access Denied. This account is not a ${currentAuthTarget} account.`);
        }

        currentUser = data.user;
        saveSession(currentUser);
        applyUserState();
      } catch(err) {
        console.error("Login error:", err);
        alert("Unable to connect to the local server. Make sure the FastAPI server is running on http://127.0.0.1:8000.");
      }
    }

    async function executeRegister() {
        const payload = {
            username: document.getElementById('reg-id').value, password: document.getElementById('reg-pass').value,
            name: document.getElementById('reg-name').value, age: parseInt(document.getElementById('reg-age').value) || 0,
            gender: document.getElementById('reg-gender').value, blood_group: document.getElementById('reg-bg').value
        };
        const res = await fetch('/api/auth/register-patient', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json(); alert(data.message);
        if(res.ok) toggleAuthMode(false);
    }

    // ==========================================
    // PROFILE MANAGEMENT
    // ==========================================
    function openProfileModal() {
      renderAvatar('modal-initial-container', 'modal-initial', currentUser);
      document.getElementById('modal-name').innerText = currentUser.name; document.getElementById('modal-id').innerText = currentUser.username;
      document.getElementById('edit-name').value = currentUser.name;
      document.getElementById('modal-details').innerHTML = Object.entries(currentUser).filter(([k]) => !['name','username','password','profile_pic'].includes(k)).map(([k,v]) => `<div class="flex justify-between border-b border-slate-100 py-1"><span class="capitalize text-slate-500">${k.replace('_',' ')}</span><span class="font-bold">${v}</span></div>`).join('');
      document.getElementById('profile-modal').classList.remove('hidden');
    }
    function closeProfileModal() { document.getElementById('profile-modal').classList.add('hidden'); }
    
    async function updateProfile() {
        const name = document.getElementById('edit-name').value; const pwd = document.getElementById('edit-pass').value;
        const picInput = document.getElementById('edit-profile-pic');
        let picBase64 = currentUser.profile_pic;
        if(picInput.files.length > 0) picBase64 = await encodeFile(picInput.files[0]);
        
        const payload = { user_id: currentUser.username, new_password: pwd || null, profile_pic: picBase64, updates: { name: name } };
        const res = await fetch('/api/user/update', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json(); alert(data.message);
        if(res.ok) { currentUser = data.user; saveSession(currentUser); applyUserState(); closeProfileModal(); }
    }

    // ==========================================
    // AI SIMULATOR
    // ==========================================
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    }

    function renderStars(avg, count = 0) {
      const value = Number(avg || 0);
      return Array.from({length: 5}, (_, i) => `<span class="${i + 1 <= Math.round(value) ? 'text-amber-400' : 'text-slate-300'}">★</span>`).join('') + ` <span class="text-[10px] text-slate-500">${value ? value.toFixed(1) : 'New'}${count ? ` (${count})` : ''}</span>`;
    }

    function aiPriorityClass(priority) {
      if(String(priority).includes('Level 1')) return 'text-rose-400 border-rose-500/40 bg-rose-500/10';
      if(String(priority).includes('Level 2')) return 'text-amber-300 border-amber-500/40 bg-amber-500/10';
      if(String(priority).includes('Level 3')) return 'text-sky-300 border-sky-500/40 bg-sky-500/10';
      return 'text-slate-300 border-slate-600 bg-slate-800';
    }

    function renderAIReport(report, targetId, compact = false) {
      const out = document.getElementById(targetId);
      if(!out) return;
      const flags = Array.isArray(report.red_flags) ? report.red_flags : [];
      const actions = Array.isArray(report.what_to_do_now) ? report.what_to_do_now : [];
      const questions = Array.isArray(report.questions_for_doctor) ? report.questions_for_doctor : [];
      const sources = Array.isArray(report.sources) ? report.sources : [];
      out.innerHTML = `
        <div class="rounded-2xl border p-4 ${aiPriorityClass(report.priority)}">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <span class="font-black">${escapeHtml(report.status)}</span>
            <span class="text-[10px] font-black px-2 py-1 rounded-full border border-current/30">${escapeHtml(report.priority)}</span>
          </div>
          <div class="grid ${compact ? 'grid-cols-1' : 'grid-cols-2'} gap-3 mt-4 text-[11px]">
            <div><span class="opacity-60">Department</span><div class="font-bold text-white mt-0.5">${escapeHtml(report.dept)}</div></div>
            <div><span class="opacity-60">Risk screen</span><div class="font-bold text-white mt-0.5">${escapeHtml(report.risk)}</div></div>
          </div>
          <p class="text-xs leading-5 mt-4 text-slate-200">${escapeHtml(report.clinical_summary)}</p>
        </div>
        ${flags.length ? `<div class="mt-3 p-3 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-200 text-xs"><b>Red flags detected</b><ul class="list-disc pl-5 mt-1">${flags.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul></div>` : ''}
        <div class="mt-3 p-3 rounded-xl border border-slate-700 bg-slate-800/80 text-slate-200 text-xs"><b>What to do now</b><ul class="list-disc pl-5 mt-1">${actions.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul></div>
        <div class="mt-3 p-3 rounded-xl border border-slate-700 bg-slate-800/80 text-slate-300 text-xs"><b>Medication safety</b><p class="mt-1">${escapeHtml(report.medication_guidance)}</p></div>
        ${questions.length ? `<div class="mt-3 p-3 rounded-xl border border-slate-700 bg-slate-800/80 text-slate-300 text-xs"><b>Useful questions for your doctor</b><ul class="list-disc pl-5 mt-1">${questions.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul></div>` : ''}
        <div class="mt-3 flex items-center justify-between gap-2 text-[10px] text-slate-500"><span>${escapeHtml(report.ai_mode)}</span><span>${sources.length ? `${sources.length} web source${sources.length > 1 ? 's' : ''}` : 'No live sources returned'}</span></div>
        ${sources.length ? `<div class="mt-2 flex flex-wrap gap-2">${sources.map(s => `<a class="px-2 py-1 rounded-lg bg-slate-800 text-sky-300 border border-slate-700 hover:border-sky-400 text-[10px]" href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title || 'Source')}</a>`).join('')}</div>` : ''}
      `;
    }

    async function runDemoAi() {
      const text = document.getElementById('demo-input').value.trim();
      if(!text) return toast('Describe the symptom narrative first.', 'warning');
      const out = document.getElementById('demo-output');
      out.innerHTML = `<div class="flex items-center gap-2 text-brand-400"><i data-lucide="loader-circle" class="animate-spin w-4 h-4"></i> Running safety screen + live web-grounded analysis...</div>`;
      lucide.createIcons();
      try {
        const res = await fetch('/api/ai/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({transcript:text})});
        const data = await res.json();
        if(!res.ok) throw new Error(data.message || 'AI request failed');
        renderAIReport(data.report, 'demo-output');
        document.getElementById('ai-connection-label').innerText = data.report.ai_mode;
      } catch(err) {
        console.error(err);
        out.innerHTML = `<div class="p-3 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-300 text-xs">${escapeHtml(err.message || 'AI analysis unavailable.')} </div>`;
      }
    }

    // ==========================================
    // PATIENT LOGIC
    // ==========================================
    function filterDepts() {
      const hid = document.getElementById('intake-hosp').value; const docs = systemMeta.doctors.filter(d => d.hospital_id === hid); const depts = [...new Set(docs.map(d => d.dept))];
      document.getElementById('intake-dept').innerHTML = `<option value="">Select Dept...</option>` + depts.map(d => `<option value="${d}">${d}</option>`).join(''); document.getElementById('intake-doc').innerHTML = '';
    }
    function filterDoctors() {
      const hid = document.getElementById('intake-hosp').value;
      const dept = document.getElementById('intake-dept').value;
      const docs = systemMeta.doctors.filter(d => d.hospital_id === hid && d.dept === dept);
      document.getElementById('intake-doc').innerHTML = `<option value="">Select Doctor...</option>` + docs.map(d => {
        const stars = Number(d.rating_count) ? ` ★ ${Number(d.rating_average).toFixed(1)}` : ' ★ New';
        return `<option value="${escapeHtml(d.id)}">${escapeHtml(d.name)}${stars}</option>`;
      }).join('');
      document.getElementById('doctor-rating-preview').classList.add('hidden');
    }

    function showSelectedDoctorRating() {
      const id = document.getElementById('intake-doc').value;
      const box = document.getElementById('doctor-rating-preview');
      const doc = systemMeta.doctors.find(d => d.id === id);
      if(!doc || !id) { box.classList.add('hidden'); return; }
      box.classList.remove('hidden');
      box.innerHTML = `<div class="font-black text-slate-800">Patient experience</div><div class="mt-0.5 text-base leading-none">${renderStars(doc.rating_average, doc.rating_count)}</div>`;
    }
    
    function toggleIntake(isFollow) {
      isFollowUpMode = isFollow; 
      document.getElementById('patient-intake-form').classList.remove('hidden'); 
      document.getElementById('intake-title').innerHTML = isFollow ? `<i data-lucide="refresh-cw"></i> Follow-Up Care` : `<i data-lucide="activity"></i> New Consultation`; 
      document.getElementById('new-consult-fields').style.display = isFollow ? 'none' : 'grid';
      document.getElementById('followup-fields').style.display = isFollow ? 'block' : 'none';
      lucide.createIcons();
    }
    
    function startVoiceRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition; if (!SpeechRecognition) return alert("Voice recognition not supported.");
      const recognition = new SpeechRecognition(); recognition.lang = document.getElementById('voice-lang').value;
      const btn = document.getElementById('mic-btn'); btn.classList.add('animate-pulse', 'bg-rose-500', 'text-white');
      recognition.onresult = (event) => { document.getElementById('intake-transcript').value += (document.getElementById('intake-transcript').value ? ' ' : '') + event.results[0][0].transcript; };
      recognition.onend = () => btn.classList.remove('animate-pulse', 'bg-rose-500', 'text-white');
      recognition.start();
    }

    async function encodeFile(file) {
      return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = e => resolve({
          name: file.name,
          type: file.type || 'application/octet-stream',
          size: file.size,
          data: e.target.result
        });
        r.onerror = () => reject(r.error || new Error("Could not read file"));
        r.readAsDataURL(file);
      });
    }

    function formatFileSize(bytes) {
      if(bytes < 1024) return `${bytes} B`;
      if(bytes < 1024 * 1024) return `${(bytes/1024).toFixed(1)} KB`;
      return `${(bytes/(1024*1024)).toFixed(1)} MB`;
    }

    function renderFileChips(listId, files, removeFn) {
      const el = document.getElementById(listId);
      if(!el) return;
      el.innerHTML = files.length ? files.map((f, i) => `
        <div class="file-chip inline-flex items-center gap-2 px-2.5 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-[10px] shadow-sm">
          <i data-lucide="paperclip" class="w-3 h-3 text-brand-500"></i><span class="max-w-[180px] truncate">${escapeHtml(f.name)}</span><span class="text-slate-400">${formatFileSize(f.size)}</span>
          <button type="button" onclick="${removeFn}(${i})" class="text-rose-500 hover:scale-110 transition" title="Remove this file"><i data-lucide="x" class="w-3 h-3"></i></button>
        </div>`).join('') : '<span class="text-[10px] text-slate-400 italic">No files selected.</span>';
      lucide.createIcons();
    }

    function syncPatientFiles(input) {
      selectedPatientFiles = Array.from(input.files || []);
      renderFileChips('intake-file-list', selectedPatientFiles, 'removePatientFile');
    }
    function removePatientFile(index) {
      selectedPatientFiles.splice(index, 1);
      renderFileChips('intake-file-list', selectedPatientFiles, 'removePatientFile');
    }
    function clearPatientFiles() {
      selectedPatientFiles = [];
      const input = document.getElementById('intake-photos');
      if(input) input.value = '';
      renderFileChips('intake-file-list', selectedPatientFiles, 'removePatientFile');
    }

    function syncDoctorFiles(input) {
      selectedDoctorFiles = Array.from(input.files || []);
      renderFileChips('doctor-file-list', selectedDoctorFiles, 'removeDoctorFile');
    }
    function removeDoctorFile(index) {
      selectedDoctorFiles.splice(index, 1);
      renderFileChips('doctor-file-list', selectedDoctorFiles, 'removeDoctorFile');
    }
    function clearDoctorFiles() {
      selectedDoctorFiles = [];
      const input = document.getElementById('presc-attach');
      if(input) input.value = '';
      renderFileChips('doctor-file-list', selectedDoctorFiles, 'removeDoctorFile');
    }

    async function submitIntake() {
      const txt = document.getElementById('intake-transcript').value.trim();
      if(!txt) return alert("Please provide symptoms.");

      try {
        const atts = selectedPatientFiles.length
          ? await Promise.all(selectedPatientFiles.map(encodeFile))
          : [];

        let payload = {
          abha_id: currentUser.username,
          is_followup: isFollowUpMode,
          transcript: txt,
          photos: atts
        };

        if(isFollowUpMode) {
          const ref = document.getElementById('intake-ref-case').value.trim();
          if(!ref) return alert("Please enter valid Central Case ID.");
          payload.ref_case_id = ref;
        } else {
          payload.hospital_id = document.getElementById('intake-hosp').value;
          payload.dept = document.getElementById('intake-dept').value;
          payload.doc_id = document.getElementById('intake-doc').value;

          if(!payload.hospital_id || !payload.dept || !payload.doc_id) {
            return alert("Fill all dropdowns.");
          }
        }

        const res = await fetch('/api/patient/intake', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });

        const data = await res.json();

        if(res.ok) {
          alert(`Intake submitted successfully.\nCase ID: ${data.case_id}`);
          document.getElementById('patient-intake-form').classList.add('hidden');
          document.getElementById('intake-transcript').value = '';
          clearPatientFiles();
          document.getElementById('intake-ref-case').value = '';
          loadPatientHistory();
        } else {
          alert(data.message || "Unable to submit patient intake.");
        }
      } catch(err) {
        console.error("Intake submission error:", err);
        alert("Unable to connect to the local server.");
      }
    }

    async function loadPatientHistory() {
      const list = document.getElementById('patient-history-list');
      try {
        const res = await fetch(`/api/cases/history/${encodeURIComponent(currentUser.username)}?role=patient`);
        const data = await res.json();

        if(!res.ok) throw new Error(data.message || "History request failed.");

        const records = Array.isArray(data.records) ? data.records : [];
        if(records.length === 0) {
          list.innerHTML = `<p class="text-xs text-slate-500 italic text-center py-4">No records found.</p>`;
          return;
        }

        list.innerHTML = records.map(r => `
        <div onclick='openPatientCaseDetail(${JSON.stringify(r).replace(/'/g, "&#39;")})' class="cursor-pointer hover:shadow-md transition border border-slate-200 dark:border-slate-700 rounded-2xl p-4 bg-white dark:bg-slate-900 shadow-sm relative overflow-hidden">
          <div class="absolute left-0 top-0 bottom-0 w-1 ${r.emergency ? 'bg-rose-500' : 'bg-brand-500'}"></div>
          <div class="flex justify-between">
            <div><span class="font-black text-slate-800 dark:text-white">${r.doc_name}</span> <span class="text-[10px] text-slate-500 ml-2">${r.hospital_name}</span></div>
            <span class="text-[10px] text-slate-500 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded">${r.date_str}</span>
          </div>
          <p class="text-[10px] font-mono text-brand-600 mt-1">${r.case_id}</p>
          <p class="text-xs text-slate-500 mt-1">Status: <b class="${r.status==='completed'?'text-green-500':'text-yellow-500'}">${r.status.toUpperCase()}</b></p>
        </div>
      `).join('');
      } catch(err) {
        console.error("Patient history error:", err);
        list.innerHTML = `<p class="text-xs text-rose-500 text-center py-4">Unable to load records. Please refresh.</p>`;
      }
    }

    function openPatientCaseDetail(c) {
        document.getElementById('pd-case-id').innerText = c.case_id;
        document.getElementById('pd-doc').innerText = c.doc_name;
        document.getElementById('pd-hosp').innerText = c.hospital_name;
        document.getElementById('pd-pat').innerText = c.patient_name;
        document.getElementById('pd-date').innerText = c.date_str;
        document.getElementById('pd-trans').innerText = c.transcript;
        document.getElementById('pd-review-print').innerText = c.review ? `${'★'.repeat(Number(c.review.rating))}${'☆'.repeat(5-Number(c.review.rating))} — ${c.review.feedback || 'No written feedback.'}` : 'Not submitted.';
        const summary = c.ai_report?.clinical_summary || 'No AI summary stored.';
        const flags = Array.isArray(c.ai_report?.red_flags) ? c.ai_report.red_flags : [];
        const sources = Array.isArray(c.ai_report?.sources) ? c.ai_report.sources : [];
        document.getElementById('pd-ai').innerHTML = `
          <div class="grid grid-cols-2 gap-2 mb-3">
            <div><span class="text-slate-500">Priority</span><div class="font-black">${escapeHtml(c.ai_report?.priority || '—')}</div></div>
            <div><span class="text-slate-500">Department</span><div class="font-black">${escapeHtml(c.ai_report?.dept || '—')}</div></div>
          </div>
          <p class="text-sm leading-6">${escapeHtml(summary)}</p>
          ${flags.length ? `<div class="mt-3 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs"><b>Red flags</b><ul class="list-disc pl-5 mt-1">${flags.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul></div>` : ''}
          <div class="mt-3 text-xs"><b>Medication safety:</b> ${escapeHtml(c.ai_report?.medication_guidance || 'Clinician review required.')}</div>
          ${sources.length ? `<div class="mt-3 flex flex-wrap gap-2">${sources.map(s => `<a class="px-2 py-1 rounded-lg bg-sky-50 text-sky-700 text-[10px]" href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title || 'Source')}</a>`).join('')}</div>` : ''}
          ${c.ai_report?.fhir_resource ? `<div class="mt-3 border-t pt-3"><button onclick="this.nextElementSibling.classList.toggle('hidden')" class="text-[10px] bg-slate-800 text-white px-2 py-1 rounded font-mono">View Raw FHIR JSON</button><pre class="hidden bg-slate-900 text-green-400 p-2 text-[10px] rounded mt-1 overflow-x-auto">${escapeHtml(JSON.stringify(c.ai_report.fhir_resource, null, 2))}</pre></div>` : ''}
        `;
        if(c.status === 'completed') {
            document.getElementById('pd-notes').innerText = c.prescription.notes;
            document.getElementById('pd-meds').innerHTML = c.prescription.medicines.map(m => `<li>${m}</li>`).join('');
            const docFiles = document.getElementById('pd-doc-files');
            if(c.prescription.attachments && c.prescription.attachments.length > 0) {
                document.getElementById('pd-doc-files-container').classList.remove('hidden');
                docFiles.innerHTML = c.prescription.attachments.map((a, i) => {
                    const url = typeof a === 'string' ? a : a.url;
                    const name = typeof a === 'string' ? `Doctor_File_${i+1}` : (a.name || `Doctor_File_${i+1}`);
                    return `<a href="${url}" download="${name.replace(/"/g, '')}" target="_blank" class="px-3 py-1.5 bg-indigo-100 text-indigo-700 rounded-lg text-xs font-bold flex items-center gap-1 hover:bg-indigo-200"><i data-lucide="download" class="w-3 h-3"></i> ${name}</a>`;
                }).join('');
            } else document.getElementById('pd-doc-files-container').classList.add('hidden');
        } else {
            document.getElementById('pd-notes').innerText = "Pending doctor review...";
            document.getElementById('pd-meds').innerHTML = "";
            document.getElementById('pd-doc-files-container').classList.add('hidden');
        }
        activePatientCase = c;
        renderPatientReviewUI(c);
        document.getElementById('case-detail-modal').classList.remove('hidden');
        lucide.createIcons();
    }

    function setReviewRating(value) {
      selectedReviewRating = value;
      const wrap = document.getElementById('pd-review-stars');
      wrap.innerHTML = Array.from({length: 5}, (_, i) => `<button type="button" class="star-btn ${i < value ? 'active' : ''}" onclick="setReviewRating(${i+1})" aria-label="${i+1} star${i ? 's' : ''}">★</button>`).join('');
    }

    function renderPatientReviewUI(c) {
      selectedReviewRating = c.review?.rating || 0;
      const section = document.getElementById('pd-review-section');
      const existing = document.getElementById('pd-review-existing');
      section.classList.add('hidden');
      existing.classList.add('hidden');
      document.getElementById('pd-review-feedback').value = '';
      document.getElementById('pd-review-stars').innerHTML = '';

      if(c.status !== 'completed') return;
      if(c.review) {
        existing.classList.remove('hidden');
        existing.innerHTML = `<div class="flex items-center justify-between"><b class="text-emerald-800">Your review</b><span class="text-amber-500 font-black">${'★'.repeat(Number(c.review.rating))}${'☆'.repeat(5-Number(c.review.rating))}</span></div><p class="mt-1 text-xs text-emerald-900">${escapeHtml(c.review.feedback || 'No written feedback.')}</p>`;
        return;
      }
      section.classList.remove('hidden');
      setReviewRating(0);
    }

    async function submitDoctorReview() {
      if(!activePatientCase || activePatientCase.status !== 'completed') return;
      if(!selectedReviewRating) return toast('Please select a star rating first.', 'warning');
      const feedback = document.getElementById('pd-review-feedback').value.trim();
      try {
        const res = await fetch('/api/patient/review', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({patient_id:currentUser.username, case_id:activePatientCase.case_id, doctor_id:activePatientCase.doc_id, rating:selectedReviewRating, feedback})});
        const data = await res.json();
        if(!res.ok) throw new Error(data.message || 'Unable to save review');
        activePatientCase.review = data.review;
        renderPatientReviewUI(activePatientCase);
        toast(`Review saved. ${data.doctor_rating.average ? data.doctor_rating.average.toFixed(1) : 'New'}★ doctor rating`, 'success');
        await fetchSysMeta();
      } catch(err) {
        toast(err.message || 'Review submission failed.', 'error');
      }
    }
    
    function downloadPDF() {
        const element = document.getElementById('printable-report');
        html2pdf().from(element).set({margin: 10, filename: 'CureCraft_Report.pdf', html2canvas: { scale: 2 }, jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }}).save();
    }

    // ==========================================
    // DOCTOR LOGIC
    // ==========================================
    async function loadDoctorQueue() {
      const res = await fetch(`/api/cases/queue/${currentUser.username}`);
      const data = await res.json();
      const qList = document.getElementById('doc-queue-list');
      if(data.queue.length === 0) { qList.innerHTML = `<div class="h-full flex items-center justify-center text-slate-400 text-xs">Queue clear</div>`; return; }
      
      qList.innerHTML = data.queue.map(c => `
        <div onclick='openCase(${JSON.stringify(c).replace(/'/g, "&#39;")})' class="p-3 rounded-xl bg-white dark:bg-slate-900 border ${c.emergency ? 'border-rose-400' : 'border-slate-200'} cursor-pointer hover:shadow-md transition">
          <div class="flex justify-between items-center mb-1"><span class="font-bold text-sm ${c.emergency ? 'text-rose-600' : ''}">${c.patient_name}</span></div>
          <p class="text-[11px] text-slate-500 truncate">${c.transcript}</p>
        </div>
      `).join('');
    }

    function openCase(c) {
      activeDoctorCase = c; selectedMedicineCodes = []; renderMedicines();
      document.getElementById('presc-notes').value = '';
      document.getElementById('doc-idle').classList.add('hidden'); document.getElementById('doc-active-case').classList.remove('hidden');
      document.getElementById('case-pat-name').innerText = c.patient_name; document.getElementById('case-pat-id').innerText = c.abha_id; document.getElementById('case-id').innerText = c.case_id; 
      document.getElementById('case-transcript').innerText = c.transcript;
      
      document.getElementById('case-hpi').innerHTML = `
        <div class="flex justify-between border-b pb-1"><span class="text-slate-500">Priority</span><span class="font-black ${c.emergency ? 'text-rose-500' : 'text-indigo-600'}">${escapeHtml(c.ai_report?.priority || '—')}</span></div>
        <div class="flex justify-between border-b pb-1"><span class="text-slate-500">Department</span><span class="font-bold">${escapeHtml(c.ai_report?.dept || '—')}</span></div>
        <div class="flex justify-between border-b pb-1"><span class="text-slate-500">Risk</span><span class="font-bold">${escapeHtml(c.ai_report?.risk || '—')}</span></div>
        <div class="pt-2"><span class="text-slate-500 block mb-1">Clinical summary</span><span class="text-xs leading-5">${escapeHtml(c.ai_report?.clinical_summary || '—')}</span></div>
      `;
      
      let badges = '';
      if(c.emergency) badges += `<span class="px-2 py-0.5 bg-rose-100 text-rose-700 text-[10px] font-black rounded shadow-sm border border-rose-200">Emergency</span> `;
      if(c.is_followup) badges += `<span class="px-2 py-0.5 bg-indigo-100 text-indigo-700 text-[10px] font-black rounded shadow-sm border border-indigo-200">Follow-Up</span>`;
      document.getElementById('case-badges').innerHTML = badges;

      const pCont = document.getElementById('case-photos-container'); const pDiv = document.getElementById('case-photos');
      if(c.patient_photos && c.patient_photos.length > 0) {
        pCont.classList.remove('hidden');
        pDiv.innerHTML = c.patient_photos.map((p,i) => {
          const url = typeof p === 'string' ? p : p.url;
          const name = typeof p === 'string' ? `Patient_File_${i+1}` : (p.name || `Patient_File_${i+1}`);
          return `<a href="${url}" download="${name.replace(/"/g, '')}" target="_blank" class="px-3 py-1 bg-brand-100 text-brand-700 rounded text-[10px] font-bold flex items-center gap-1 hover:bg-brand-200"><i data-lucide="download" class="w-3 h-3"></i> ${name}</a>`;
        }).join('');
      } else pCont.classList.add('hidden');
      lucide.createIcons();
    }

    function addMedicine() { const val = document.getElementById('presc-med-select').value; if(val && !selectedMedicineCodes.includes(val)) { selectedMedicineCodes.push(val); renderMedicines(); } }
    function removeMedicine(idx) { selectedMedicineCodes.splice(idx, 1); renderMedicines(); }
    function renderMedicines() {
      document.getElementById('presc-med-list').innerHTML = selectedMedicineCodes.map((code, i) => {
        const name = systemMeta.medicines.find(m => m.code === code)?.name || code;
        return `<li class="flex justify-between p-2 rounded bg-teal-50 dark:bg-teal-900/20 text-teal-800 dark:text-teal-400 text-xs font-bold"><span>[${code}] ${name}</span><button onclick="removeMedicine(${i})" class="text-rose-500"><i data-lucide="x" class="w-3 h-3"></i></button></li>`
      }).join(''); lucide.createIcons();
    }

    async function submitPrescription() {
      if(!activeDoctorCase) return; const notes = document.getElementById('presc-notes').value; if(!notes) return alert("Enter notes.");
      const atts = selectedDoctorFiles.length ? await Promise.all(selectedDoctorFiles.map(encodeFile)) : [];
      const payload = { case_id: activeDoctorCase.case_id, abha_id: activeDoctorCase.abha_id, doc_id: currentUser.username, clinical_notes: notes, medicine_codes: selectedMedicineCodes, attachments: atts };
      const res = await fetch('/api/doctor/prescribe', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json();
      if(res.ok) { toast("E-Prescription secured and sent to the patient.", "success"); clearDoctorFiles(); document.getElementById('doc-active-case').classList.add('hidden'); document.getElementById('doc-idle').classList.remove('hidden'); currentUser.consults++; saveSession(currentUser); document.getElementById('doc-prof-consults').innerText = currentUser.consults; loadDoctorQueue(); }
      else toast(data.message || "Unable to save prescription.", "error");
    }
    
    async function showDocHistory() {
      const res = await fetch(`/api/cases/history/${currentUser.username}?role=doctor`);
      const data = await res.json();
      const list = document.getElementById('doc-hist-list');
      if(data.records.length === 0) { list.innerHTML = `<p class="text-center text-slate-500 text-sm">No consults today.</p>`; }
      else {
        list.innerHTML = data.records.map(r => `
          <div class="border border-slate-200 dark:border-slate-800 p-4 rounded-xl">
             <div class="flex justify-between font-bold text-sm mb-2 text-indigo-600 dark:text-indigo-400"><span>${r.patient_name} (${r.case_id})</span><span>${r.prescription.date.split(' ')[1]}</span></div>
             <p class="text-xs text-slate-600 dark:text-slate-400 italic">${r.prescription.notes}</p>
          </div>
        `).join('');
      }
      document.getElementById('doc-history-modal').classList.remove('hidden');
    }

    async function loadAdminRatings() {
      const wrap = document.getElementById('admin-rating-aggregates');
      const list = document.getElementById('admin-review-list');
      if(!wrap || !list) return;
      wrap.innerHTML = '<div class="text-xs text-slate-500">Loading ratings…</div>';
      try {
        const res = await fetch(`/api/admin/ratings?admin_id=${encodeURIComponent(currentUser.username)}`);
        const data = await res.json();
        if(!res.ok) throw new Error(data.message || 'Unable to load ratings');
        wrap.innerHTML = data.aggregates.length ? data.aggregates.map(d => `
          <div class="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 shadow-sm">
            <div class="flex justify-between gap-2"><div><div class="font-black">${escapeHtml(d.doctor_name)}</div><div class="text-[10px] text-slate-500">${escapeHtml(d.dept || '—')} • ${escapeHtml(d.hospital_id || '—')}</div></div><div class="text-right"><div class="text-lg leading-none">${renderStars(d.average, d.count)}</div><div class="text-[10px] text-slate-400 mt-1">${d.count} review${d.count === 1 ? '' : 's'}</div></div></div>
          </div>`).join('') : '<div class="text-xs text-slate-500">No doctors found.</div>';
        list.innerHTML = data.reviews.length ? data.reviews.map(r => `
          <div class="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900"><div class="flex justify-between"><div class="font-bold text-xs">${escapeHtml(r.doctor_id)} • ${escapeHtml(r.case_id)}</div><div class="text-amber-500 text-sm">${'★'.repeat(Number(r.rating))}${'☆'.repeat(5-Number(r.rating))}</div></div><p class="text-xs text-slate-600 dark:text-slate-300 mt-1">${escapeHtml(r.feedback || 'No written feedback.')}</p></div>`).join('') : '<p class="text-xs text-slate-500">No reviews yet.</p>';
      } catch(err) {
        wrap.innerHTML = `<div class="text-xs text-rose-500">${escapeHtml(err.message || 'Unable to load ratings')}</div>`;
        list.innerHTML = '';
      }
    }

    // ==========================================
    // ADMIN LOGIC
    // ==========================================
    function switchAdminTab(tab) {
      ['hospital', 'staff', 'medicine', 'analytics', 'ratings'].forEach(t => {
        document.getElementById('atab-'+t).className = "admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500";
        document.getElementById('apanel-'+t).classList.add('hidden');
      });
      document.getElementById('atab-'+tab).className = "admin-tab px-4 py-2 font-bold text-sm border-b-2 border-brand-500 text-brand-600";
      document.getElementById('apanel-'+tab).classList.remove('hidden');
      if(tab === 'ratings') loadAdminRatings();
    }
    
    function toggleAdminDept() {
      const isDoc = document.getElementById('astaff-role').value === 'doctor';
      document.getElementById('astaff-doc-fields').style.display = isDoc ? 'flex' : 'none';
    }

    async function manageHospital(action) {
      const payload = { admin_id: currentUser.username, action: action, id: document.getElementById('ahosp-id').value, name: document.getElementById('ahosp-name').value, location: document.getElementById('ahosp-loc').value };
      const res = await fetch('/api/admin/manage-hospital', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json(); alert(data.message); if(res.ok) await fetchSysMeta();
    }
    async function manageMedicine(action) {
      const payload = { admin_id: currentUser.username, action: action, code: document.getElementById('amed-code').value, name: document.getElementById('amed-name').value };
      const res = await fetch('/api/admin/manage-medicine', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json(); alert(data.message); if(res.ok) await fetchSysMeta();
    }
    async function manageStaff(action) {
      const isDoc = document.getElementById('astaff-role').value === 'doctor';
      const payload = { admin_id: currentUser.username, action: action, user_id: document.getElementById('astaff-id').value, role: document.getElementById('astaff-role').value, name: document.getElementById('astaff-name').value, new_password: document.getElementById('astaff-pass').value, dept: isDoc ? document.getElementById('astaff-dept').value : null, hospital_id: isDoc ? document.getElementById('astaff-hosp').value : null };
      const res = await fetch('/api/admin/manage-staff', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await res.json(); alert(data.message); if(res.ok) await fetchSysMeta();
    }

    async function searchDB() {
      const filters = {
          admin_id: currentUser.username,
          date: document.getElementById('search-date').value, hospital_id: document.getElementById('search-hosp').value,
          doc_id: document.getElementById('search-doc').value, abha_id: document.getElementById('search-pat').value, dept: document.getElementById('search-dept').value
      };
      const res = await fetch('/api/admin/search', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(filters) });
      const data = await res.json();
      const div = document.getElementById('admin-search-results');
      if(data.data.length === 0) { div.innerHTML = "<p class='text-sm text-slate-500 mt-4 text-center'>No records found.</p>"; return; }
      
      lastSearchResults = data.data;
      div.innerHTML = `<table class="w-full text-left text-xs border-collapse"><thead><tr class="border-b"><th class="py-2">Date</th><th>Case ID</th><th>Patient</th><th>Dept</th><th>Doctor</th><th>Hospital</th><th>Status</th></tr></thead><tbody>` + 
          data.data.map(r => `<tr class="border-b hover:bg-slate-100 dark:hover:bg-slate-800"><td class="py-2">${r.date_str}</td><td class="font-mono text-brand-600">${r.case_id}</td><td>${r.patient_name} (${r.abha_id})</td><td>${r.dept}</td><td onclick="showAdminStats('doc','${r.doc_id}')" class="cursor-pointer text-indigo-600 hover:underline font-bold">${r.doc_name}</td><td onclick="showAdminStats('hosp','${r.hospital_id}')" class="cursor-pointer text-indigo-600 hover:underline font-bold">${r.hospital_name}</td><td><span class="${r.status==='completed'?'text-green-600':'text-yellow-600'} font-bold">${r.status.toUpperCase()}</span></td></tr>`).join('') + `</tbody></table>`;
    }
    
    async function showAdminStats(type, id) {
        const res = await fetch(`/api/admin/stats/${type}/${id}?admin_id=${encodeURIComponent(currentUser.username)}`);
        const data = await res.json();
        alert(`Total Checkups completed by this ${type === 'doc' ? 'Doctor' : 'Hospital'}: ${data.count}`);
    }

    function exportToCSV() {
        if(!lastSearchResults || lastSearchResults.length === 0) return alert("Search and load data first.");
        let csv = "Date,Case ID,Patient ID,Patient Name,Hospital,Department,Doctor,Status\\n";
        lastSearchResults.forEach(r => { csv += `${r.date_str},${r.case_id},${r.abha_id},${r.patient_name},"${r.hospital_name}",${r.dept},"${r.doc_name}",${r.status}\\n`; });
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a'); a.setAttribute('href', url); a.setAttribute('download', 'CureCraft_Search_Report.csv'); a.click();
    }


    // ==========================================
    // MODERN UI / MOTION LAYER
    // ==========================================
    function toast(message, type = 'info') {
      let tray = document.getElementById('toast-tray');
      if(!tray) {
        tray = document.createElement('div');
        tray.id = 'toast-tray';
        tray.className = 'fixed bottom-5 right-5 z-[300] space-y-2 w-[min(360px,calc(100vw-2rem))]';
        document.body.appendChild(tray);
      }
      const item = document.createElement('div');
      const tone = type === 'success' ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : type === 'error' ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-slate-200 bg-white text-slate-800';
      item.className = `p-3 rounded-2xl border shadow-xl text-xs font-semibold ${tone}`;
      item.textContent = message;
      tray.appendChild(item);
      setTimeout(() => item.remove(), 3600);
    }

    function initModernUI() {
      const progress = document.getElementById('scroll-progress');
      const glow = document.getElementById('cursor-glow');
      window.addEventListener('scroll', () => {
        if(progress) {
          const max = document.documentElement.scrollHeight - window.innerHeight;
          progress.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
        }
      }, {passive:true});
      window.addEventListener('mousemove', e => {
        if(glow) { glow.style.left = `${e.clientX}px`; glow.style.top = `${e.clientY}px`; }
      }, {passive:true});

      const observer = new IntersectionObserver(entries => entries.forEach(entry => entry.isIntersecting && entry.target.classList.add('revealed')), {threshold:.10});
      document.querySelectorAll('.glass, .dna-bg').forEach(el => { el.classList.add('reveal'); observer.observe(el); });

      const canvas = document.getElementById('ambient-canvas');
      if(canvas) {
        const ctx = canvas.getContext('2d');
        const dots = Array.from({length: 38}, () => ({x:Math.random(), y:Math.random(), vx:(Math.random()-.5)*.00025, vy:(Math.random()-.5)*.00025, r:Math.random()*1.7+.3}));
        function fit(){canvas.width=window.innerWidth; canvas.height=window.innerHeight;}
        fit(); window.addEventListener('resize', fit, {passive:true});
        function draw(){
          ctx.clearRect(0,0,canvas.width,canvas.height);
          dots.forEach(d => { d.x += d.vx; d.y += d.vy; if(d.x<0||d.x>1)d.vx*=-1; if(d.y<0||d.y>1)d.vy*=-1; ctx.beginPath(); ctx.arc(d.x*canvas.width,d.y*canvas.height,d.r,0,Math.PI*2); ctx.fillStyle = 'rgba(16,185,129,.42)'; ctx.fill(); });
          requestAnimationFrame(draw);
        }
        draw();
      }
      renderFileChips('intake-file-list', selectedPatientFiles, 'removePatientFile');
      renderFileChips('doctor-file-list', selectedDoctorFiles, 'removeDoctorFile');
    }

    window.onload = init;
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
