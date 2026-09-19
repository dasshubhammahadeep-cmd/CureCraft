import os
import time
import uuid
import json
import base64
import mimetypes
import re
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

STATIC_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCTOR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CureCraft AI - Advanced Triage & EHR")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# =====================================================================
# CENTRALIZED DATABASE
# =====================================================================

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

# =====================================================================
# FILE STORAGE HELPERS
# =====================================================================

def _safe_filename(filename: str) -> str:
    filename = os.path.basename(filename or "attachment")
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip() or "attachment"
    return filename[:150]


def save_attachments(items: List[Any], category: str, case_id: str) -> List[Dict[str, Any]]:
    """Save browser-uploaded files locally and return download metadata."""
    saved = []
    if not items:
        return saved

    folder = UPLOADS_DIR / category
    folder.mkdir(parents=True, exist_ok=True)

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
                raw = base64.b64decode(encoded)
            else:
                raw = base64.b64decode(data)

            ext = Path(original_name).suffix
            if not ext:
                ext = mimetypes.guess_extension(mime) or ""

            filename = f"{case_id}_{uuid.uuid4().hex}{ext}"
            path = folder / filename
            path.write_bytes(raw)

            saved.append({
                "name": _safe_filename(original_name),
                "type": mime,
                "size": len(raw),
                "url": f"/uploads/{category}/{filename}"
            })
        except Exception as exc:
            # A bad attachment should not cancel the whole consultation.
            print(f"Attachment save failed: {exc}")

    return saved


# =====================================================================
# AI ANALYSIS ENGINE (Simulated)
# =====================================================================

def analyze_symptoms(text: str):
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["fine", "good", "nothing", "well", "thik"]):
        return {"status": "No Intervention", "priority": "Green - Safe", "dept": "N/A - Rest", "risk": "Low"}
    elif any(kw in text_lower for kw in ["chest", "heart", "breath", "stroke", "severe", "blood", "pain"]):
        return {"status": "Immediate Triage", "priority": "Red - Emergency", "dept": "Cardiology / ER", "risk": "High"}
    elif any(kw in text_lower for kw in ["head", "fever", "cough", "cold", "stomach", "dizzy"]):
        return {"status": "Consultation Required", "priority": "Yellow - Standard", "dept": "General Medicine", "risk": "Moderate"}
    else:
        return {"status": "Pending Doctor Review", "priority": "Yellow - Standard", "dept": "General Medicine", "risk": "Unknown"}

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/api/sys/meta")
async def get_sys_meta():
    doctors = [{"id": k, "name": v["name"], "dept": v.get("dept"), "hospital_id": v.get("hospital_id")} 
               for k, v in CENTRAL_DB["users"].items() if v["role"] == "doctor"]
    hospitals = list(CENTRAL_DB["hospitals"].values())
    meds = list(CENTRAL_DB["medicines"].values())
    return {"doctors": doctors, "hospitals": hospitals, "medicines": meds}

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

@app.post("/api/admin/manage-hospital")
async def manage_hosp(req: HospReq):
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
    if req.action == "create":
        CENTRAL_DB["medicines"][req.code] = {"code": req.code, "name": req.name}
        return {"status": "success", "message": f"Medicine '{req.name}' added."}
    elif req.action == "delete" and req.code in CENTRAL_DB["medicines"]:
        del CENTRAL_DB["medicines"][req.code]
        return {"status": "success", "message": "Medicine deleted."}
    return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid operation."})

@app.post("/api/admin/manage-staff")
async def manage_staff(req: StaffReq):
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
async def get_stats(stype: str, sid: str):
    count = 0
    for c in CENTRAL_DB["cases"].values():
        if stype == "doc" and c["doc_id"] == sid: count += 1
        elif stype == "hosp" and c["hospital_id"] == sid: count += 1
    return {"status": "success", "count": count}

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

    ai_report = analyze_symptoms(req.transcript)

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
        "emergency": ai_report["risk"] == "High",
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
    
    records.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"records": records}

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
  </style>
</head>
<body class="bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 min-h-screen flex flex-col transition-colors duration-300">

  <!-- HEADER NAVIGATION -->
  <header class="glass sticky top-0 z-50 px-6 py-3 flex items-center justify-between shadow-sm">
    <div class="flex items-center space-x-3 cursor-pointer" onclick="handleLogoClick()">
      <img src="/static/logo.png" alt="CureCraft Logo" class="h-10 w-auto object-contain" onerror="this.onerror=null; this.src='https://via.placeholder.com/120x40?text=CureCraft';">
      <div class="border-l-2 border-slate-300 dark:border-slate-700 pl-3">
        <h1 class="text-lg font-black text-brand-dark dark:text-white leading-none">CureCraft</h1>
        <p class="text-[10px] font-bold text-brand-600 uppercase tracking-wider flex items-center gap-1"><i data-lucide="lock" class="w-3 h-3"></i> AES-256 Encrypted</p>
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
          <div><h3 class="text-xl font-bold">Interactive AI Symptom Simulator</h3><p class="text-xs text-slate-500">Test how CureCraft's AI engine analyzes raw symptoms.</p></div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
          <div class="flex flex-col">
            <label class="block text-xs font-bold uppercase text-slate-500 mb-2">Simulate Patient Input</label>
            <textarea id="demo-input" rows="4" class="w-full p-4 rounded-2xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none focus:ring-2 focus:ring-brand-500 transition flex-1" placeholder="Type here (e.g., Severe chest pain vs I am feeling fine)..."></textarea>
            <button onclick="runDemoAi()" class="mt-4 w-full px-5 py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm rounded-xl transition shadow flex justify-center items-center gap-2"><i data-lucide="activity" class="w-4 h-4"></i> Evaluate Symptoms</button>
          </div>
          <div class="bg-slate-900 rounded-2xl border border-slate-700 flex flex-col overflow-hidden shadow-inner h-full min-h-[200px]">
            <div class="bg-slate-800 p-3 border-b border-slate-700 flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-rose-500"></div><div class="w-2 h-2 rounded-full bg-yellow-500"></div><div class="w-2 h-2 rounded-full bg-green-500"></div>
              <span class="text-[10px] font-bold text-slate-400 ml-2 uppercase">AI Diagnostic Engine</span>
            </div>
            <div class="p-5 overflow-y-auto flex-1 flex flex-col gap-3" id="demo-output">
              <div class="text-xs text-slate-500 italic text-center mt-10">Awaiting input...</div>
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
        <p class="text-center text-xs text-slate-500 mb-6 flex justify-center items-center gap-1"><i data-lucide="shield-check" class="w-3 h-3 text-brand-500"></i> AES-256 Encrypted</p>
        
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
                <select id="intake-doc" class="w-full p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm outline-none shadow-inner"></select>
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
              <input type="file" id="intake-photos" accept=".pdf,.png,.jpg,.jpeg,.webp,.gif,.doc,.docx,.xls,.xlsx,.txt,.csv,.zip,application/pdf,image/*,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" multiple class="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:font-bold file:bg-slate-200 file:text-slate-700">
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
             <h4 class="font-black text-brand-600">E2E Secure</h4>
             <p class="text-xs text-slate-500 mt-1">Data encrypted in central hospital DB.</p>
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
            <p class="text-xs text-indigo-600 dark:text-indigo-400 font-bold mb-3 relative z-10"><span id="doc-prof-dept"></span> &bull; <span id="doc-prof-hosp"></span></p>
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
                    <input type="file" id="presc-attach" accept=".pdf,.png,.jpg,.jpeg,.webp,.gif,.doc,.docx,.xls,.xlsx,.txt,.csv,.zip,application/pdf,image/*,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" multiple class="block w-full text-[10px] text-slate-500 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:font-bold file:bg-slate-200 file:text-slate-700">
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
    }

    async function fetchSysMeta() {
      const res = await fetch('/api/sys/meta');
      if(!res.ok) throw new Error(`Metadata request failed: ${res.status}`);

      const data = await res.json();
      systemMeta = {
        doctors: Array.isArray(data.doctors) ? data.doctors : [],
        hospitals: Array.isArray(data.hospitals) ? data.hospitals : [],
        medicines: Array.isArray(data.medicines) ? data.medicines : []
      };

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
    function runDemoAi() {
      const text = document.getElementById('demo-input').value.toLowerCase().trim(); if(!text) return;
      const out = document.getElementById('demo-output'); out.innerHTML = `<div class="flex items-center gap-2 text-brand-500"><i data-lucide="loader-2" class="animate-spin w-4 h-4"></i> Analyzing semantics...</div>`; lucide.createIcons();
      setTimeout(() => {
        if(text.includes('fine') || text === "i am fine" || text.includes('good')) { out.innerHTML = `<div class="bg-slate-800 p-3 rounded-xl text-green-400 border border-green-500/30"><b>Status:</b> No medical intervention required.<br><b>Priority:</b> Green - Safe<br><b>Recommendation:</b> Maintain health routine.</div>`; } 
        else { const isEmerg = text.includes('chest') || text.includes('pain') || text.includes('breath'); out.innerHTML = `<div class="bg-slate-800 p-3 rounded-xl ${isEmerg?'text-rose-400 border-rose-500/30':'text-yellow-400 border-yellow-500/30'} border"><b>Status:</b> Consultation Recommended<br><b>Priority:</b> ${isEmerg?'Level 1 - Emergency':'Level 3 - Standard'}<br><b>Dept:</b> ${isEmerg?'Cardiology':'General Medicine'}</div>`; }
      }, 600);
    }

    // ==========================================
    // PATIENT LOGIC
    // ==========================================
    function filterDepts() {
      const hid = document.getElementById('intake-hosp').value; const docs = systemMeta.doctors.filter(d => d.hospital_id === hid); const depts = [...new Set(docs.map(d => d.dept))];
      document.getElementById('intake-dept').innerHTML = `<option value="">Select Dept...</option>` + depts.map(d => `<option value="${d}">${d}</option>`).join(''); document.getElementById('intake-doc').innerHTML = '';
    }
    function filterDoctors() {
      const hid = document.getElementById('intake-hosp').value; const dept = document.getElementById('intake-dept').value; const docs = systemMeta.doctors.filter(d => d.hospital_id === hid && d.dept === dept);
      document.getElementById('intake-doc').innerHTML = `<option value="">Select Doctor...</option>` + docs.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
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

    async function submitIntake() {
      const txt = document.getElementById('intake-transcript').value.trim();
      if(!txt) return alert("Please provide symptoms.");

      try {
        const fileInp = document.getElementById('intake-photos');
        const atts = fileInp.files.length
          ? await Promise.all(Array.from(fileInp.files).map(encodeFile))
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
          document.getElementById('intake-photos').value = '';
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
        document.getElementById('pd-ai').innerHTML = Object.entries(c.ai_report).map(([k,v]) => `<div><b class="capitalize">${k}:</b> ${v}</div>`).join('');
        
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
        document.getElementById('case-detail-modal').classList.remove('hidden');
        lucide.createIcons();
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
      
      document.getElementById('case-hpi').innerHTML = Object.entries(c.ai_report).map(([k,v]) => `<div class="flex justify-between border-b border-slate-200 dark:border-slate-800 pb-1"><span class="text-slate-500 capitalize">${k}:</span><span class="font-bold ${k==='priority'&&c.emergency?'text-rose-500':''}">${v}</span></div>`).join('');
      
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
      const fileInp = document.getElementById('presc-attach'); const atts = fileInp.files.length ? await Promise.all(Array.from(fileInp.files).map(encodeFile)) : [];
      const payload = { case_id: activeDoctorCase.case_id, abha_id: activeDoctorCase.abha_id, doc_id: currentUser.username, clinical_notes: notes, medicine_codes: selectedMedicineCodes, attachments: atts };
      const res = await fetch('/api/doctor/prescribe', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      if(res.ok) { alert("E-Prescription sent to DB."); document.getElementById('doc-active-case').classList.add('hidden'); document.getElementById('doc-idle').classList.remove('hidden'); currentUser.consults++; saveSession(currentUser); document.getElementById('doc-prof-consults').innerText = currentUser.consults; loadDoctorQueue(); }
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

    // ==========================================
    // ADMIN LOGIC
    // ==========================================
    function switchAdminTab(tab) {
      ['hospital', 'staff', 'medicine', 'analytics'].forEach(t => {
        document.getElementById('atab-'+t).className = "admin-tab px-4 py-2 font-bold text-sm border-b-2 border-transparent text-slate-500";
        document.getElementById('apanel-'+t).classList.add('hidden');
      });
      document.getElementById('atab-'+tab).className = "admin-tab px-4 py-2 font-bold text-sm border-b-2 border-brand-500 text-brand-600";
      document.getElementById('apanel-'+tab).classList.remove('hidden');
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
        const res = await fetch(`/api/admin/stats/${type}/${id}`);
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

    window.onload = init;
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)