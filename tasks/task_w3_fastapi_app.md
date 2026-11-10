# Goldfish Task Spec: W3 — FastAPI Application (`webapp/app.py`)

## 1. Context
This project is the CPE310 Facial Dermatosis Classifier web application (`DermaScan Face AI`).
Tasks W1 (`webapp/service.py`), W2 (`webapp/knowledge_base.json`), and W4 (`webapp/static/*`) are complete and verified.
Task W3 connects the backend service layer, knowledge base, and static frontend into a production-ready FastAPI application at `webapp/app.py` and updates `requirements.txt`.
Models are loaded once at startup via FastAPI's `lifespan` context manager and reused across all requests.

## 2. Goal
Update `requirements.txt` with web/test dependencies and create `webapp/app.py` implementing lifespan model loading, `GET /api/health`, `POST /api/analyze`, friendly bilingual error handling, and static frontend serving.

## 3. Files to Read First
Read these exact files before writing any code:
- `tasks/WEBAPP_MASTER_PLAN.md`
- `webapp/service.py`
- `webapp/static/index.html` (lines 1–40 and 430–442 to see asset paths `style.css`, `logo.svg`, `app.js`)
- `requirements.txt`

### Copied Signatures & Constants from `webapp/service.py` (Verified)
```python
SYSTEM_NAME: str = "DermaScan Face AI"
CALIBRATION_T_RESNET: float = 0.60
CALIBRATION_T_EFFNET: float = 1.20
CALIBRATION_W_RESNET: float = 0.40
CALIBRATION_W_EFFNET: float = 0.60
ABSTENTION_THRESHOLD: float = 0.7852
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
ALLOWED_MIME_TYPES: Set[str] = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

class WebAppInputError(ValueError):
    def __init__(self, detail: str, detail_th: str, error_code: str, status_code: int = 400): ...
    def to_dict(self) -> Dict[str, Any]:
        # Returns {"detail": self.detail, "detail_th": self.detail_th, "error_code": self.error_code}

def get_service(device: Optional[Union[str, torch.device]] = None) -> WebAppDermatosisService: ...
def analyze(image_bytes: bytes, triage_mode: bool = False) -> Dict[str, Any]: ...
```

### Copied Frozen Endpoint Contracts from `tasks/WEBAPP_MASTER_PLAN.md`
1. **`GET /api/health` (`200 OK`):**
```json
{
  "status": "ok",
  "system_name": "DermaScan Face AI",
  "device": "cuda",
  "models_loaded": true,
  "abstention_threshold": 0.7852,
  "calibration": {
    "t_resnet": 0.6,
    "t_effnet": 1.2,
    "w_resnet": 0.4,
    "w_effnet": 0.6
  }
}
```
2. **`POST /api/analyze` (`200 OK`):**
   - Accepts `multipart/form-data` with `file` (`UploadFile`) and optional `triage_mode` (`bool`, default `False`, accepted from either `Form(False)` or query parameter).
   - Returns the exact dictionary produced by `webapp.service.analyze(image_bytes, triage_mode=triage_mode)`.
3. **Error Responses (`400` / `413` / `415` / `422` / `500`):**
```json
{
  "detail": "Clear human-readable message in English",
  "detail_th": "ข้อความแจ้งเตือนภาษาไทย",
  "error_code": "invalid_file_type | file_too_large | empty_file | unreadable_image | no_skin_detected | server_error"
}
```

## 4. Files to Create or Edit
Create or edit ONLY these files (everything else is strictly off limits):
1. `requirements.txt` (edit: append web & test packages)
2. `webapp/app.py` (create)

## 5. Step-by-Step Instructions

### Step 1: Update `requirements.txt` and ensure packages are installed
1. Append a `# Web application & API testing` section to `requirements.txt` containing:
   ```text
   # Web application & API testing
   fastapi>=0.110
   uvicorn>=0.29
   python-multipart>=0.0.9
   httpx>=0.27
   pytest>=8.0
   ```
2. Check if `fastapi`, `uvicorn`, `multipart` (`python-multipart`), `httpx`, and `pytest` are importable in Python; if any are missing, install them via `pip install fastapi uvicorn python-multipart httpx pytest`.

### Step 2: Create `webapp/app.py`
Implement `webapp/app.py` with the following structure:
1. **Lifespan Context Manager:**
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       svc = get_service()
       app.state.service = svc
       yield
   ```
   *(Note: do NOT destroy the singleton `_SERVICE_INSTANCE` on lifespan exit so that multiple `TestClient(app)` blocks in pytest can reuse the already-loaded GPU models without reloading checkpoints repeatedly).*
2. **FastAPI App Instance & Exception Handlers:**
   - Instantiate `app = FastAPI(title=SYSTEM_NAME, version="1.0.0", lifespan=lifespan)`.
   - Register `@app.exception_handler(WebAppInputError)` returning `JSONResponse(status_code=exc.status_code, content=exc.to_dict())`.
   - Register `@app.exception_handler(RequestValidationError)` returning `JSONResponse(status_code=400, content={"detail": "Missing or invalid upload file. Please select a valid image file (JPG, PNG, WEBP).", "detail_th": "ไม่พบไฟล์รูปภาพหรือข้อมูลที่ส่งมาไม่ถูกต้อง กรุณาเลือกไฟล์ภาพ (JPG, PNG, WEBP)", "error_code": "empty_file"})`.
   - Register a fallback `@app.exception_handler(Exception)` returning HTTP `500` with `{"detail": "Internal server error during image analysis.", "detail_th": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์ระหว่างการวิเคราะห์ภาพ", "error_code": "server_error"}`.
3. **API Routes:**
   - **`GET /api/health`:**
     Calls `svc = get_service()`, checks that `svc.ensemble is not None` and `len(svc.get_knowledge_base_entry("acne")) > 0`, and returns:
     ```python
     {
         "status": "ok",
         "system_name": SYSTEM_NAME,
         "device": str(svc.device),
         "models_loaded": True,
         "knowledge_base_loaded": bool(svc.get_knowledge_base_entry("acne")),
         "abstention_threshold": ABSTENTION_THRESHOLD,
         "calibration": {
             "t_resnet": CALIBRATION_T_RESNET,
             "t_effnet": CALIBRATION_T_EFFNET,
             "w_resnet": CALIBRATION_W_RESNET,
             "w_effnet": CALIBRATION_W_EFFNET,
         },
     }
     ```
   - **`POST /api/analyze`:**
     Signature:
     ```python
     @app.post("/api/analyze")
     async def analyze_endpoint(
         file: Optional[UploadFile] = File(default=None),
         triage_mode: bool = Form(default=False),
     ) -> JSONResponse:
     ```
     Validation logic before calling `analyze()`:
     1. If `file is None`: raise `WebAppInputError("No image file provided. Please upload a JPG, PNG, or WEBP photo.", "ไม่ได้เลือกไฟล์รูปภาพ กรุณาอัปโหลดไฟล์ภาพ JPG, PNG หรือ WEBP", "empty_file", 400)`.
     2. Validate filename extension & MIME type:
        - Extract `filename = (file.filename or "").strip()` and `ext = Path(filename).suffix.lower()`.
        - Extract `content_type = (file.content_type or "").lower().strip()`.
        - If `ext` is present and `ext not in ALLOWED_EXTENSIONS`:
          raise `WebAppInputError(f"Unsupported file extension '{ext}'. Allowed formats: JPG, PNG, WEBP.", f"ไม่รองรับนามสกุลไฟล์ '{ext}' กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP", "invalid_file_type", 415)`.
        - If `content_type` is present and `content_type not in ALLOWED_MIME_TYPES` and `content_type != "application/octet-stream"`:
          raise `WebAppInputError(f"Unsupported content type '{content_type}'. Allowed formats: JPG, PNG, WEBP.", f"ไม่รองรับประเภทไฟล์ '{content_type}' กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP", "invalid_file_type", 415)`.
        - If both `ext == ""` and `content_type not in ALLOWED_MIME_TYPES`:
          raise `WebAppInputError("Unsupported file type. Allowed formats: JPG, PNG, WEBP.", "รูปแบบไฟล์ไม่รองรับ กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP", "invalid_file_type", 415)`.
     3. Read `image_bytes = await file.read()`.
     4. Run `result = await run_in_threadpool(analyze, image_bytes, bool(triage_mode))` (using `fastapi.concurrency.run_in_threadpool` so the event loop is not blocked during GPU inference).
     5. Return `JSONResponse(status_code=200, content=result)`.
4. **Static Frontend Serving:**
   - Define `STATIC_DIR = Path(__file__).resolve().parent / "static"`.
   - Define `@app.get("/")` returning `FileResponse(STATIC_DIR / "index.html")`.
   - Mount `/static`: `app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")`.
   - Mount `/` last (`app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="root_static")`) so that relative references in `index.html` (`href="style.css"`, `src="logo.svg"`, `src="app.js"`, `mock_result.json`) resolve with `200 OK` both at `/style.css` and `/static/style.css`.

## 6. Hard Rules
- **DO NOT** modify `webapp/service.py`, `webapp/knowledge_base.json`, `webapp/static/*`, `src/*`, `data/raw/`, `checkpoints/`, or `.env`.
- **DO NOT** save uploaded files to disk.
- Run all commands from `e:\CPE310FacialDermatorsisProject` (no `cd`). Use UTF-8 encoding.

## 7. Acceptance Criteria
Run this verification command from `e:\CPE310FacialDermatorsisProject` and paste its exact output into your handoff report:
```powershell
$env:PYTHONIOENCODING='utf-8'; python -c "from pathlib import Path; from fastapi.testclient import TestClient; from webapp.app import app; client = TestClient(app); r_health = client.get('/api/health'); print('Health:', r_health.status_code, r_health.json()); r_root = client.get('/'); r_css = client.get('/style.css'); r_js = client.get('/app.js'); r_mock = client.get('/static/mock_result.json'); print('Static status codes:', r_root.status_code, r_css.status_code, r_js.status_code, r_mock.status_code); img_bytes = Path('data/raw/skin-problems-34-on-the-iga-scale/images/104/1.jpg').read_bytes(); r_ok = client.post('/api/analyze', files={'file': ('test.jpg', img_bytes, 'image/jpeg')}); d = r_ok.json(); print('Analyze 200:', r_ok.status_code, d['prediction'], d['confidence'], 'info_name:', d['info']['name'], 'th_name:', d['info']['th']['name']); r_bad_ext = client.post('/api/analyze', files={'file': ('notes.txt', b'hello', 'text/plain')}); print('Bad ext:', r_bad_ext.status_code, r_bad_ext.json()['error_code']); r_corrupt = client.post('/api/analyze', files={'file': ('bad.jpg', b'not_jpeg_bytes', 'image/jpeg')}); print('Corrupt:', r_corrupt.status_code, r_corrupt.json()['error_code']); r_big = client.post('/api/analyze', files={'file': ('big.jpg', b'0' * (10 * 1024 * 1024 + 10), 'image/jpeg')}); print('Too large:', r_big.status_code, r_big.json()['error_code']); assert r_health.status_code == 200 and r_ok.status_code == 200 and r_bad_ext.status_code == 415 and r_corrupt.status_code == 400 and r_big.status_code == 413; print('ALL FASTAPI APP CHECKS PASSED')"
```
Expected: Prints `Health: 200`, `Static status codes: 200 200 200 200`, `Analyze 200: 200 acne ...`, `Bad ext: 415 invalid_file_type`, `Corrupt: 400 unreadable_image`, `Too large: 413 file_too_large`, and `ALL FASTAPI APP CHECKS PASSED`.

## 8. Stop Conditions
- If `webapp/service.py`, `webapp/knowledge_base.json`, or `webapp/static/index.html` is missing or raises an unexpected import error, **STOP** and report it in `tasks/task_w3_handoff_report.md`.

## 9. Report Instructions
When finished, write `tasks/task_w3_handoff_report.md` containing:
- **Status:** DONE / PARTIAL / BLOCKED
- **Files created or changed:** list with one line each
- **How I verified:** exact commands run and their actual output (pasted verbatim)
- **Deviations from the spec:** anything done differently, and why (write "none" if none)
- **Open issues or doubts:** anything the Elephant should check
- **Numbers to carry forward:** HTTP status codes for all endpoints/error cases and `/api/analyze` response latency via `TestClient`.

---

### Launch Prompt for Del (Copy-Paste into Goldfish Session)
```text
You are a Goldfish agent with no prior memory. Your entire job is in
tasks/task_w3_fastapi_app.md in e:\CPE310FacialDermatorsisProject.
Read that file completely, do exactly what it says, do not go outside the
files it lists, and when finished write the report it asks for. If anything
in the spec does not match reality, stop and say so in the report.
```
**What Del should check before pasting the report back:** The acceptance command printed `ALL FASTAPI APP CHECKS PASSED` with `Static status codes: 200 200 200 200` and `Analyze 200: 200 acne`.
