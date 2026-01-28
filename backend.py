import uuid
import threading 
import os
import re
import json
from psycopg2.extras import Json, RealDictCursor
import base64
import time
import logging
import tempfile
from datetime import datetime
import time
from fastapi import Request
from typing import Dict, List
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from fastapi.responses import Response
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from image_processor2 import enhance_image_for_ocr
from dotenv import load_dotenv

load_dotenv()

import psycopg2
from psycopg2.extras import Json
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", 5432),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def insert_log(job_id, client_ip, api_client, filename, items_extracted, status, duration):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO logs (
            job_id, client_ip, api_client, filename,
            items_extracted, status, duration_sec
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        job_id, client_ip, api_client, filename,
        items_extracted, status, duration
    ))

    conn.commit()
    cur.close()
    conn.close()


def insert_document_data(job_id, filename, extracted_data, api_key):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO document_data (
            job_id, filename, extracted_data, api_key, status
        ) VALUES (%s, %s, %s, %s, 'Processing')
    """, (
        job_id,
        filename,
        Json(extracted_data),
        api_key
    ))

    conn.commit()
    cur.close()
    conn.close()


def update_document_status(job_id, filename, status):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE document_data
        SET status = %s
        WHERE job_id = %s AND filename = %s
    """, (status, job_id, filename))

    conn.commit()
    cur.close()
    conn.close()


def background_invoice_processing(job_id: str, filename: str, tmp_path: str, x_api_key: str):
    try:
        enhance_image_for_ocr(tmp_path)
        extracted_data = extract_invoice_from_path(tmp_path)
        items_count = len(extracted_data.get("items", []))

        # Update DB with extracted data and status Success
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE document_data
            SET extracted_data = %s, status = 'Success'
            WHERE job_id = %s AND filename = %s
        """, (Json(extracted_data), job_id, filename))
        conn.commit()
        cur.close()
        conn.close()

        # Log success
        insert_log(
            job_id=job_id,
            client_ip="background",
            api_client=VALID_API_KEYS[x_api_key],
            filename=filename,
            items_extracted=items_count,
            status="SUCCESS",
            duration=0
        )

    except Exception as e:
        # Mark as failed if any exception
        update_document_status(job_id, filename, "Fail")
        insert_log(
            job_id=job_id,
            client_ip="background",
            api_client=VALID_API_KEYS[x_api_key],
            filename=filename,
            items_extracted=0,
            status="FAILED",
            duration=0
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


##########################################
# LOGGING SETUP
##########################################
log_dir = os.path.join("logs", "invoice_api")
os.makedirs(log_dir, exist_ok=True)
##########################################
# LOGGING SETUP
##########################################
log_dir = os.path.join("logs", "invoice_api")
os.makedirs(log_dir, exist_ok=True)

main_log_file = os.path.join(log_dir, "invoice_api.log")
success_log_file = os.path.join(log_dir, "success.log")
error_log_file = os.path.join(log_dir, "error.log")

# Base logger
logger = logging.getLogger("invoice_api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Main log (everything)
main_handler = logging.FileHandler(main_log_file)
main_handler.setFormatter(formatter)

# Success log
success_handler = logging.FileHandler(success_log_file)
success_handler.setLevel(logging.INFO)
success_handler.setFormatter(formatter)

# Error log
error_handler = logging.FileHandler(error_log_file)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Console log
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(main_handler)
logger.addHandler(success_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)

##########################################
# API KEY AUTHENTICATION
##########################################
VALID_API_KEYS = {
    "a7f3c2e9b1d4567890abcdef1234567890abcd": "Mobile App Client",
}

##########################################
# WATSONX CONFIG
##########################################
API_KEY = os.getenv("IBM_API_KEY")
SERVICE_URL = os.getenv("IBM_SERVICE_URL")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
MODEL_ID = os.getenv("IBM_MODEL_ID")

if not all([API_KEY, SERVICE_URL, PROJECT_ID, MODEL_ID]):
    raise RuntimeError("One or more IBM Watsonx environment variables are missing!")

creds = Credentials(url=SERVICE_URL, api_key=API_KEY)
api_client = APIClient(creds)
api_client.set.default_project(PROJECT_ID)

model = ModelInference(api_client=api_client, model_id=MODEL_ID)

generation_params = {
    "max_new_tokens": 12000,
    "temperature": 0.05,
    "top_p": 0.95,
    "repetition_penalty": 1.15
}


invoice_prompt = """
You are an expert OCR-based invoice data extractor with 99% accuracy requirement.

Extract ALL visible information from this invoice image with MAXIMUM PRECISION.

═══════════════════════════════════════════════════════════════
🔴 CRITICAL EXTRACTION RULES
═══════════════════════════════════════════════════════════════

1. **ACCURACY FIRST**: If unsure about any field, return "" rather than guessing
2. **VERIFY TWICE**: Double-check each extracted value before finalizing
3. **LOCATION AWARENESS**: Know where to look for each field type
4. **NO ASSUMPTIONS**: Only extract what is CLEARLY VISIBLE

═══════════════════════════════════════════════════════════════
📋 INVOICE NUMBER & DATE (CRITICAL - 100% Required)
═══════════════════════════════════════════════════════════════

**invoiceNumber**:
- Location: Top section, usually labeled "Invoice No", "Bill No", "Inv#"
- Can be: Numeric, alphanumeric, or with special characters
- Common formats: "INV-001", "2024/123", "BILL001"
- If handwritten, extract carefully character by character
- Return "" ONLY if absolutely not visible

**invoiceNumberType**:
- "Printed" if in standard typed/printed font
- "Handwritten" if written by hand
- Look at font consistency and style

**invoiceDate**:
- Location: Near invoice number, labeled "Date", "Invoice Date", "Bill Date"
- Extract EXACTLY as shown (DD/MM/YYYY, DD-MM-YYYY, etc.)
- Do NOT reformat or change the date format
- Return "" if not visible

═══════════════════════════════════════════════════════════════
🏢 SUPPLIER INFORMATION (CRITICAL)
═══════════════════════════════════════════════════════════════

**DealerName**:
- Location: TOP of invoice (header section)
- Usually the LARGEST text or company logo text
- May be in bold or different font
- Extract the full business name
- Do NOT include address or contact details here

**dealerPhoneNumber**:
- PURPOSE:
  - Extract ALL visible dealer/supplier mobile or phone numbers related to the DEALER only

- PRIMARY LOCATION:
  - Immediately AFTER or NEAR the Dealer/Supplier Name at the TOP section of the invoice

- KEYWORDS TO IDENTIFY:
  - "Cell No", "Mobile No", "Mob", "Ph", "Phone", "Contact", "Tel"

- EXTRACTION RULES:
  - Extract ALL dealer phone numbers if multiple are visible
  - If only ONE number is visible, extract only that one
  - Preserve the EXACT format as printed
  - Allowed formats:
    - 10-digit Indian numbers
    - Numbers with country code (+91)
    - Spaces, commas, slashes, or hyphens allowed
  - Separate multiple numbers using a comma (`,`)

- STRICT EXCLUSIONS:
  - ❌ DO NOT extract customer phone numbers
  - ❌ DO NOT extract transporter, bank, or service center numbers

- SECONDARY LOCATION (FALLBACK – ONLY IF NOT FOUND AT TOP):
  - Look in the BOTTOM section of the invoice:
    - Inside company stamp
    - Near or below signature area

- VALIDATION:
  - Each extracted number must contain at least 10 digits
  - Ignore fax numbers unless explicitly labeled as phone/mobile

- OUTPUT:
  - Return a SINGLE STRING
  - Multiple numbers → comma-separated
  - Example:
    - "9876543210"
    - "+91 98765 43210, 080-23456789"

- RETURN:
  - Return "" if no dealer phone number is clearly visible



**DealerAddress**
- PRIMARY LOCATION:
  - Text BLOCK directly BELOW the Dealer/Supplier Name
  - May span multiple lines
  - Include:
    - Street
    - Area
    - City
    - State
    - PIN code
- SECONDARY LOCATION (Fallback):
  - If not present below dealer name:
  - Extract FULL address from:
    - Company stamp (if available)
- Extract address EXACTLY as shown
- Combine all visible address lines into a single string
- Return "" if not visible

**gstNumber** (GSTIN):
- MUST be EXACTLY 15 characters
- Should compulsarily match this Format: [0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z][A-Z][0-9A-Z]
- Example: 27AABCT1234F1Z5
- First 2 digits = State code (01-37)
- Validate format before returning
- Return "" if doesn't match pattern EXACTLY

═══════════════════════════════════════════════════════════════
👤 CUSTOMER INFORMATION
═══════════════════════════════════════════════════════════════

**customerName**:
- Location: Look for "Bill To", "Customer", "Buyer", "Sold To" section
- Usually in middle-left of invoice
- Extract full name (person or company)
- Return "" if not present

**customerPhone**:
- Look for phone/mobile number in customer section
- Format: Extract as-is (may include +91, spaces, hyphens)
- Must be 10 digits (Indian) or with country code
- Do NOT extract supplier's phone
- Return "" if not present

**customerAddress**:
- Complete address under customer/buyer section
- Include all lines (street, city, state, PIN)
- Return "" if not present

═══════════════════════════════════════════════════════════════
💰 FINANCIAL FIELDS
═══════════════════════════════════════════════════════════════

**downPayment**:
- Look for: "Down Payment", "DP", "Advance", "DP Amount" "Deposit"
-If it is handwritten also it should extract
- Extract numeric value only (no currency symbols)
- Return "" if not present
- This value is usually present near the "Hypothecation" stamp or word 

**netTotal**:
- Look for: "Grand Total", "Net Total", "Total Amount", "Amount Payable"
- Usually at BOTTOM of invoice in summary section
- Extract the FINAL total amount
- Return "" if not clearly visible

═══════════════════════════════════════════════════════════════
📦 ITEM TABLE EXTRACTION (CRITICAL FOR MULTI-LINE INVOICES)
═══════════════════════════════════════════════════════════════

🔴 COLUMN IDENTIFICATION (EXTREMELY IMPORTANT):
In the items table, columns typically appear in this order from LEFT to RIGHT:

Position 1-2: S.No / Item No / Description
Position 3-4: HSN/SAC Code / Brand / Quantity
Position 5: **RATE** (Unit Price) ← This is the price per single unit
Position 6-8: Tax Percentage Columns (SGST%, CGST%, IGST%)
Position 9-10: **AMOUNT** (Final Line Total) ← This is Rate × Qty ± Tax

🔴 CRITICAL RULES FOR RATE vs AMOUNT:
1. **RATE** is ALWAYS the unit price (price for 1 item)
2. **AMOUNT** is ALWAYS the final calculated value for that line item
3. For multi-line invoices, Amount is usually the RIGHTMOST numeric column
4. Rate column appears BEFORE tax percentage columns
5. Amount column appears AFTER tax percentage columns
6. If Quantity > 1, then Amount should be LARGER than Rate
7. Amount = Rate × Quantity × (1 + Tax%)

🔴 VISUAL CUES TO IDENTIFY COLUMNS:
- Rate column header: "Rate", "Unit Price", "Price", "Basic Rate", "MRP"
- Amount column header: "Amount", "Total", "Line Total", "Value", "Taxable Value"
- Rate values are typically SMALLER numbers
- Amount values are typically LARGER numbers (especially when qty > 1)
- Amount column is usually RIGHT-ALIGNED in the table

Extract EVERY row from the items table. For each item:

**itemNo**: Sequential number (1, 2, 3...) or as shown

**description**:
- Full item/product description
- Include model numbers, specifications
- Do NOT include IMEI/Serial numbers here

**brandName** (MANDATORY):
- Extract brand from description or separate field
- Common: Samsung, Apple, Vivo, Oppo, Realme, Xiaomi, OnePlus, TVS, Honda, Hero, Bajaj
- This is CRITICAL - try your best to identify
- Return "" only if absolutely no brand visible

**imeiNumber**:
- 15-digit number for mobile phones
- Look for "IMEI", "IMEI1", "IMEI2" labels
- Can be in description or separate field
- Return "" if not present

**serialNumber**:
- Look for "Serial No", "S.No", "Chassis No", "Engine No"
- Alphanumeric code
- Return "" if not present

**quantity**:
- Number of units
- Usually labeled "Qty", "Quantity", "Nos"
- Extract numeric value
- Return "" if not visible

**rate**:
- Unit price per item (NOT the final amount)
- Column labeled: "Rate", "Unit Price", "Price", "Basic Rate"
- Extract from ITEMS TABLE ONLY (not from summary)
- This is the price for ONE unit, not the total
- Look for the column that comes BEFORE tax columns
- Do NOT confuse with: Amount, Tax%, Quantity, HSN code
- For multi-line invoices, scan carefully from left to identify the rate column
- Return "" if not clearly visible in rate column

**itemAmount**:
- Final amount for this line item (Rate × Qty with or without tax)
- Column labeled: "Amount", "Total", "Line Total", "Value"
- This is the RIGHTMOST numeric column in most cases
- This value should be LARGER than rate when quantity > 1
- Extract the exact value shown in the Amount column
- Do NOT calculate - extract what is printed
- For multi-line invoices, this is CRITICAL - look for the rightmost column
- Return "" if not visible

═══════════════════════════════════════════════════════════════
💸 TAX EXTRACTION (CRITICAL - Percentage ONLY)
═══════════════════════════════════════════════════════════════

**VALID TAX %**: 0, 0.25, 1, 1.5, 3, 5, 6, 9, 12, 14, 18, 28, 1.46,

**RULES**:
1. Extract PERCENTAGE ONLY (not amounts)
2. Must see % symbol next to number
3. Value must be in valid list above
4. If not in list → return ""

**EXAMPLES**:

✅ "GST @ 18%" → sgst: 9, cgst: 9, igst: ""
✅ "IGST 12%" → sgst: "", cgst: "", igst: 12
✅ "SGST 9%, CGST 9%" → sgst: 9, cgst: 9, igst: ""
❌ "Tax: 2578.50" → sgst: "", cgst: "", igst: "" (this is amount, not %)

**sgst**: State GST percentage
**cgst**: Central GST percentage
**igst**: Integrated GST percentage

**IMPORTANT**:
- If SGST/CGST present → igst must be ""
- If IGST present → sgst and cgst must be ""
- example: If only "GST 18%" visible → sgst: (visible GST% /2), cgst: (visible GST% /2), igst: ""

═══════════════════════════════════════════════════════════════
🔖 STAMP & SIGNATURE DETECTION
═══════════════════════════════════════════════════════════════

**CRITICAL**: Scan BOTTOM 30% of invoice thoroughly!

**stampPresent**:
- Look for company stamp/seal (circular or rectangular)
- Common location: Bottom-right corner
- May be in RED, BLUE, or BLACK ink
- Can be faded or light
- Return "Present" or "Absent"

**informationInStamp**:
- Extract ALL text inside the company stamp
- May include company name, GSTIN, address
- Text may be curved or at angles
- Return "" if no stamp

**signaturePresent**:
- Look for handwritten signature
- Usually near stamp or at bottom
- Return "Yes" or "No"

**hypothecationStamp**:
- Look for "Hypothecated to..." text
- This is DIFFERENT from company stamp
- Return "Present" or "Absent"

**stampCompanyMatching_score**:
- Compare supplierName with informationInStamp
- Return similarity score 0-100
- Return 0 if no stamp

═══════════════════════════════════════════════════════════════
📤 OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return ONLY valid JSON. No markdown. No explanations. No extra text.

{
  "invoiceNumber": "",
  "invoiceNumberType": "",
  "invoiceDate": "",
  "DealerName": "",
  "DealerPhone": "",
  "DealerAddress": "",
  "EMIAmount": "",
  "gstNumber": "",
  "customerName": "",
  "customerPhone": "",
  "customerAddress": "",
  "downPayment": "",
  "netTotal": "",
  "stampPresent": "",
  "informationInStamp": "",
  "signaturePresent": "",
  "hypothecationStamp": "",
  "stampCompanyMatching_score": 0,
  "items": [
    {
      "itemNo": "",
      "Asset Model No": "",
      "brandName": "",
      "imeiNumber": "",
      "serialNumber": "",
      "quantity": "",
      "rate": "",
      "sgst": "",
      "cgst": "",
      "igst": "",
      "tax": "",
      "itemAmount": ""
    }
  ]
}

═══════════════════════════════════════════════════════════════
✅ FINAL CHECKLIST
═══════════════════════════════════════════════════════════════

Before returning JSON, verify:
- [ ] Invoice number extracted correctly
- [ ] All items from table included
- [ ] Tax values are PERCENTAGES (not amounts)
- [ ] GST number is 15 characters (if present)
- [ ] Stamp checked in bottom section
- [ ] For multi-line invoices: Rate and Amount columns identified correctly
- [ ] Rate is unit price, Amount is line total
- [ ] All "" for missing values (no null, no "N/A")


═══════════════════════════════════════════════════════════════
🧩 ADDITIONAL FIELD EXTRACTION (DYNAMIC KEY–VALUE PAIRS)
═══════════════════════════════════════════════════════════════

Objective:
Extract ALL other clearly visible information from the invoice that does NOT fit into the predefined fields, and return them as structured key–value pairs.
🔴 EXTRACTION RULES (STRICT)
VISIBILITY ONLY
Extract ONLY text that is clearly readable on the invoice
If unsure → return "" (do NOT guess)
NO DUPLICATION
Do NOT repeat values already extracted in predefined fields
Example:
GSTIN already in gstNumber → do NOT repeat it again
KEY NAMING RULES
Keys must be:
Meaningful
Human-readable
CamelCase
Examples:
"paymentMode"
"invoiceType"
"vehicleNumber"
"engineNumber"
"chassisNumber"
"ewayBillNumber"
"loanAccountNumber"
"financeCompany"
"bankName"
"accountNumber"
"ifscCode"
"placeOfSupply"
"stateCode"
"deliveryDate"
"dueDate"
"salesPersonName"
"referenceNumber"
VALUE RULES
Preserve exact formatting as printed
Do NOT normalize dates, numbers, or text
Extract full text for multi-line values as a single string
HANDWRITTEN TEXT
If handwritten and readable → extract
If unclear → return ""
TABLE-INDEPENDENT
These fields may appear:
In header
Side notes
Footer
Stamp
Free-text areas
"""

##########################################
# FASTAPI SETUP
##########################################
app = FastAPI(
    title="Invoice Extraction API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

##########################################
# REQUEST LOGGING MIDDLEWARE (HTTPS REQUESTS)
##########################################
@app.middleware("http")
async def job_context(request: Request, call_next):
    start_time = time.time()
    job_id = f"{datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:10]}"
    request.state.job_id = job_id
    request.state.start_time = start_time

    response = await call_next(request)

    response.headers["X-Job-Id"] = job_id
    return response

##########################################
# ROOT & HEALTH
##########################################
@app.get("/")
async def root():
    return {
        "service": "Invoice Extraction API",
        "version": "1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

##########################################
# ROBUST JSON PARSER
##########################################
def parse_json_robust(raw_text: str) -> Dict:
    try:
        return json.loads(raw_text)
    except:
        cleaned = re.sub(r"```json|```", "", raw_text).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end + 1])
    return {}


##########################################
# EXTRACTION AUDIT (HEADER + ITEMS)
##########################################
def audit_extracted_fields(data: Dict):
    header_extracted = []
    header_missing = []

    # -------- HEADER LEVEL --------
    for key, value in data.items():
        if key == "items":
            continue

        if value not in ("", None, [], {}):
            header_extracted.append(key)
        else:
            header_missing.append(key)

    # -------- ITEM LEVEL --------
    items = data.get("items", [])
    item_audit = []

    for idx, item in enumerate(items, start=1):
        extracted_fields = []
        missing_fields = []

        for field, value in item.items():
            if value not in ("", None):
                extracted_fields.append(field)
            else:
                missing_fields.append(field)

        item_audit.append({
            "item_no": idx,
            "extracted_fields": extracted_fields,
            "missing_fields": missing_fields
        })

    return {
        "header": {
            "extracted": header_extracted,
            "missing": header_missing
        },
        "items": {
            "count": len(items),
            "details": item_audit
        }
    }


##########################################
# CORE EXTRACTION
##########################################
def extract_invoice_from_path(image_path: str) -> Dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    mime = "image/jpeg" if image_path.lower().endswith(("jpg", "jpeg")) else "image/png"
    data_url = f"data:{mime};base64,{img_b64}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": invoice_prompt.strip()}
        ]
    }]

    for attempt in range(3):
        try:
            response = model.chat(messages=messages, params=generation_params)
            raw = response["choices"][0]["message"]["content"]
            return parse_json_robust(raw)
        except Exception as e:
            logger.error(f"OCR attempt {attempt+1} failed: {str(e)}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

##########################################
# API ENDPOINT
##########################################
@app.post("/extract-invoice")
async def extract_invoice_api(
    request: Request,
    files: List[UploadFile] = File(...),
    x_api_key: str = Header(None)
):
    job_id = request.state.job_id
    client_ip = request.client.host if request.client else "unknown"

    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    api_client_name = VALID_API_KEYS[x_api_key]
    response_payload = []

    for file in files:
        # 1️⃣ Insert row immediately with empty data + Processing
        insert_document_data(
            job_id=job_id,
            filename=file.filename,
            extracted_data={},  # empty initially
            api_key=x_api_key
        )

        # 2️⃣ Save file temporarily
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename)[1]
        )
        tmp_file.write(await file.read())
        tmp_file.close()

        # 3️⃣ Start background thread for extraction
        threading.Thread(
            target=background_invoice_processing,
            args=(job_id, file.filename, tmp_file.name, x_api_key),
            daemon=True
        ).start()

        # 4️⃣ Add file info to response
        response_payload.append({
            "filename": file.filename,
            "status": "Processing"
        })

    # 5️⃣ Immediate return with Job ID
    return {
        "job_id": job_id,
        "status": "Processing",
    }

##########################################
# NEW: CHECK JOB STATUS ENDPOINT
##########################################
@app.get("/check-job/{job_id}")
def check_job_status(
    job_id: str,
    x_api_key: str = Header(None)
):
    # 🔐 API key validation
    if not x_api_key:
        return {
            "error": "API key is required"
        }

    if x_api_key not in VALID_API_KEYS:
        return {
            "error": "Invalid API key"
        }

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("""
        SELECT filename, status, extracted_data
        FROM document_data
        WHERE job_id = %s
    """, (job_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {
            "jobId": job_id,
            "status": "not_found",
            "results": []
        }

    results = []
    for row in rows:
        results.append({
            "filename": row["filename"],
            "status": row["status"],
            "data": row["extracted_data"] or {}
        })

    overall_status = "processing"
    if all(r["status"] == "Success" for r in results):
        overall_status = "success"
    elif any(r["status"] == "Fail" for r in results):
        overall_status = "fail"

    return {
        "jobId": job_id,
        "status": overall_status,
        "count": len(results),
        "results": results
    }
