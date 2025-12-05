# INVOICE EXTRACTION

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/IBM-WatsonX-orange?logo=ibm" />
  <img src="https://img.shields.io/badge/OCR-Enabled-green" />
  <img src="https://img.shields.io/badge/Automation-Multi--Threaded-purple" />
  <img src="https://img.shields.io/badge/Excel-Export-success?logo=microsoft-excel" />
</p>

---

# 🚀 **Overview**

This project is a complete **AI-driven invoice extraction pipeline** that:

✔ Enhances raw images for OCR

✔ Extracts structured fields using **IBM WatsonX LLM**

✔ Performs **high-accuracy validation** (GST, phone, IMEI, dates, tax %)

✔ Handles **multi-line invoices**, tax logic & amount corrections

✔ Generates **Excel reports** with Item-Details & Invoice-Summary

✔ Calculates **confidence scores** for audit reliability

✔ Applies **proportional adjustment** when extracted totals mismatch

It is built for real-world business use where invoices vary widely in format, clarity, and structure.

---

# 📁 **Project Structure**

```
📦 Invoice-Extraction-System
│
├── Extractor.py                # Main pipeline: OCR→LLM→Validation→Excel export
├── image_processor.py          # Image enhancement (upscale, denoise, threshold)
│
├── INPUT_IMAGES/               # Raw invoice images
├── PROCESSED_IMAGES/           # Automatically enhanced images
│
├── extraction_log.txt          # Logs: warnings, corrections, retries
├── output.xlsx                 # Final Excel output (auto-generated)
│
└── README.md                   # Project documentation
```

---

# ✨ **Features**

## 🔧 **1. Advanced Image Preprocessing**

✔ Upscaling for better DPI

✔ Denoising, Grayscale conversion

✔ Adaptive thresholding

✔ Batch processing support and Automated folder processing

Improves OCR accuracy significantly for low-quality invoices.

---

## 🤖 **2. AI Extraction using IBM WatsonX**

The LLM extracts **structured JSON** including:

### **Header Fields**

* Invoice Number (with handwritten/printed detection)
* Invoice Date
* Supplier & Customer details
* GSTIN (validated)
* Stamp & Signature detection

### **Item Table Extraction**

* Item numbers
* Description (cleaned + brand identification)
* IMEI / Serial Number extraction
* Quantity, Rate, Amount
* SGST / CGST / IGST %
* Tax calculation correctness rules

---

## 🧠 **3. Multi-Layer Validation Engine**

✔ GST format & state-code validation

✔ IMEI (15-digit) verification

✔ Phone number & date validation

✔ Tax % correctness checks

✔ Cross-field corrections with logging

---

## 📤 **4. Clean Excel Reporting**

Exports 2 sheets:

### **Sheet 1 → Item_Details**

* One row per item
* All extracted + validated fields
* Excel table formatting

### **Sheet 2 → Invoice_Summary**

* One row per invoice
* Total amounts
* Confidence scores
* Stamp match score
* Tax-inclusive flag

---

# ⚙️ **Setup & Installation**

## 1️⃣ Install Python Packages

```sh
pip install opencv-python numpy pandas openpyxl ibm-watsonx-ai
```

---

## 2️⃣ Configure API & Paths

Inside `Extractor.py`, update:

```python
RAW_IMAGES_DIR = r".../INPUT_IMAGES"
PREPROCESSED_DIR = r".../PROCESSED_IMAGES"
EXCEL_OUTPUT_PATH = r".../output.xlsx"

API_KEY = "YOUR_API_KEY"
PROJECT_ID = "YOUR_PROJECT_ID"
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
```

---
## 3️⃣ Run the Complete Extraction Pipeline

```sh
python Extractor.py
```

---

# 🧪 **Output**

### ✔ Excel file with:

* Item-level details
* Invoice summary
* Confidence scores

### ✔ Logs include:

* Invalid GST fixes
* Phone/date corrections
* Amount mismatches
* Retry attempts

---
