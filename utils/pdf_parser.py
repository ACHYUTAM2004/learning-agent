import pdfplumber
import fitz  # PyMuPDF
import io
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_pdfplumber(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_pymupdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

def extract_text(file_bytes):
    """
    Extracts text from PDF bytes, prioritizing PyMuPDF for speed and
    falling back to pdfplumber for robustness.
    """
    primary_text = ""
    try:
        # 1. Try PyMuPDF first for its speed.
        logging.info("Attempting PDF parsing with PyMuPDF...")
        primary_text = extract_text_pymupdf(file_bytes)
    except Exception as e:
        logging.warning(f"PyMuPDF failed: {e}. Falling back to pdfplumber.")
        # 2. If PyMuPDF throws an error, fall back to pdfplumber.
        try:
            primary_text = extract_text_pdfplumber(file_bytes)
        except Exception as e2:
            logging.error(f"PDF parsing failed with both libraries. Last error: {e2}")
            return "" # Return empty if both fail

    # 3. If the primary attempt yielded very little text, try the other library.
    if len(primary_text) < 100:
        logging.info("Primary parsing resulted in short text. Trying fallback parser.")
        try:
            fallback_text = extract_text_pdfplumber(file_bytes)
            # Return the text from whichever library did a better job.
            if len(fallback_text) > len(primary_text):
                logging.info("Fallback parser (pdfplumber) provided a better result.")
                return fallback_text
        except Exception as e:
            logging.warning(f"Fallback parser (pdfplumber) also failed: {e}")

    return primary_text