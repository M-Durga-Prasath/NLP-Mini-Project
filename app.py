from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image
import pdfplumber
import os
import requests  # to call your model API

app = Flask(__name__)
CORS(app)  # allow all origins for development

# Flask route
@app.route("/summarize", methods=["POST"])
def summarize():
    text_input = request.form.get("text")
    file = request.files.get("file")
    extracted_text = ""

    # Case 1: Direct text input
    if text_input and text_input.strip():
        extracted_text = text_input.strip()

    # Case 2: File upload (OCR / PDF)
    elif file:
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        if file.filename.lower().endswith(".pdf"):
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() or ""
        else:
            img = Image.open(filepath)
            extracted_text = pytesseract.image_to_string(img, lang="tam")

        os.remove(filepath)
    else:
        return jsonify({"error": "No text or file provided"}), 400

    if not extracted_text.strip():
        return jsonify({"error": "No text extracted"}), 400

    # ---- Call the actual model backend API ----
    try:
        # Replace with your model backend URL
        MODEL_API_URL = "http://127.0.0.1:5001/summarize"  
        response = requests.post(MODEL_API_URL, json={"text": extracted_text})
        response_data = response.json()
        summary = response_data.get("summary", "⚠️ Model did not return a summary.")
    except Exception as e:
        return jsonify({"error": f"Failed to reach model backend: {str(e)}"}), 500

    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
