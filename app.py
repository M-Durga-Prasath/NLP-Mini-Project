from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from PIL import Image
import pytesseract
import pdfplumber

# ---- Import abstractive model ----
from transformers import pipeline
abstractive_summarizer = pipeline(
    "summarization",
    model="./IndicBART-XLSum",
    tokenizer="./IndicBART-XLSum",
    device=0,
    use_fast=False
)


# ---- Import extractive model ----
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def extractive_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))  # works for Tamil too
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in summary)


app = Flask(__name__)
CORS(app)

@app.route("/summarize", methods=["POST"])
def summarize():
    text_input = request.form.get("text")
    file = request.files.get("file")
    extracted_text = ""

    # Case 1: Direct text input
    if text_input and text_input.strip():
        extracted_text = text_input.strip()

    # Case 2: File upload
    elif file:
        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
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

    try:
        # ---- Abstractive summary ----
        abs_summary = abstractive_summarizer(
            extracted_text,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']

        # ---- Extractive summary ----
        ext_summary = extractive_summary(extracted_text, sentence_count=3)

    except Exception as e:
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

    return jsonify({
        "extractive_summary": abs_summary,
        "abstractive_summary": ext_summary
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
