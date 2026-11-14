
import os
import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from google import genai


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "ecosort_augmented.keras"
CLASS_PATH = BASE_DIR / "class_indices.json"
FRONTEND_PATH = BASE_DIR / "VAI hackathon.html"
DATABASE_PATH = BASE_DIR / "ecosort.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

CONFIDENCE_THRESHOLD = 70

app = FastAPI(
    title="EcoSort AI",
    description="AI-powered waste classification assistant"
)


# --------------------------------------------------
# 2. DATABASE INITIALIZATION
# --------------------------------------------------

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    try:

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mobilenet_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                gemini_category TEXT,
                comparison_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.commit()

    finally:

        connection.close()


initialize_database()


# --------------------------------------------------
# 3. LOAD MODEL AND CLASS LABELS
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_PATH.exists():
    raise FileNotFoundError(f"Class mapping not found: {CLASS_PATH}")

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_indices = json.load(file)

class_names = {
    index: name
    for name, index in class_indices.items()
}

print("EcoSort AI model loaded successfully.")
print("Classes:", list(class_indices.keys()))


# --------------------------------------------------
# 4. GEMINI CLIENT
# --------------------------------------------------

gemini_client = None

if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_API_KEY_HERE":

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    print("Gemini client configured.")

else:

    print("Gemini API key not configured. Using MobileNetV2 only.")


# --------------------------------------------------
# 5. MOBILE IMAGE CLASSIFICATION
# --------------------------------------------------

def classify_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))

    confidence = float(predictions[predicted_index]) * 100

    return {
        "category": class_names[predicted_index],
        "confidence": round(confidence, 2)
    }


# --------------------------------------------------
# 6. GEMINI IMAGE REVIEW
# --------------------------------------------------

def review_with_gemini(image):

    if gemini_client is None:
        return None

    prompt = """
You are an educational waste identification assistant.

Examine this image independently.

Choose exactly one category:

cardboard, glass, metal, paper, plastic, trash, or review.

Use "review" if the image contains mixed waste, is unclear,
or cannot reasonably be classified as one category.

Return ONLY valid JSON in this format:

{
  "category": "plastic",
  "explanation": "Brief explanation of what is visible.",
  "recommendation": "A practical, general disposal suggestion."
}

Do not invent local recycling facilities or claim that a
material is recyclable everywhere.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image]
    )

    response_text = response.text or ""

    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL
    )

    if not match:
        raise ValueError("Gemini did not return valid JSON.")

    result = json.loads(match.group())

    if not isinstance(result, dict):
        raise ValueError("Gemini returned an invalid response.")

    allowed_categories = set(class_indices.keys()) | {"review"}

    category = str(
        result.get("category", "review")
    ).lower().strip()

    if category not in allowed_categories:
        category = "review"

    return {
        "category": category,
        "explanation": str(result.get("explanation", "")),
        "recommendation": str(result.get("recommendation", ""))
    }


# --------------------------------------------------
# 7. COMPARE THE TWO AI SYSTEMS
# --------------------------------------------------

def compare_predictions(mobilenet, gemini):

    confidence = mobilenet["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:

        return {
            "status": "LOW_CONFIDENCE",
            "message": "The image classification confidence is low. Human review is recommended."
        }

    if gemini is None:

        return {
            "status": "REVIEW_REQUIRED",
            "message": "Only MobileNetV2 was available. A second opinion is recommended."
        }

    if gemini["category"] == "review":

        return {
            "status": "REVIEW_REQUIRED",
            "message": "Gemini could not confidently identify one material. Human review is recommended."
        }

    if mobilenet["category"] == gemini["category"]:

        return {
            "status": "AGREEMENT",
            "message": "Both AI systems identified the same category. This does not guarantee the prediction is correct."
        }

    return {
        "status": "DISAGREEMENT",
        "message": "The AI systems disagree. Human review is recommended."
    }


# --------------------------------------------------
# 8. GENERAL DISPOSAL GUIDANCE
# --------------------------------------------------

def get_recommendation(category):

    recommendations = {

        "cardboard": "Keep cardboard dry and flatten it. Check whether local recycling services accept it.",

        "glass": "Separate glass from other waste. Rinse it if appropriate and check local collection rules.",

        "metal": "Separate metal containers and rinse them where appropriate. Check local recycling rules.",

        "paper": "Keep paper clean and dry. Check whether local recycling services accept that paper type.",

        "plastic": "Check the plastic type and local recycling rules before placing it in a recycling collection.",

        "trash": "This item may not fit the recyclable categories. Check whether it can be separated into individual materials."

    }

    return recommendations.get(
        category,
        "Please inspect the item and seek human guidance."
    )


# --------------------------------------------------
# 9. SAVE SCAN TO DATABASE
# --------------------------------------------------

def save_scan_record(mobilenet_result, gemini_result, comparison):

    connection = sqlite3.connect(DATABASE_PATH)

    try:

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO scans (
                mobilenet_category,
                confidence,
                gemini_category,
                comparison_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (

            mobilenet_result["category"],

            mobilenet_result["confidence"],

            gemini_result["category"] if gemini_result else None,

            comparison["status"],

            datetime.now().isoformat(timespec="seconds")

        ))

        connection.commit()

    finally:

        connection.close()


# --------------------------------------------------
# 10. FRONTEND ROUTE
# --------------------------------------------------

@app.get("/")
async def home():

    if not FRONTEND_PATH.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend file not found."
        )

    return FileResponse(FRONTEND_PATH)


# --------------------------------------------------
# 11. IMAGE SCANNING API
# --------------------------------------------------

@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10 MB."
        )

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    try:

        # First AI: MobileNetV2
        mobilenet_result = classify_image(image)

        # Second AI: Gemini
        gemini_result = None
        gemini_error = None

        try:

            gemini_result = review_with_gemini(image)

        except Exception as error:

            gemini_error = str(error)

            print("Gemini review error:", gemini_error)

        # Compare predictions
        comparison = compare_predictions(
            mobilenet_result,
            gemini_result
        )

        # Save scan to analytics database
        save_scan_record(
            mobilenet_result,
            gemini_result,
            comparison
        )

        # Select disposal recommendation
        recommendation = (

            gemini_result["recommendation"]

            if gemini_result and gemini_result["recommendation"]

            else get_recommendation(
                mobilenet_result["category"]
            )

        )

        return {

            "success": True,

            "mobilenet": mobilenet_result,

            "gemini": gemini_result,

            "comparison": comparison,

            "recommendation": recommendation,

            "gemini_error": gemini_error

        }

    except Exception as error:

        print("Scanning error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing this image."
        )


# --------------------------------------------------
# 12. HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "project": "EcoSort AI",

        "model_loaded": True,

        "classes": list(class_indices.keys()),

        "gemini_available": gemini_client is not None

    }


# --------------------------------------------------
# 13. IMPROVED ANALYTICS API
# --------------------------------------------------

@app.get("/api/analytics")
async def get_analytics():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        # Total scans
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM scans
        """)
        total_scans = cursor.fetchone()["total"]

        # Confirmed agreements
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM scans
            WHERE comparison_status = 'AGREEMENT'
        """)
        agreements = cursor.fetchone()["total"]

        # Confirmed disagreements
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM scans
            WHERE comparison_status = 'DISAGREEMENT'
        """)
        disagreements = cursor.fetchone()["total"]

        # Completed comparisons
        confirmed_comparisons = agreements + disagreements

        # Agreement rate based only on completed comparisons
        agreement_rate = (
            round(
                (agreements / confirmed_comparisons) * 100,
                1
            )
            if confirmed_comparisons > 0 else None
        )

        # Gemini unavailable
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM scans
            WHERE gemini_category IS NULL
        """)
        gemini_unavailable = cursor.fetchone()["total"]

        # Scans requiring review
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM scans
            WHERE comparison_status != 'AGREEMENT'
        """)
        review_count = cursor.fetchone()["total"]

        # Waste category distribution
        cursor.execute("""
            SELECT mobilenet_category, COUNT(*) AS total
            FROM scans
            GROUP BY mobilenet_category
            ORDER BY total DESC
        """)
        distribution = {
            row["mobilenet_category"]: row["total"]
            for row in cursor.fetchall()
        }

        # Most recent 8 scans
        cursor.execute("""
            SELECT
                id,
                mobilenet_category,
                confidence,
                gemini_category,
                comparison_status,
                created_at
            FROM scans
            ORDER BY id DESC
            LIMIT 8
        """)

        recent_scans = [
            {
                "id": row["id"],
                "category": row["mobilenet_category"],
                "confidence": row["confidence"],
                "gemini_category": row["gemini_category"],
                "status": row["comparison_status"],
                "created_at": row["created_at"]
            }
            for row in cursor.fetchall()
        ]

        return {
            "success": True,
            "total_scans": total_scans,
            "agreement_rate": agreement_rate,
            "confirmed_comparisons": confirmed_comparisons,
            "agreements": agreements,
            "disagreements": disagreements,
            "gemini_unavailable": gemini_unavailable,
            "review_count": review_count,
            "category_count": len(class_indices),
            "distribution": distribution,
            "recent_scans": recent_scans
        }

    finally:
        connection.close()