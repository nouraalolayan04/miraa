import os
import io
import base64
import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for
from db import init_db, save_interaction, update_feedback, list_interactions
import time
from dotenv import load_dotenv

# =====================
# Load ENV
# =====================
load_dotenv()

def clean_model_id(raw: str, fallback: str) -> str:
  
    if not raw:
        return fallback
    raw = raw.strip()
    if raw.startswith("HF_MODEL_ID="):
        raw = raw.split("HF_MODEL_ID=", 1)[1].strip()
    # also handle accidental quotes
    raw = raw.strip('"').strip("'").strip()
    return raw or fallback


# Use HF_TOKEN (Router) not HF_API_KEY
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or "").strip()

# Allowed model from your router message
DEFAULT_MODEL ="Qwen/Qwen3-VL-8B-Instruct"
HF_MODEL_ID = clean_model_id(os.getenv("HF_MODEL_ID", ""), DEFAULT_MODEL)

print("HF_MODEL_ID =", repr(HF_MODEL_ID))
print("HF_TOKEN loaded =", bool(HF_TOKEN))

# =====================
# Flask
# =====================
app = Flask(__name__)
init_db()

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

# =====================
# Helpers
# =====================
def image_to_base64(file_storage) -> str:
    img = Image.open(file_storage.stream).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_hf_router_vlm(image_b64: str, question_ar: str) -> str:
    """
    Calls Hugging Face Router (OpenAI-compatible chat completions)
    with multimodal content: text + image_url.
    Prints latency + tokens in CMD.
    """
    if not HF_TOKEN:
        return "خطأ: لم يتم ضبط HF_TOKEN (أو HF_API_KEY) في ملف .env"

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "أجب باللغة العربية فقط. "
                    "إذا كان السؤال عن عدد أشخاص أو أشياء في الصورة، فقم بالعد مباشرة إذا كان ذلك ممكنًا من الصورة. "
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"السؤال: {question_ar}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 180,
        "temperature": 0.2,
    }

    # ===== Latency timing =====
    start_time = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    end_time = time.perf_counter()
    latency = round(end_time - start_time, 3)

    # ===== Helpful errors =====
    if r.status_code == 401:
        print(f"[{HF_MODEL_ID}] ❌ 401 Unauthorized | Latency: {latency}s")
        return "غير مصرح: تحقق من HF_TOKEN."
    if r.status_code == 400:
        print(f"[{HF_MODEL_ID}] ❌ 400 Bad Request | Latency: {latency}s")
        return f"خطأ (400): {r.text}"
    if r.status_code == 404:
        print(f"[{HF_MODEL_ID}] ❌ 404 Not Found | Latency: {latency}s")
        return "404: الموديل غير متاح عبر مزوّد حسابك. جرّبي موديل آخر."
    if not r.ok:
        print(f"[{HF_MODEL_ID}] ❌ Error {r.status_code} | Latency: {latency}s")
        return f"خطأ ({r.status_code}): {r.text}"

    data = r.json()

    try:
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", "N/A")
        prompt_tokens = usage.get("prompt_tokens", "N/A")
        completion_tokens = usage.get("completion_tokens", "N/A")

        print("======================================")
        print(f"MODEL      : {HF_MODEL_ID}")
        print(f"LATENCY    : {latency} seconds")
        print(f"TOKENS     : total={total_tokens} | prompt={prompt_tokens} | completion={completion_tokens}")
        print(f"ANSWER LEN : {len(answer)} chars")
        print("======================================")

        return answer

    except Exception:
        print(f"[{HF_MODEL_ID}] ⚠ Unexpected response | Latency: {latency}s")
        return f"استجابة غير متوقعة: {data}"

# =====================
# Routes
# =====================
@app.route("/", methods=["GET"])
def home():
    # Prevent opening /?image=...&question=... (forces clean UI behavior)
    if request.args.get("question") or request.args.get("image"):
        return redirect(url_for("home"))
    return render_template("index.html")

@app.route("/api/vqa", methods=["POST"])
def vqa():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "لم يتم إرسال صورة."}), 400

    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "اكتب السؤال أولاً."}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({"ok": False, "error": "اختر صورة صحيحة."}), 400

    try:
        image_b64 = image_to_base64(image_file)
    except Exception:
        return jsonify({"ok": False, "error": "تعذر قراءة الصورة. جرّب صورة أخرى."}), 400

    answer = call_hf_router_vlm(image_b64, question)


    interaction_id = save_interaction(
    question=question,
    answer=answer,
    model_id="",  # removed
    image_b64=image_b64  # store image
    )

    return jsonify({"ok": True, "answer": answer, "interaction_id": interaction_id})

@app.route("/api/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 20))
    items = list_interactions(limit=limit)
    return jsonify({"ok": True, "items": items})

@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    interaction_id = int(data.get("interaction_id"))
    rating = int(data.get("rating"))
    feedback_text = (data.get("feedback_text") or "").strip()

    if rating < 1 or rating > 5:
        return jsonify({"ok": False, "error": "التقييم يجب أن يكون بين 1 و 5."}), 400

    update_feedback(interaction_id, rating, feedback_text)
    return jsonify({"ok": True})




if __name__ == "__main__":
    app.run(debug=True)
