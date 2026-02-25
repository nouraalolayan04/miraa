import faiss
import numpy as np
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ===== Config =====
INDEX_FILE = "vector.index"
DATA_FILE = "data.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== In-Memory Storage =====
data_store = []
id_map = {}

# ===== Load / Init =====
def init_db():
    global index, data_store, id_map

    dim = 384
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(dim)

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data_store = json.load(f)
        id_map = {i: i for i in range(len(data_store))}
    else:
        data_store = []
        id_map = {}

# ===== Save =====
def save_all():
    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, "w") as f:
        json.dump(data_store, f)

# ===== Embedding =====
def embed(text):
    return model.encode([text])[0].astype("float32")

# ===== Save Interaction =====
def save_interaction(question, answer, model_id, image_b64=None):
    vec = embed(question + " " + answer)

    idx = len(data_store)

    data_store.append({
        "id": idx,
        "created_at": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "image": image_b64,
        "rating": None,
        "feedback_text": None
    })

    index.add(np.array([vec]))
    id_map[index.ntotal - 1] = idx

    save_all()
    return idx

# ===== Feedback =====
def update_feedback(interaction_id, rating, feedback_text=None):
    if 0 <= interaction_id < len(data_store):
        data_store[interaction_id]["rating"] = rating
        data_store[interaction_id]["feedback_text"] = feedback_text
        save_all()

# ===== List =====
def list_interactions(limit=20):
    return list(reversed(data_store))[:limit]
