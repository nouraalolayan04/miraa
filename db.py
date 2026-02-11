import sqlite3
from datetime import datetime

DB_PATH = "miraa.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        model_id TEXT NOT NULL,
        image_name TEXT,
        rating INTEGER,          -- 1..5
        feedback_text TEXT       -- optional
    );
    """)
    conn.commit()
    conn.close()

def save_interaction(question, answer, model_id, image_name=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interactions(created_at, question, answer, model_id, image_name)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), question, answer, model_id, image_name))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def update_feedback(interaction_id, rating, feedback_text=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE interactions
        SET rating = ?, feedback_text = ?
        WHERE id = ?
    """, (rating, feedback_text, interaction_id))
    conn.commit()
    conn.close()

def list_interactions(limit=20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at, question, answer, model_id, image_name, rating, feedback_text
        FROM interactions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
