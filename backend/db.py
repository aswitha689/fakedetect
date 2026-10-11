import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent / "data.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS Statement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS Prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (statement_id) REFERENCES Statement(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Explanation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            weight REAL NOT NULL,
            direction TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES Prediction(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS KnowledgeFact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER NOT NULL,
            claim TEXT,
            rating TEXT,
            publisher TEXT,
            url TEXT,
            FOREIGN KEY (statement_id) REFERENCES Statement(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            user_label TEXT NOT NULL,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES Prediction(id) ON DELETE CASCADE
        );
        """)


init_db()


def insert_statement(text: str, status: str = "Submitted") -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO Statement (text, status) VALUES (?, ?)", (text, status))
        conn.commit()
        return cur.lastrowid


def update_statement_status(statement_id: int, status: str):
    with get_connection() as conn:
        conn.execute("UPDATE Statement SET status = ? WHERE id = ?", (status, statement_id))
        conn.commit()


def insert_prediction(statement_id: int, label: str, confidence: float) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Prediction (statement_id, label, confidence) VALUES (?, ?, ?)",
            (statement_id, label, confidence)
        )
        conn.commit()
        return cur.lastrowid


def update_prediction(prediction_id: int, label: str, confidence: float):
    with get_connection() as conn:
        conn.execute(
            "UPDATE Prediction SET label = ?, confidence = ? WHERE id = ?",
            (label, confidence, prediction_id)
        )
        conn.commit()


def insert_explanations(prediction_id: int, explanations: list[dict[str, Any]]):
    with get_connection() as conn:
        for exp in explanations:
            conn.execute(
                "INSERT INTO Explanation (prediction_id, word, weight, direction) VALUES (?, ?, ?, ?)",
                (prediction_id, exp.get("word", ""), exp.get("weight", 0.0), exp.get("direction", ""))
            )
        conn.commit()


def insert_knowledge_facts(statement_id: int, facts: list[dict[str, Any]]):
    with get_connection() as conn:
        for f in facts:
            conn.execute(
                "INSERT INTO KnowledgeFact (statement_id, claim, rating, publisher, url) VALUES (?, ?, ?, ?, ?)",
                (statement_id, f.get("claim", ""), f.get("rating", ""), f.get("publisher", ""), f.get("url", ""))
            )
        conn.commit()


def insert_feedback(prediction_id: int, user_label: str, comment: Optional[str] = None) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Feedback (prediction_id, user_label, comment) VALUES (?, ?, ?)",
            (prediction_id, user_label, comment)
        )
        conn.commit()
        return cur.lastrowid


def get_history(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.text, s.submitted_at, s.status,
                   p.id AS prediction_id, p.label, p.confidence
            FROM Statement s
            LEFT JOIN Prediction p ON p.statement_id = s.id
            ORDER BY s.id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def get_statement(statement_id: int) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM Statement WHERE id = ?", (statement_id,))
        statement = cur.fetchone()
        if not statement:
            return None

        result = dict(statement)

        cur.execute("SELECT * FROM Prediction WHERE statement_id = ? ORDER BY id DESC LIMIT 1", (statement_id,))
        pred = cur.fetchone()
        if pred:
            result["prediction"] = dict(pred)
            cur.execute("SELECT word, weight, direction FROM Explanation WHERE prediction_id = ?", (pred["id"],))
            result["explanation"] = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT id, user_label, comment, created_at FROM Feedback WHERE prediction_id = ?", (pred["id"],))
            result["feedback"] = [dict(r) for r in cur.fetchall()]
        else:
            result["prediction"] = None
            result["explanation"] = []
            result["feedback"] = []

        cur.execute("SELECT claim, rating, publisher, url FROM KnowledgeFact WHERE statement_id = ?", (statement_id,))
        result["knowledge_facts"] = [dict(r) for r in cur.fetchall()]

        return result


def delete_statement(statement_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM Statement WHERE id = ?", (statement_id,))
        if not cur.fetchone():
            return False
        cur.execute("DELETE FROM Statement WHERE id = ?", (statement_id,))
        conn.commit()
        return True
