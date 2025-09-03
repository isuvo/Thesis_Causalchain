# src/preprocess/transformer_cache.py
import sqlite3, os, hashlib
import numpy as np

class EmbeddingCache:
    def __init__(self, db_path="work/embeddings/line_cache.sqlite"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )

    @staticmethod
    def _key(model_name: str, text: str) -> str:
        # model-aware key; trim to avoid whitespace dupes
        return hashlib.sha1((model_name + "||" + text.strip()).encode("utf-8")).hexdigest()

    def get_many(self, model_name, lines):
        keys = [self._key(model_name, s) for s in lines]
        if not keys:
            return [], {}
        q = ",".join("?" for _ in keys)
        cur = self.conn.execute(f"SELECT key, dim, vec FROM cache WHERE key IN ({q})", keys)
        found = {row[0]: np.frombuffer(row[2], dtype=np.float32) for row in cur.fetchall()}
        return keys, found

    def put_many(self, model_name, lines, vecs):
        rows = []
        for s, v in zip(lines, vecs):
            k = self._key(model_name, s)
            rows.append((k, v.shape[0], v.astype(np.float32).tobytes()))
        if rows:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO cache(key, dim, vec) VALUES (?, ?, ?)", rows
                )
