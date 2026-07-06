"""
Layer 3: 长期记忆 (Postgres pgvector + BM25 混合检索)
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager

try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    psycopg2 = None
    HAS_PG = False

import numpy as np
from config.settings import PostgresConfig, EmbeddingConfig, RetrievalConfig


class LongTermMemory:
    """Layer 3: 长期记忆"""

    def __init__(self, pg_config: PostgresConfig, embed_config: EmbeddingConfig,
                 retrieval_config: RetrievalConfig):
        self.pg_config = pg_config
        self.embed_config = embed_config
        self.retrieval_config = retrieval_config
        self._conn = None
        self._embedder = None
        self._bm25 = None
        self._bm25_texts: List[str] = []
        self._bm25_metadata: List[Dict] = []
        self._bm25_path = Path("data/bm25_index.pkl")
        self._init_db()

    @contextmanager
    def _cursor(self):
        if not HAS_PG:
            raise RuntimeError("psycopg2 未安装")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = psycopg2.connect(
                        host=self.pg_config.host, port=self.pg_config.port,
                        database=self.pg_config.database,
                        user=self.pg_config.user, password=self.pg_config.password
                    )
                cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    yield cur
                    self._conn.commit()
                    return
                finally:
                    cur.close()
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if self._conn:
                    self._conn.close()
                self._conn = None
                if attempt == max_retries - 1:
                    raise RuntimeError(f"PG 重连失败(已重试{max_retries}次): {e}") from e
                import time
                time.sleep(1.0 * (attempt + 1))  # 递增等待: 1s, 2s, 3s

    def _init_db(self):
        with self._cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pg_config.vector_table} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type VARCHAR(32) DEFAULT 'general',
                    source_session_id VARCHAR(64),
                    importance FLOAT DEFAULT 0.5,
                    embedding vector({self.pg_config.vector_dim}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

    # === Embedding ===

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_config.model)
            if self.embed_config.device != "cpu":
                self._embedder = self._embedder.to(self.embed_config.device)
        return self._embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=self.embed_config.normalize)

    # === CRUD ===

    def add(self, content: str, memory_type: str = "general",
            source_session_id: str = None, importance: float = 0.5,
            metadata: Dict = None) -> int:
        vec = self._encode([content])[0].tolist()
        with self._cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.pg_config.vector_table}
                    (content, memory_type, source_session_id, importance, embedding, metadata)
                VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
            """, (content, memory_type, source_session_id, importance, str(vec),
                  json.dumps(metadata or {}, ensure_ascii=False)))
            mem_id = cur.fetchone()["id"]
        self._bm25_texts.append(content)
        self._bm25_metadata.append({"id": mem_id, "memory_type": memory_type})
        self._rebuild_bm25()
        return mem_id

    def add_batch(self, items: List[Dict]) -> List[int]:
        if not items:
            return []
        texts = [item["content"] for item in items]
        vectors = self._encode(texts)
        ids = []
        with self._cursor() as cur:
            for i, item in enumerate(items):
                cur.execute(f"""
                    INSERT INTO {self.pg_config.vector_table}
                        (content, memory_type, source_session_id, importance, embedding, metadata)
                    VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
                """, (item["content"], item.get("memory_type", "general"),
                      item.get("source_session_id"), item.get("importance", 0.5),
                      str(vectors[i].tolist()),
                      json.dumps(item.get("metadata", {}), ensure_ascii=False)))
                mem_id = cur.fetchone()["id"]
                ids.append(mem_id)
                self._bm25_texts.append(item["content"])
                self._bm25_metadata.append({"id": mem_id, "memory_type": item.get("memory_type", "general")})
        self._rebuild_bm25()
        return ids

    # === Dense 检索 ===

    def dense_search(self, query: str, top_k: int = 20, memory_type: str = None) -> List[Dict]:
        query_vec = self._encode([query])[0].tolist()
        type_filter = f"AND memory_type = '{memory_type}'" if memory_type else ""
        with self._cursor() as cur:
            cur.execute(f"""
                SELECT id, content, memory_type, importance, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.pg_config.vector_table}
                WHERE 1=1 {type_filter}
                ORDER BY embedding <=> %s::vector LIMIT %s
            """, (str(query_vec), str(query_vec), top_k))
            rows = cur.fetchall()
        return [{"id": r["id"], "content": r["content"], "type": r["memory_type"],
                 "importance": r["importance"], "score": float(r["similarity"]),
                 "metadata": r["metadata"]} for r in rows]

    # === BM25 检索 ===

    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            import jieba
            if self._bm25_path.exists():
                with open(self._bm25_path, 'rb') as f:
                    data = pickle.load(f)
                self._bm25_texts = data.get("texts", [])
                self._bm25_metadata = data.get("metadata", [])
                tokenized = [list(jieba.cut(t)) for t in self._bm25_texts]
                self._bm25 = BM25Okapi(tokenized)
            else:
                self._bm25 = self._rebuild_bm25()
        return self._bm25

    def _rebuild_bm25(self):
        from rank_bm25 import BM25Okapi
        import jieba
        if not self._bm25_texts:
            self._bm25 = BM25Okapi([[""]])
        tokenized = [list(jieba.cut(t)) for t in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._bm25_path, 'wb') as f:
            pickle.dump({"texts": self._bm25_texts, "metadata": self._bm25_metadata}, f)
        return self._bm25

    def sparse_search(self, query: str, top_k: int = 20) -> List[Dict]:
        import jieba
        bm25 = self._get_bm25()
        tokenized = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized)
        if len(scores) == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        max_score = scores.max() if scores.max() > 0 else 1
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metadata[idx] if idx < len(self._bm25_metadata) else {}
            results.append({
                "id": meta.get("id", idx), "content": self._bm25_texts[idx],
                "type": meta.get("memory_type", "general"),
                "score": float(scores[idx] / max_score), "metadata": meta,
            })
        return results[:top_k]

    # === 混合检索 ===

    def hybrid_search(self, query: str, top_k: int = 10, memory_type: str = None) -> List[Dict]:
        cfg = self.retrieval_config
        dense_results = self.dense_search(query, cfg.top_k * 2, memory_type)
        sparse_results = self.sparse_search(query, cfg.top_k * 2)
        fused = self._rrf_fusion(dense_results, sparse_results,
                                 cfg.dense_weight, cfg.sparse_weight, cfg.rrf_k)
        return fused[:top_k]

    def _rrf_fusion(self, dense: List[Dict], sparse: List[Dict],
                    dense_weight: float = 0.6, sparse_weight: float = 0.4,
                    k: int = 60) -> List[Dict]:
        score_map = {}
        for rank, item in enumerate(dense):
            key = item["id"]
            score_map[key] = {"item": item, "score": dense_weight / (k + rank + 1)}
        for rank, item in enumerate(sparse):
            key = item["id"]
            rrf = sparse_weight / (k + rank + 1)
            if key in score_map:
                score_map[key]["score"] += rrf
            else:
                score_map[key] = {"item": item, "score": rrf}
        return [s["item"] for s in sorted(score_map.values(), key=lambda x: x["score"], reverse=True)]

    def get_context_for_query(self, query: str, max_tokens: int = 800) -> str:
        results = self.hybrid_search(query, top_k=self.retrieval_config.top_k)
        if not results:
            return ""
        lines = ["## 相关历史记忆"]
        for r in results[:self.retrieval_config.rerank_top_k]:
            lines.append(f"[{r['type']}] {r['content'][:300]}")
        return "\n".join(lines)

    def count(self) -> int:
        with self._cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {self.pg_config.vector_table}")
            return cur.fetchone()["cnt"]
