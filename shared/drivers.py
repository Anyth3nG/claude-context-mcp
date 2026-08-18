"""
Storage drivers. ContextStore owns every rule; a driver owns only how records
are physically read and written.

WHY THIS SEAM EXISTS. ContextStore reached into a ChromaDB collection in 23
places, so replacing the storage engine meant rewriting the file that holds all
the semantics — splitting, archive thresholds, key gating, refusal shapes. That
is a rewrite pretending to be a migration, and it would leave two copies of the
rules to drift apart. With a driver, the port becomes a second implementation of
seven methods and the rules stay written once.

THE VOCABULARY IS NARROW ON PURPOSE. A filter is a conjunction of equality
tests plus one source exclusion — nothing else. That is the entire grammar
DynamoDB's SearchConditionExpression accepts (architecture/dynamodb-vector-search),
so a filter expressible here is portable by construction, and one that needed
more would have failed at the port instead of at review.

A RECORD is a plain dict: {"id", "document", "metadata"} and optionally
"embedding". Drivers return the same shape, plus "distance" from similar().

fetch_slot() TAKES BOTH AN ID AND THE COMPONENTS, which looks redundant and is
not. The store owns id construction, so the id is what identifies the record;
but a backend whose id lookup runs through an asynchronously maintained index
cannot read its own writes through it, and a summary read that returns the
previous version is not a stale cache — patch_summary computes its match, its
archive copy and its patch counter from that read, so every refusal passes and
the result is a self-consistent success that silently drops the intervening
edit. Handing over the components as well lets such a backend address the item
by its natural key instead and read it strongly consistently. A backend already
addressed by id ignores them.

DRIVERS OWN EMBEDDING. put() embeds any record that arrives without a vector,
and similar() embeds the query text. This keeps two things working that a
vector-only interface would have broken: Chroma's local-fallback path, where the
engine supplies its own default embedding function and no callable is exposed to
us; and batch embedding, where N documents in one put() must cost one API call
rather than N.
"""
from __future__ import annotations

from typing import Optional, Protocol


def chroma_where(
    project: Optional[str] = None,
    category: Optional[str] = None,
    type: Optional[str] = None,
    source: Optional[str] = None,
    key: Optional[str] = None,
    superseded_from: Optional[str] = None,
    exclude_sources: Optional[tuple] = None,
) -> Optional[dict]:
    """Translate the neutral filter vocabulary into a Chroma `where` clause."""
    clauses: list[dict] = []
    for field, value in (("project", project), ("category", category),
                         ("type", type), ("source", source), ("key", key),
                         ("superseded_from", superseded_from)):
        if value is not None:
            clauses.append({field: value})
    if exclude_sources:
        excluded = list(exclude_sources)
        clauses.append({"source": {"$ne": excluded[0]} if len(excluded) == 1
                        else {"$nin": excluded}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class StorageDriver(Protocol):
    """
    Eight methods. Anything a backend must provide, and nothing a backend should
    decide — no id construction, no archive policy, no key gating.
    """

    def put(self, records: list[dict]) -> None: ...
    def fetch(self, ids: list[str], *, with_embeddings: bool = False) -> list[dict]: ...
    def fetch_slot(self, record_id: str, *, project: Optional[str], category: str,
                   key: Optional[str],
                   with_embeddings: bool = False) -> Optional[dict]: ...
    def scan(self, filt: dict, *, with_documents: bool = True,
             with_embeddings: bool = False) -> list[dict]: ...
    def count(self, filt: Optional[dict] = None) -> int: ...
    def update_metadata(self, record_id: str, metadata: dict) -> None: ...
    def remove(self, ids: list[str]) -> None: ...
    def similar(self, text: str, filt: dict, top_k: int) -> list[dict]: ...


# Chroma Cloud refuses a Get asking for more than this, and silently returns at
# most this many when asked for everything — so a scan that trusts one call is
# wrong above the cap without saying so. See tasks/get-pagination.
CHROMA_PAGE = 300


class ChromaDriver:
    """The incumbent backend, wrapping one ChromaDB collection."""

    def __init__(self, collection):
        self.collection = collection

    # ---- writes ----------------------------------------------------------
    def put(self, records: list[dict]) -> None:
        if not records:
            return
        kwargs = {
            "ids": [r["id"] for r in records],
            "documents": [r["document"] for r in records],
            "metadatas": [dict(r["metadata"]) for r in records],
        }
        # All or nothing: Chroma cannot mix caller-supplied and auto-generated
        # embeddings in one upsert, so if any record lacks a vector, let the
        # collection embed the whole batch — one call, not one per document.
        embeddings = [r.get("embedding") for r in records]
        if all(e is not None for e in embeddings):
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)

    def update_metadata(self, record_id: str, metadata: dict) -> None:
        # update(), not upsert(): the document and its vector stay exactly as
        # they are, so nothing is re-embedded.
        self.collection.update(ids=[record_id], metadatas=[dict(metadata)])

    def remove(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    # ---- reads -----------------------------------------------------------
    def fetch(self, ids: list[str], *, with_embeddings: bool = False) -> list[dict]:
        if not ids:
            return []
        include = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
        got = self.collection.get(ids=ids, include=include)
        return self._records(got, with_embeddings)

    def fetch_slot(self, record_id: str, *, project: Optional[str], category: str,
                   key: Optional[str],
                   with_embeddings: bool = False) -> Optional[dict]:
        """
        Plain id lookup. A Chroma get() by id reads the collection itself rather
        than a secondary index, so it already returns the latest write and the
        natural-key components are unused here.
        """
        found = self.fetch([record_id], with_embeddings=with_embeddings)
        return found[0] if found else None

    def scan(self, filt: dict, *, with_documents: bool = True,
             with_embeddings: bool = False) -> list[dict]:
        include = ["metadatas"]
        if with_documents:
            include.append("documents")
        if with_embeddings:
            include.append("embeddings")
        where = chroma_where(**filt)
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []
        embs: list = []
        offset = 0
        while True:
            kwargs = {"limit": CHROMA_PAGE, "offset": offset, "include": include}
            if where is not None:
                kwargs["where"] = where
            page = self.collection.get(**kwargs)
            if not page["ids"]:
                break
            ids.extend(page["ids"])
            metas.extend(page.get("metadatas") or [])
            docs.extend(page.get("documents") or [])
            page_embs = page.get("embeddings")
            if page_embs is not None:
                embs.extend(page_embs)
            if len(page["ids"]) < CHROMA_PAGE:
                break
            offset += CHROMA_PAGE
        return [
            {
                "id": i,
                "document": docs[n] if with_documents and n < len(docs) else None,
                "metadata": metas[n] if n < len(metas) else {},
                **({"embedding": embs[n]} if with_embeddings and n < len(embs) else {}),
            }
            for n, i in enumerate(ids)
        ]

    def count(self, filt: Optional[dict] = None) -> int:
        if not filt:
            return self.collection.count()
        return len(self.scan(filt, with_documents=False))

    def similar(self, text: str, filt: dict, top_k: int) -> list[dict]:
        where = chroma_where(**filt)
        kwargs = {"query_texts": [text], "n_results": top_k}
        if where is not None:
            kwargs["where"] = where
        got = self.collection.query(**kwargs)
        ids = got.get("ids", [[]])[0]
        docs = (got.get("documents") or [[]])[0]
        metas = (got.get("metadatas") or [[]])[0]
        dists = (got.get("distances") or [[]])[0]
        return [
            {
                "id": i,
                "document": docs[n] if n < len(docs) else None,
                "metadata": metas[n] if n < len(metas) else {},
                "distance": dists[n] if n < len(dists) else None,
            }
            for n, i in enumerate(ids)
        ]

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _records(got: dict, with_embeddings: bool) -> list[dict]:
        embs = got.get("embeddings")
        out = []
        for n, i in enumerate(got["ids"]):
            rec = {
                "id": i,
                "document": got["documents"][n],
                "metadata": got["metadatas"][n],
            }
            if with_embeddings and embs is not None and n < len(embs):
                rec["embedding"] = embs[n]
            out.append(rec)
        return out
