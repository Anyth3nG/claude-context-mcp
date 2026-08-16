"""
Thin wrapper around ChromaDB implementing the schema in docs/schema.md.
Single collection, metadata-filtered queries. This is the module both
the MCP server and the backfill script will import — logic lives here once.
"""
from __future__ import annotations
import chromadb
import hashlib
import httpx
import numpy as np
import os
import time
from datetime import datetime, timezone
from typing import Optional

COLLECTION_NAME = "context_store"

# Locked taxonomy from docs/schema.md. Extending this is a one-line change
# here (not a data migration) — but it must be deliberate, not a typo that
# silently creates an invisible bucket search_context can never filter to.
VALID_CATEGORIES = {
    "tech_stack",
    "architecture",
    "config",
    "decisions",
    "preference",
    "fact",
    "tasks",
    "note",
}
VALID_TYPES = {"summary", "chunk"}
VALID_TIERS = {"client", "personal"}

# Retrieval budget. Keeps a single search_context call from dumping
# unbounded tokens into the conversation.
DEFAULT_TOP_K = 5
MAX_TOP_K = 10

# Where search_context clips a document. Raised from 800 on 2026-08-16, when
# archived summaries became ordinary search results (see SEARCH_HIDDEN_SOURCES).
# 800 was calibrated against the old population of hits — short, disciplined,
# single-fact chunks, which is what add_update's own guidance asks for. Ex-summaries
# are a different shape entirely, and 800 clipped them far harder than it ever
# clipped what it was tuned for.
MAX_DOC_CHARS = 1000

# The no-split buffer from decisions/chunk-size-buffer, now enforced in code
# rather than left to a caller's judgement. That decision set the policy for a
# HUMAN deciding whether to split as they wrote: don't split to save a few
# characters, because near-duplicate entries pollute retrieval worse than a
# clipped trailing clause does. _archive() has no caller in the loop to make that
# call, so the same tolerance has to be a number.
SPLIT_BUFFER_RATIO = 1.10
SPLIT_THRESHOLD = int(MAX_DOC_CHARS * SPLIT_BUFFER_RATIO)

# How close a typo must be to auto-correct (0-1, difflib ratio).
CATEGORY_MATCH_CUTOFF = 0.75

# Marks a summary that has been replaced. Archived copies keep the full text and,
# since 2026-08-16, are VISIBLE in ordinary search alongside any other chunk —
# they are history, and history is what the chunk tier is for. The provenance is
# still recorded (superseded_from names the slot it came from); it just no longer
# gates visibility.
SUPERSEDED_SOURCE = "superseded"

# Marks a chunk that turned out to be WRONG, as opposed to merely old. This is
# the ONLY source hidden from search by default, and the distinction is the
# reason: a superseded summary was true when written and remains valid history,
# while a retired chunk is a fact later shown to be incorrect. Handing the second
# back to a caller asking about the past would be answering with known-wrong
# material.
RETIRED_SOURCE = "retired"

# Hidden from search unless a caller opts back in. Narrowed from
# (superseded, retired) on 2026-08-16 — see below for why hiding both was wrong.
#
# Hiding them identically contradicted the store's own distinction between them,
# and it never achieved what it was for. The goal was to keep current-state
# questions free of contradiction, but a LIVE chunk that quietly goes stale (one
# that was never part of a summary, so never formally superseded) stayed fully
# visible forever — so the protection was never consistent. Worst case was
# archive_slot's "nothing replaces this": the only content that ever existed on a
# topic went invisible by default, a false negative strictly worse than the
# contradiction risk being guarded against.
SEARCH_HIDDEN_SOURCES = (RETIRED_SOURCE,)

# What index() leaves out of its history_chunks count — deliberately NOT the same
# set as SEARCH_HIDDEN_SOURCES, though the two were one constant until 2026-08-16.
# Search visibility and map arithmetic are different questions: the index already
# reports archived material separately as prior_versions and archived_slots, so
# counting superseded copies here as well would double-count them and make every
# edited slot look like the project had grown.
INDEX_EXCLUDED_SOURCES = (SUPERSEDED_SOURCE, RETIRED_SOURCE)

# A replacement shorter than this fraction of the current summary is refused.
# Catches the characteristic failure of overwrite-in-place: a caller that read a
# four-line summary, noticed one fact changed, and sends back only that fact —
# silently destroying the rest. Overridable, because deliberately condensing a
# bloated summary is legitimate.
SHRINK_GUARD_RATIO = 0.5

# When a patch archives a full copy of the pre-patch document.
#
# update_summary archives unconditionally because a replacement can destroy the
# whole slot. patch_summary structurally cannot: it rewrites exactly one matched
# substring, so its blast radius is bounded by len(old_str). The need for a
# recovery copy is therefore proportional to patch size, which is what the ratio
# encodes — a patch touching this fraction of the document is approaching a
# rewrite and gets archived like one.
#
# The counter exists because that reasoning holds per-patch but not in
# aggregate: twenty surgical edits, each far under the ratio, can still rewrite
# a document between archives. Forcing a copy every N patches bounds that drift.
# Both are cheap — archiving reuses the embedding Chroma already holds, so a
# forced archive costs no Voyage call.
PATCH_ARCHIVE_RATIO = 0.2
PATCH_ARCHIVE_EVERY = 5


class SummaryShrinkRefused(Exception):
    """
    Raised instead of performing a suspiciously destructive summary overwrite.
    Carries the current content so the caller can merge rather than guess.
    """

    def __init__(self, summary_id: str, previous: str, proposed: str):
        self.summary_id = summary_id
        self.previous = previous
        self.proposed = proposed
        super().__init__(
            f"Refusing to replace '{summary_id}': the new content is "
            f"{len(proposed)} chars against {len(previous)} currently stored "
            f"(under {int(SHRINK_GUARD_RATIO * 100)}%). A summary write REPLACES "
            "the whole slot — send the complete new state, not just what changed. "
            "Pass allow_shrink=True if the summary is genuinely being condensed."
        )


class UnknownSummaryKey(Exception):
    """
    Raised instead of silently creating a new sub-key alongside a similar one.

    Carries the category's existing keys so the caller can pick one rather than
    inventing a near-duplicate. This is deliberately a caller decision and not a
    similarity threshold: measured against this store, key-name embeddings do
    NOT separate synonyms from unrelated keys (lowest same-concept pair 0.538,
    highest different-concept pair 0.635 — the classes overlap), so any cutoff
    either misses lambda/compute or merges deploy/secrets. Content embeddings
    rank candidates well but still misroute, so they are used to ORDER the
    options here, never to choose among them.
    """

    def __init__(self, project: str, category: str, key: str, existing: list):
        self.project = project
        self.category = category
        self.key = key
        self.existing = existing
        shown = ", ".join(repr(k) if k else "(unkeyed)" for k in existing) or "none"
        super().__init__(
            f"'{key}' is not an existing key under {project}/{category}. "
            f"Existing keys, closest first: {shown}. Reuse one of those if it "
            "means the same thing — a near-duplicate key fragments the category "
            "silently and nothing later will notice. Pass create_key=True if "
            "this genuinely is a new topic."
        )


class MissingSummaryKey(Exception):
    """
    A summary write arrived without a key.

    Its own class rather than a bare ValueError so the tool layer can catch it
    precisely and answer with the keys already in use — the caller almost always
    meant one of them. A blanket `except ValueError` there would also swallow a
    bad tier or an unusable key slug and report all three as the same thing.
    """

    def __init__(self, project: str, category: str, existing: list):
        self.project = project
        self.category = category
        self.existing = existing
        shown = ", ".join(repr(k) for k in existing if k) or "none yet"
        super().__init__(
            f"key is required — every summary lives under one (project, category, key). "
            f"A keyless slot in '{category}' would make get_context(project, category) "
            f"ambiguous between the category's own summary and everything filed under it. "
            f"Keys already in use under {project}/{category}: {shown}."
        )


class PatchFailed(Exception):
    """
    Base for every reason a patch was not applied.

    All of them carry `current` — the document as it actually stands. A patch
    fails precisely when the caller's idea of the text has drifted from the
    stored text, so an error string alone leaves them guessing at the same
    wrong content again. Same reasoning as SummaryShrinkRefused: refuse, and
    hand back what's needed to retry correctly.
    """

    def __init__(self, summary_id: str, current: Optional[str], message: str):
        self.summary_id = summary_id
        self.current = current
        super().__init__(message)


class ChunkNotFound(Exception):
    """No entry at that id. Ids come back from search_context alongside each hit."""

    def __init__(self, chunk_id: str):
        self.chunk_id = chunk_id
        super().__init__(
            f"No entry with id '{chunk_id}'. Chunk ids are returned by search_context "
            "with each result — copy one from there rather than constructing it."
        )


class PatchSlotMissing(PatchFailed):
    """Nothing to patch — the slot has no summary yet."""

    def __init__(self, summary_id: str):
        super().__init__(
            summary_id,
            None,
            f"No summary at '{summary_id}' to patch. Use update_summary to create it.",
        )


class PatchNoMatch(PatchFailed):
    """old_str does not occur in the stored document."""

    def __init__(self, summary_id: str, current: str, old_str: str):
        self.old_str = old_str
        super().__init__(
            summary_id,
            current,
            f"No occurrence of the given text in '{summary_id}'. The stored "
            "document is returned so the patch can be re-derived from what is "
            "actually there.",
        )


class PatchAmbiguous(PatchFailed):
    """old_str occurs more than once — which one was meant is unknowable."""

    def __init__(self, summary_id: str, current: str, old_str: str, count: int):
        self.old_str = old_str
        self.count = count
        super().__init__(
            summary_id,
            current,
            f"The given text occurs {count} times in '{summary_id}'. Patching "
            "the wrong one is silent and near-impossible to spot afterwards, so "
            "extend old_str with surrounding text until it identifies exactly "
            "one place.",
        )


class PatchNoOp(PatchFailed):
    """old_str and new_str are identical — applying this would change nothing."""

    def __init__(self, summary_id: str):
        super().__init__(
            summary_id,
            None,
            f"old_str and new_str are identical, so this patch would leave "
            f"'{summary_id}' unchanged. Refused rather than reporting a "
            "successful write that did nothing.",
        )


DEFAULT_VOYAGE_MODEL = "voyage-3.5"


def _normalize_category(category: str) -> tuple[str, Optional[str]]:
    """
    Returns (normalized_category, corrected_from). corrected_from is None if
    the input was already valid, otherwise it's the original (wrong) input —
    so callers can surface that a correction happened rather than silently
    swallowing it. Raises only when nothing is a close enough match, since
    guessing at that distance risks filing something under the wrong
    category with no way to tell.
    """
    if category in VALID_CATEGORIES:
        return category, None
    import difflib

    matches = difflib.get_close_matches(
        category, VALID_CATEGORIES, n=1, cutoff=CATEGORY_MATCH_CUTOFF
    )
    if matches:
        return matches[0], category
    raise ValueError(
        f"Unknown category '{category}' — no close match found. "
        f"Valid: {sorted(VALID_CATEGORIES)}"
    )


MAX_KEY_CHARS = 40


def _normalize_key(key: Optional[str]) -> Optional[str]:
    """
    Slugify a sub-key: lowercase, hyphen-separated, alphanumerics only.

    Unlike categories there is no fixed set to correct against, so this only
    removes the differences that are unambiguously meaningless — case, spacing,
    underscores, punctuation. `Lambda Settings`, `lambda_settings` and
    `lambda-settings` are the same slot; `lambda` and `compute` are not, and no
    amount of string handling can decide whether they should be. That judgment
    is pushed to the caller by UnknownSummaryKey instead.

    None passes through rather than raising, because this sits on read paths too
    and a keyless lookup should return not-found, not blow up. Writes are where
    the requirement bites: update_summary rejects a None key outright.
    """
    if key is None:
        return None
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", key.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"Key '{key}' contains nothing usable — keys must have letters or digits.")
    if len(slug) > MAX_KEY_CHARS:
        raise ValueError(
            f"Key '{slug}' is {len(slug)} characters, over the {MAX_KEY_CHARS} limit. "
            "Keys name a topic ('cognito', 'deploy'); they are not descriptions."
        )
    if slug == "summary":
        # Under the old SUFFIX id form this genuinely collided with the unkeyed
        # id f"{project}-{category}-summary". The prefix form no longer collides,
        # but the reservation stays: a slot addressed as summary-{p}-{c}-summary
        # reads as a formatting mistake, and "summary" names no topic anyway.
        raise ValueError("'summary' is reserved and cannot be used as a key.")
    return slug


def _split_for_archive(document: str) -> list[str]:
    """
    Cut an oversized document into retrieval-sized pieces at paragraph
    boundaries. Returns [document] unchanged when it fits, which is the common
    case and costs nothing.

    MECHANICAL, NOT SEMANTIC, and that is the constraint that shapes it.
    add_update can ask a caller "is this a second distinct fact?" because a
    caller is present. _archive() runs automatically with nobody in the loop, so
    this only uses a signal already in the text — the blank line the author
    already put there — and never invents a boundary.

    Paragraphs are packed GREEDILY rather than one per piece. One paragraph per
    piece would shred a document of short paragraphs into a pile of fragments,
    which is the near-duplicate problem decisions/chunk-size-buffer warns about
    arriving by a different route.

    A single paragraph longer than the threshold comes back whole. Splitting it
    would mean cutting mid-sentence on a character count, and a piece that starts
    partway through a clause is worse than one that gets clipped in display —
    the clip is at least visible as a clip. Rare in practice, since the documents
    this handles are archived summaries, which are written in paragraphs.
    """
    if len(document) <= SPLIT_THRESHOLD:
        return [document]

    pieces: list[str] = []
    current = ""
    for paragraph in document.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > SPLIT_THRESHOLD:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageRestEmbedding:
    """
    Real embedding function for actual use (decision: Voyage, not local).

    Deliberately calls Voyage's REST endpoint directly instead of using the
    `voyageai` SDK. Same endpoint, same request, same vectors — but the SDK
    declares pillow, tokenizers, hf_xet and aiohttp for multimodal and local
    tokenizer features this project never touches, ~90MB of dependencies. That
    matters because the deploy target is Lambda, where the whole bundle has to
    fit in 250MB unzipped; dropping the SDK took it from 200MB to 110MB.

    Chroma's embedding-function protocol is duck-typed — a callable taking a
    list of texts and returning a list of vectors, plus name() — so this drops
    straight into the (embedding_function, embedding_function_name) seam.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_VOYAGE_MODEL,
        timeout: float = 30.0,
        max_retries: int = 4,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        # One client, reused: on Lambda this lives at module scope across warm
        # invocations, so a per-call TLS handshake would be pure waste. It also
        # opens no socket at construction, which is what lets SnapStart
        # snapshot this object without freezing a dead connection into the
        # image (no after_restore hook needed).
        self._client = httpx.Client(timeout=timeout)

    def name(self) -> str:
        return self.model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        # A 429 raises straight out of collection.upsert()/query(), so without
        # this a rate limit surfaces as a failed write rather than a slow one.
        # Backoff starts at the length of a rate-limit window, not 1-2s —
        # short retries just re-hit the same closed window. Note Voyage
        # throttles an account with no payment method to 3 RPM regardless of
        # tier; with one, voyage-3.5 allows 2000 RPM.
        delay = 25.0
        for attempt in range(self.max_retries):
            resp = self._client.post(
                VOYAGE_EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(input)},
            )
            if resp.status_code != 429 or attempt == self.max_retries - 1:
                break
            time.sleep(float(resp.headers.get("retry-after", delay)))
            delay = min(delay * 2, 60.0)
        resp.raise_for_status()
        data = resp.json()["data"]
        # Sort defensively rather than trusting response order — a silent
        # misalignment here attaches the wrong vector to the wrong document
        # and is near-impossible to debug afterwards.
        ordered = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        # MUST be numpy, not plain lists. Chroma's HTTP/Cloud client calls
        # .tolist() on query embeddings (api/fastapi.py::_query ->
        # convert_np_embeddings_to_list); plain lists raise AttributeError
        # there. Writes tolerate lists and the local PersistentClient never
        # takes that path, so this breaks ONLY on reads against Cloud.
        return [np.array(vec, dtype=np.float32) for vec in ordered]

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)


def _make_client(persist_path: str, use_cloud: Optional[bool] = None):
    """
    Chroma Cloud when the CHROMA_* trio is configured (the deployed path —
    Lambda has no persistent disk to put a database on), local PersistentClient
    otherwise (stdio dev, offline work).

    `use_cloud` overrides the env sniffing: False forces a local store even
    when Cloud credentials are present. Tests MUST pass False — otherwise a
    developer with a populated .env runs the smoke suite straight into the
    real shared store and writes fixture data into it.

    Partial configuration raises rather than quietly falling back to a local
    store: an incomplete Cloud config almost always means a deploy is
    misconfigured, and silently writing to a local disk instead would look
    like it worked while the entries went nowhere anyone else can read.

    NOTE: PersistentClient requires the full `chromadb` package. The Lambda
    bundle installs `chromadb-client`, which exposes the name but raises
    "http-only client mode" on construction — so the deployed path must always
    resolve to Cloud. See requirements.txt vs requirements-lambda.txt.
    """
    tenant = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")
    api_key = os.environ.get("CHROMA_API_KEY")
    provided = [n for n, v in
                (("CHROMA_TENANT", tenant), ("CHROMA_DATABASE", database),
                 ("CHROMA_API_KEY", api_key)) if v]

    if use_cloud is False:
        return chromadb.PersistentClient(path=persist_path)
    if provided and len(provided) < 3:
        missing = {"CHROMA_TENANT", "CHROMA_DATABASE", "CHROMA_API_KEY"} - set(provided)
        raise RuntimeError(
            f"Partial Chroma Cloud config: {', '.join(sorted(provided))} set but "
            f"{', '.join(sorted(missing))} missing. Set all three to use Cloud, "
            "or none to use a local store."
        )
    if use_cloud and not provided:
        raise RuntimeError(
            "use_cloud=True but no CHROMA_TENANT / CHROMA_DATABASE / CHROMA_API_KEY "
            "are set."
        )
    if provided:
        return chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    return chromadb.PersistentClient(path=persist_path)


class ContextStore:
    def __init__(
        self,
        persist_path: str = "./chroma_data",
        embedding_function=None,
        embedding_function_name: Optional[str] = None,
        allow_local_fallback: bool = False,
        use_cloud: Optional[bool] = None,
    ):
        """
        Voyage is REQUIRED by default (see docs/schema.md decision) — this
        raises loudly rather than silently falling back to Chroma's weaker
        local default, so the "Voyage because retrieval quality matters"
        decision can't be undone by someone forgetting a kwarg.

        - Normal use: set VOYAGE_API_KEY in the environment, call with no args.
        - Explicit override (e.g. offline dev/testing): pass both
          embedding_function AND embedding_function_name.
        - Explicit, deliberate opt-out of Voyage: allow_local_fallback=True.
        """
        if embedding_function is None:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if api_key:
                embedding_function = VoyageRestEmbedding(api_key)
                embedding_function_name = DEFAULT_VOYAGE_MODEL
            elif allow_local_fallback:
                embedding_function = None  # Chroma's built-in local default
                embedding_function_name = embedding_function_name or "local-default"
            else:
                raise RuntimeError(
                    "No VOYAGE_API_KEY set and no embedding_function provided. "
                    "This project deliberately chose Voyage for retrieval quality "
                    "(see docs/schema.md) — silently falling back to the weaker "
                    "local model would undo that decision. Set VOYAGE_API_KEY, "
                    "pass an explicit embedding_function + embedding_function_name, "
                    "or pass allow_local_fallback=True if you're intentionally "
                    "opting out."
                )
        elif embedding_function_name is None:
            raise ValueError(
                "embedding_function_name is required when passing a custom "
                "embedding_function, so the collection can record what created "
                "it and detect a mismatch if reopened differently later."
            )

        self.client = _make_client(persist_path, use_cloud)
        self.collection = self._open_or_create_collection(
            embedding_function, embedding_function_name
        )

    def _open_or_create_collection(self, embedding_function, embedding_function_name):
        existing_names = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing_names:
            get_kwargs = {}
            if embedding_function is not None:
                get_kwargs["embedding_function"] = embedding_function
            collection = self.client.get_collection(COLLECTION_NAME, **get_kwargs)
            existing_tag = (collection.metadata or {}).get("embedding_function_name")
            if existing_tag and existing_tag != embedding_function_name:
                raise RuntimeError(
                    f"This collection was created with embedding function "
                    f"'{existing_tag}', but you're opening it with "
                    f"'{embedding_function_name}'. Mixing embedding spaces "
                    "silently corrupts similarity search — either match the "
                    "original embedding function or use a fresh persist_path."
                )
            return collection
        else:
            create_kwargs = {"metadata": {"embedding_function_name": embedding_function_name}}
            if embedding_function is not None:
                create_kwargs["embedding_function"] = embedding_function
            return self.client.get_or_create_collection(COLLECTION_NAME, **create_kwargs)

    def save(
        self,
        document: str,
        category: str,
        type: str,  # "summary" | "chunk"
        project: Optional[str] = None,  # None -> "general"
        tier: Optional[str] = None,  # required if project set
        source: str = "live",
        chat_title: Optional[str] = None,
    ):
        category, corrected_from = _normalize_category(category)
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES}, got '{type}'")

        project_key = project or "general"
        if not project:
            tier = None  # tier is meaningless without a project — don't let it leak in
        elif tier and tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        elif not tier:
            raise ValueError("tier is required when project is set")

        # Upsert semantics: summaries are ONE living document per
        # project+category — deterministic id, always overwritten. Chunks
        # accumulate as a raw fallback layer and are never edited in place.
        if type == "summary":
            id = self.summary_id(project, category)
        else:
            id = self.chunk_id(document, project_key, category)

        metadata = {
            "project": project_key,
            "category": category,
            "type": type,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tier:
            metadata["tier"] = tier
        if chat_title:
            metadata["chat_title"] = chat_title
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from

        self.collection.upsert(ids=[id], documents=[document], metadatas=[metadata])
        return {"id": id, "corrected_from": corrected_from, **metadata}

    def summary_id(self, project: Optional[str], category: str, key: Optional[str] = None) -> str:
        """
        The deterministic slot id — one living document per project+category(+key).

        PREFIX-based: `summary-{project}-{category}` or, keyed,
        `summary-{project}-{category}-{key}`. Chunk ids already announce their
        type from the first token — `chunk-<hash>`, `superseded-<hash>` — and
        this used to be the one exception, a SUFFIX form requiring an
        "ends with -summary" check instead of the "starts with X-" rule used
        everywhere else. That asymmetry surfaced concretely while designing a
        merged archive()/retire tool (tasks/tool-surface): target-type dispatch
        had to special-case the suffix form. This makes id-based dispatch
        uniform across all three types.

        UNLIKE sub-keys, this is NOT byte-compatible with what came before —
        every existing summary id changes shape, keyed or not. That is why it
        shipped with a one-time rename migration (scripts/summary_id_prefix.py)
        rather than the lazy, additive rollout sub-keys got: an old-format id
        left unmigrated would silently stop matching this function's output,
        and the next update_summary/patch_summary against it would create a
        SECOND, duplicate live document instead of updating the original.
        """
        base = f"{project or 'general'}-{category}"
        return f"summary-{base}-{key}" if key else f"summary-{base}"

    def summary_keys(self, project: Optional[str], category: str) -> list[Optional[str]]:
        """
        Every key currently in use under one project+category, `None` first if an
        unkeyed slot exists.

        Since 2026-08-12 update_summary refuses a keyless write, so `None` can no
        longer appear on anything written after that date. The handling stays for
        data predating the migration: a store restored from an older backup still
        reads correctly, and a `None` in this list is the signal that it has not
        been migrated rather than a crash.
        """
        category, _ = _normalize_category(category)
        got = self.collection.get(
            where={"$and": [{"project": project or "general"}, {"category": category},
                            {"type": "summary"}]},
            include=["metadatas"],
        )
        keys = {meta.get("key") or None for meta in got["metadatas"]}
        return ([None] if None in keys else []) + sorted(k for k in keys if k)

    def summary_keys_ranked(
        self, project: Optional[str], category: str, content: str
    ) -> list[Optional[str]]:
        """
        Existing keys in a category, ordered by how close each slot's CONTENT is
        to `content` — closest first.

        Ranked by content rather than by key name, because measurement says key
        names don't work: embedded name-to-name, the lowest same-concept pair
        (0.538) sits below the highest unrelated pair (0.635), so the classes
        overlap and no cutoff separates them. Content embeddings put the right
        slot on top in most cases and, more importantly, put it near the top in
        all of them — which is all a list of suggestions needs to do.

        This is a suggestion, never a decision. It costs one query (one embedding
        call) and only runs on the refusal path, where a caller is about to
        create a key and needs to see what it might already be.
        """
        category, _ = _normalize_category(category)
        existing = self.summary_keys(project, category)
        if not existing:
            return []
        got = self.collection.query(
            query_texts=[content],
            n_results=len(existing),
            where={"$and": [{"project": project or "general"}, {"category": category},
                            {"type": "summary"}]},
        )
        ranked = [meta.get("key") or None for meta in got["metadatas"][0]]
        # Anything the query didn't return still belongs in the list, just last.
        return ranked + [k for k in existing if k not in ranked]

    def chunk_id(
        self, document: str, project_key: str, category: str, key: Optional[str] = None
    ) -> str:
        """
        Content-addressed chunk id: same text, same project, same category, same
        key -> same id, forever.

        This used to mix datetime.now().timestamp() into the hash, which made
        every write unique — and therefore made a re-run after a partial failure
        DUPLICATE everything it had already written instead of resuming. That is
        fatal for a bulk backfill, where a partial failure is the expected case,
        not the exceptional one.

        The trade-off is deliberate: writing genuinely identical text twice into
        the same project+category+key now collapses to one entry rather than two.
        That is the correct reading — a chunk carries no position or ordering,
        so a byte-identical duplicate holds no information the first one didn't,
        and its only effect on retrieval is to occupy a second slot in top_k.

        `key` is OPTIONAL, same reasoning as summary_id: its absence reproduces
        the original id byte for byte, so widening the hash to include it is
        additive rather than a migration. The same text filed under two
        different keys now gets two distinct ids instead of colliding into one.
        """
        raw = f"{project_key}-{category}-{key}-{document}" if key else f"{project_key}-{category}-{document}"
        return f"chunk-{hashlib.sha1(raw.encode()).hexdigest()[:16]}"

    def save_chunks(
        self,
        documents: list[str],
        category: str,
        project: Optional[str] = None,
        tier: Optional[str] = None,
        source: str = "live",
        chat_title: Optional[str] = None,
        timestamp: Optional[str] = None,
        key: Optional[str] = None,
    ) -> dict:
        """
        Write several chunks in ONE upsert, and therefore one embedding call.

        This is what makes splitting a long fact at write time free. Voyage
        bills and rate-limits per REQUEST, not per document, and Chroma issues
        exactly one embed call per upsert regardless of how many documents it
        carries — so five focused chunks cost the same as one sprawling one.
        Without this, splitting would trade a retrieval problem for five times
        the write latency, and nobody would do it.

        Byte-identical documents within a single call collapse (same
        content-addressed id), so the returned ids may be shorter than the
        input list. `oversized` names the entries that still exceed the display
        cap, so a caller can be told its split didn't go far enough.

        `timestamp` exists for rewriting entries that already have a history —
        splitting an old entry into retrieval-sized pieces must not make those
        pieces look like they were learned today. Live writes leave it unset and
        get the current time.

        `key` is OPTIONAL and, unlike a summary key, ungated: it does not need
        to match an existing summary slot, and there is no UnknownSummaryKey-style
        refusal for coining a new one. Chunks are high-volume and low-commitment
        by design — filing one under a key that doesn't have a summary yet is
        valid, raw material waiting to possibly be promoted later. It only
        widens chunk_id()'s hash so the same text under two keys doesn't
        collapse into one entry; a pile of chunks under a key is never current
        state on its own, only a summary is.
        """
        category, corrected_from = _normalize_category(category)
        project_key = project or "general"
        key = _normalize_key(key)
        if not project:
            tier = None
        elif tier and tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        elif not tier:
            raise ValueError("tier is required when project is set")

        cleaned = [d.strip() for d in documents if d and d.strip()]
        if not cleaned:
            raise ValueError("save_chunks called with no non-empty documents")

        metadata = {
            "project": project_key,
            "category": category,
            "type": "chunk",
            "source": source,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if tier:
            metadata["tier"] = tier
        if chat_title:
            metadata["chat_title"] = chat_title
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from
        if key:
            metadata["key"] = key

        # Deduplicate before the upsert rather than relying on Chroma to
        # tolerate a repeated id inside one batch.
        by_id: dict[str, str] = {}
        for doc in cleaned:
            by_id.setdefault(self.chunk_id(doc, project_key, category, key), doc)

        ids = list(by_id)
        docs = [by_id[i] for i in ids]
        self.collection.upsert(
            ids=ids, documents=docs, metadatas=[dict(metadata) for _ in ids]
        )
        return {
            "ids": ids,
            "count": len(ids),
            "duplicates_collapsed": len(cleaned) - len(ids),
            "oversized": [
                {"id": i, "chars": len(d)} for i, d in zip(ids, docs) if len(d) > MAX_DOC_CHARS
            ],
            "corrected_from": corrected_from,
            **metadata,
        }

    def get_summary(
        self,
        project: Optional[str],
        category: str,
        with_embedding: bool = False,
        key: Optional[str] = None,
    ):
        """
        Current content of one summary slot, or None if empty.
        Returns (document, metadata, embedding) so callers that are about to
        overwrite can archive the old version without re-embedding it.
        """
        category, _ = _normalize_category(category)
        key = _normalize_key(key)
        include = ["documents", "metadatas"] + (["embeddings"] if with_embedding else [])
        got = self.collection.get(ids=[self.summary_id(project, category, key)], include=include)
        if not got["ids"]:
            return None
        emb = got.get("embeddings")
        return (
            got["documents"][0],
            got["metadatas"][0],
            emb[0] if with_embedding and emb is not None and len(emb) else None,
        )

    def get_brief(
        self, project: Optional[str] = None, category: Optional[str] = None
    ) -> list[dict]:
        """
        Every living summary for a project, returned WHOLE — no similarity
        ranking, no top_k, no truncation.

        This exists because search() is the wrong instrument for "what is the
        current state of X". Ranked search returns the first ~1000 chars of the
        five best-matching documents, which for long entries means most of the
        content is unreachable no matter how the query is phrased. A brief is a
        deterministic id lookup, so it can hand back the complete text.

        Passing `category` narrows it to one category. That matters once a
        project has real depth: context-mcp's full brief is ~17k characters, but
        a session working on auth wants architecture + config + decisions about
        Cognito, not the retrieval roadmap. Loading the whole thing to read a
        fifth of it is the same waste get_brief was built to avoid at the
        search layer, one level up.
        """
        project_key = project or "general"
        clauses: list[dict] = [{"project": project_key}, {"type": "summary"}]
        corrected_from = None
        if category is not None:
            category, corrected_from = _normalize_category(category)
            clauses.append({"category": category})
        got = self.collection.get(
            where={"$and": clauses},
            include=["documents", "metadatas"],
        )
        entries = [
            {
                "category": meta.get("category"),
                "key": meta.get("key"),
                "content": doc,
                "tier": meta.get("tier"),
                "source": meta.get("source"),
                "timestamp": meta.get("timestamp"),
            }
            for doc, meta in zip(got["documents"], got["metadatas"])
        ]
        # Stable, readable order rather than whatever the store returns. Slots
        # of one category group together, the unkeyed one leading.
        entries.sort(key=lambda e: (e["category"] or "", e["key"] or ""))
        if corrected_from:
            for e in entries:
                e["category_corrected_from"] = corrected_from
        return entries

    def slot_history(
        self,
        project: Optional[str],
        category: str,
        key: Optional[str] = None,
    ) -> dict:
        """
        Every archived version of ONE slot, newest first.

        History was only ever reachable through search(), which means guessing
        at words: a caller who knows exactly which slot it cares about still had
        to phrase a query and hope ranking cooperated. But every archive already
        records superseded_from (the slot it came from) and superseded_at (when
        it stopped being current), so the version chain existed all along with
        nothing able to read it.

        A metadata lookup, so no embedding call and no ranking — ask for a slot's
        past and get exactly that slot's past, complete and in order.

        Deliberately slot-scoped. "How did this entry evolve" and "what happened
        around the 3rd" are different questions on the same metadata; the second
        is easy to add later against superseded_at, and is not built until
        something actually needs it.
        """
        category, corrected_from = _normalize_category(category)
        key = _normalize_key(key)
        sid = self.summary_id(project, category, key)
        got = self.collection.get(
            where={"superseded_from": sid}, include=["documents", "metadatas"]
        )
        # An oversized version was archived as several pieces (see
        # _split_for_archive), and one archival event is ONE version of the slot,
        # not three. Group by superseded_at — every piece of a single _archive
        # call carries the same stamp by construction — and stitch the pieces
        # back in order, so history reads as a version chain rather than as a
        # pile of fragments the caller has to reassemble.
        by_event: dict[str, list] = {}
        for vid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            by_event.setdefault(meta.get("superseded_at") or "", []).append((vid, doc, meta))

        versions = []
        for stamp, members in by_event.items():
            members.sort(key=lambda m: m[2].get("split_index") or 0)
            content = "\n\n".join(doc for _, doc, _ in members)
            head_meta = members[0][2]
            version = {
                "archived_id": members[0][0],
                "content": content,
                "superseded_at": stamp or None,
                "written_at": head_meta.get("timestamp"),
                "chars": len(content),
                "reason": head_meta.get("archived_reason"),
            }
            if len(members) > 1:
                # Named only when it happened, so the ordinary case stays terse.
                # A missing piece shows up here as a length mismatch rather than
                # as text that quietly isn't there.
                version["reassembled_from"] = [vid for vid, _, _ in members]
                version["pieces"] = len(members)
                expected = head_meta.get("split_count")
                if expected and expected != len(members):
                    version["incomplete"] = (
                        f"expected {expected} pieces, found {len(members)} — "
                        "this version is missing part of its text"
                    )
            versions.append(version)
        # Newest first: the most recent previous value is nearly always the one
        # being asked for, and reverse-chronological reads as a changelog.
        versions.sort(key=lambda v: v["superseded_at"] or "", reverse=True)

        current = self.get_summary(project, category, key=key)
        return {
            "slot": category + (f"/{key}" if key else ""),
            "project": project or "general",
            "summary_id": sid,
            "current": (
                {"content": current[0], "chars": len(current[0]),
                 "timestamp": current[1].get("timestamp")}
                if current else None
            ),
            "archived_slot": current is None and bool(versions),
            "versions": versions,
            "version_count": len(versions),
            "category_corrected_from": corrected_from,
        }

    def _archive(self, sid: str, previous_doc: str, previous_meta: dict, previous_emb) -> list[str]:
        """
        Stash a copy of a summary about to be changed, as superseded chunk(s).

        Deterministic ids from the content being archived, so re-running the same
        replacement doesn't pile up duplicate copies of one old value.

        Returns a LIST because an oversized document is split into
        retrieval-sized pieces (see _split_for_archive). Nearly always one id;
        callers that want a single handle should take the first.

        COST, and the split changes it. Unsplit, this reuses the vector Chroma
        already holds — the text is unchanged, so its embedding is still exactly
        right and archiving costs no Voyage call at all. That is what makes a
        forced periodic archive (PATCH_ARCHIVE_EVERY) cheap enough to run on a
        schedule. Split, the pieces are new texts and the old vector no longer
        describes any of them, so they must be embedded — but all of them go in
        ONE upsert and therefore one Voyage call, however many pieces there are.
        So the common case stays free and the rare case costs one call, not one
        per piece.
        """
        pieces = _split_for_archive(previous_doc)
        archive_meta = {
            **previous_meta,
            "type": "chunk",
            "source": SUPERSEDED_SOURCE,
            "superseded_from": sid,
            # One timestamp for the whole archival event, so the pieces of a
            # single version share a group key that slot_history can gather on.
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        # A copy is a historical record, not a live slot — the live document's
        # patch counter says nothing about it and would only be confusing here.
        archive_meta.pop("unarchived_patches", None)

        if len(pieces) == 1:
            digest = hashlib.sha1(f"{sid}-{previous_doc}".encode()).hexdigest()[:16]
            archive_kwargs = {
                "ids": [f"superseded-{digest}"],
                "documents": [previous_doc],
                "metadatas": [archive_meta],
            }
            if previous_emb is not None:
                archive_kwargs["embeddings"] = [previous_emb]
            self.collection.upsert(**archive_kwargs)
            return archive_kwargs["ids"]

        ids, metas = [], []
        for i, piece in enumerate(pieces):
            # split_index is in the hash as well as the metadata: two pieces of
            # one document could in principle carry identical text, and without
            # it they would collapse onto one id and lose a piece silently.
            digest = hashlib.sha1(f"{sid}-{i}-{piece}".encode()).hexdigest()[:16]
            ids.append(f"superseded-{digest}")
            metas.append({**archive_meta, "split_index": i, "split_count": len(pieces)})
        # No embeddings passed: each piece is a new text and needs its own vector.
        # One upsert, so Chroma issues one embed call for the whole batch.
        self.collection.upsert(ids=ids, documents=pieces, metadatas=metas)
        return ids

    def archive_slot(
        self,
        project: Optional[str],
        category: str,
        key: Optional[str] = None,
        reason: str = "",
    ) -> dict:
        """
        Retire a whole summary slot: archive its text, then remove the live
        document so it stops loading with every brief.

        For work that is FINISHED rather than wrong. A completed phase is not
        current state — it is a record of what was done — but a summary slot has
        no way to say "done", so it keeps costing tokens on every get_brief for a
        question nobody is asking any more. change_update can't express this
        either: it archives the old text but insists on leaving a new live value
        behind, and there is no value that means "this slot no longer applies".

        The archived copy is a superseded chunk, not a retired one, and that
        distinction is the point. Retired means wrong. Superseded means it was
        true when written and still is as history — which is exactly what a
        finished phase is, and why the copy stays visible in ordinary search
        rather than being hidden behind an opt-in flag.

        Reuses _archive(), so the existing vector is carried over and this costs
        no Voyage call at all — unless the slot is long enough to be split into
        pieces, which costs exactly one. The delete only removes the live
        pointer; the text survives in the archive.
        """
        sid = self.summary_id(project, category, key)
        current = self.get_summary(project, category, with_embedding=True, key=key)
        if current is None:
            raise PatchSlotMissing(sid)
        doc, meta, emb = current

        archive_meta = dict(meta)
        if reason and reason.strip():
            archive_meta["archived_reason"] = reason.strip()
        archived_ids = self._archive(sid, doc, archive_meta, emb)
        # Only now drop the live document. If _archive raised, the slot is still
        # intact — losing the copy and the original in one call is the failure
        # this ordering exists to prevent.
        self.collection.delete(ids=[sid])
        return {
            "archived": True,
            "id": sid,
            "archived_id": archived_ids[0],
            "archived_ids": archived_ids,
            "split_into": len(archived_ids) if len(archived_ids) > 1 else None,
            "chars_freed": len(doc),
            "reason": reason.strip() or None,
            "project": project or "general",
            "category": category,
            "key": key,
        }

    def retire_chunk(
        self,
        chunk_id: str,
        reason: str,
        superseded_by: Optional[str] = None,
    ) -> dict:
        """
        Mark a chunk as WRONG so it stops competing with current entries.

        Chunks are append-only: a fact recorded in good faith and later disproved
        stays semantically searchable forever, ranking on the same queries as the
        entry that corrected it. Nothing in the document says it was contradicted,
        so a reader has no way to tell which of two plausible answers still holds.

        This is metadata-only — collection.update() leaves the document and its
        existing vector untouched, so retiring costs no Voyage call. The text is
        kept rather than deleted: the record that something was believed, and when
        it stopped being true, is worth more than the space it occupies.

        Refuses to retire a summary. A summary's current value is replaced through
        change_update or patch_context, which archives the old text properly;
        flagging a live slot as wrong would leave the project with no value at all
        for that category and nothing pointing at what replaced it.
        """
        if not reason or not reason.strip():
            raise ValueError("reason is required — a retired chunk with no stated reason is worse than none")

        got = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not got["ids"]:
            raise ChunkNotFound(chunk_id)

        meta = dict(got["metadatas"][0])
        document = got["documents"][0]
        if meta.get("type") != "chunk":
            raise ValueError(
                f"'{chunk_id}' is a {meta.get('type')}, not a chunk. Replace a summary's "
                "value with change_update or patch_context instead — that archives the "
                "old text and leaves a current value in place."
            )
        if meta.get("source") == RETIRED_SOURCE:
            return {
                "retired": False,
                "already_retired": True,
                "id": chunk_id,
                # archived_reason is current; retired_reason is a fallback for
                # chunks retired before the reason fields collapsed into one.
                "reason": meta.get("archived_reason") or meta.get("retired_reason"),
                "retired_at": meta.get("retired_at"),
            }

        previous_source = meta.get("source")
        meta.update(
            {
                "source": RETIRED_SOURCE,
                "archived_reason": reason.strip(),
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "retired_from_source": previous_source,
            }
        )
        if superseded_by:
            meta["superseded_by"] = superseded_by
        # update(), not upsert(): the document and its embedding stay exactly as
        # they are, so this is a metadata write and nothing is re-embedded.
        self.collection.update(ids=[chunk_id], metadatas=[meta])
        return {
            "retired": True,
            "id": chunk_id,
            "reason": reason.strip(),
            "superseded_by": superseded_by,
            "previous_source": previous_source,
            "project": meta.get("project"),
            "category": meta.get("category"),
            "excerpt": document[:200],
        }

    def patch_summary(
        self,
        old_str: str,
        new_str: str,
        category: str,
        project: Optional[str] = None,
        source: str = "live",
        key: Optional[str] = None,
    ):
        """
        Change one passage of a summary in place, leaving the rest untouched.

        The point is cost, and it is a write-side cost: update_summary requires
        the caller to reproduce the ENTIRE new document, so changing one line of
        a thousand-token summary means generating a thousand tokens — the
        expensive kind, and the slow kind, to move a handful of characters.
        A patch sends only what changed.

        It is also the safer of the two. update_summary needs a shrink guard
        because a replacement can silently destroy everything it forgot to
        mention; a patch structurally cannot, since it only ever rewrites text
        it has matched. That is why no equivalent guard exists here, and why
        archiving is conditional (see PATCH_ARCHIVE_RATIO) rather than automatic.

        Three refusals, all of which hand back the stored document so the caller
        can retry against reality rather than against its own stale copy:
        no match, more than one match, and a patch that would change nothing.
        `tier` is inherited from the slot rather than accepted as an argument —
        a patch is editing something that already exists, so re-declaring its
        tier could only introduce a disagreement.
        """
        category, corrected_from = _normalize_category(category)
        key = _normalize_key(key)
        sid = self.summary_id(project, category, key)

        current = self.get_summary(project, category, with_embedding=True, key=key)
        if current is None:
            raise PatchSlotMissing(sid)
        previous_doc, previous_meta, previous_emb = current

        if old_str == new_str:
            raise PatchNoOp(sid)

        occurrences = previous_doc.count(old_str)
        if occurrences == 0:
            raise PatchNoMatch(sid, previous_doc, old_str)
        if occurrences > 1:
            raise PatchAmbiguous(sid, previous_doc, old_str, occurrences)

        patched = previous_doc.replace(old_str, new_str, 1)

        # Two independent triggers: this patch is large enough to approach a
        # rewrite, or enough small ones have accumulated since the last copy
        # that the document may have drifted substantially anyway.
        #
        # max(old, new), not len(old_str) alone. The ratio originally measured
        # only how much text was DISPLACED, which is blind to the shape of edit
        # that causes the most drift: an append, where a short old_str is swapped
        # for a much longer new_str. Measured live, three slots grew 45-145% in a
        # single session and archived nothing, because each edit displaced under
        # 20% while adding multiples of it — and get_history then reported they
        # had "only ever been patched in small pieces", which was untrue. A patch
        # that injects far more than it removes rewrites the document just as
        # thoroughly as one that replaces a long passage.
        touched = (
            max(len(old_str), len(new_str)) / len(previous_doc) if previous_doc else 1.0
        )
        unarchived = int(previous_meta.get("unarchived_patches") or 0)
        archive_now = touched >= PATCH_ARCHIVE_RATIO or (unarchived + 1) >= PATCH_ARCHIVE_EVERY

        archived_id = None
        archived_ids = None
        if archive_now:
            archived_ids = self._archive(sid, previous_doc, previous_meta, previous_emb)
            archived_id = archived_ids[0]

        # Rebuilt field by field rather than spread from previous_meta: tier and
        # chat_title are properties of the slot and must survive, but source and
        # timestamp describe THIS write, and a stale category_corrected_from
        # would outlive the correction it recorded.
        metadata = {
            "project": previous_meta.get("project", project or "general"),
            "category": category,
            "type": "summary",
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unarchived_patches": 0 if archive_now else unarchived + 1,
        }
        if key:
            metadata["key"] = key
        if previous_meta.get("tier"):
            metadata["tier"] = previous_meta["tier"]
        if previous_meta.get("chat_title"):
            metadata["chat_title"] = previous_meta["chat_title"]
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from

        self.collection.upsert(ids=[sid], documents=[patched], metadatas=[metadata])
        return {
            "id": sid,
            # Deliberately NOT the patched document. Returning it would hand
            # back every token the patch existed to avoid sending. The three
            # lengths verify the write completely: if the delta is exactly
            # len(new_str) - len(old_str), nothing outside the match moved.
            "chars_before": len(previous_doc),
            "chars_after": len(patched),
            "delta": len(patched) - len(previous_doc),
            "archived_id": archived_id,
            "archived_split_into": len(archived_ids) if archived_ids and len(archived_ids) > 1 else None,
            "corrected_from": corrected_from,
            **metadata,
        }

    def index(self, project: Optional[str] = None, detail: str = "slots") -> dict:
        """
        A map of what this store holds, without any of the contents.

        The cheap way to get oriented. get_brief answers "what does this project
        say" by returning every summary whole — measured at ~4,280 tokens for one
        project here. This answers the prior question, "what is there to ask
        about", in roughly fifty: one line per slot, sizes and dates only.

        Costs no Voyage call at all. Both reads are metadata lookups rather than
        similarity queries, so nothing is embedded — and the summary side stays
        small by construction, since there is only ever one living summary per
        project+category no matter how much history accumulates underneath it.

        Superseded archives and retired chunks are both excluded. Neither is part
        of what the store currently knows: counting archives would make every
        edited slot look like it had grown, and counting retired chunks would
        report material the store has been told is wrong.

        detail="projects" drops the per-slot map and returns only the shape of
        each project — the table of contents a session should OPEN with. At
        eleven projects the full form is ~1,600 tokens and this is ~170, which is
        the difference between a lookup a session will make by default and one it
        has to be talked into. Everything below it is unchanged, so a caller that
        wants slots asks for slots.
        """
        if detail not in ("slots", "projects"):
            raise ValueError(f"detail must be 'slots' or 'projects', got '{detail}'")
        summary_where: dict = {"type": "summary"}
        chunk_clauses: list[dict] = [{"type": "chunk"},
                                     {"source": {"$nin": list(INDEX_EXCLUDED_SOURCES)}}]
        if project:
            summary_where = {"$and": [summary_where, {"project": project}]}
            chunk_clauses.append({"project": project})

        summaries = self.collection.get(
            where=summary_where, include=["documents", "metadatas"]
        )
        # Documents are fetched only to measure them; the text never leaves this
        # method. Cheap because there is one summary per slot — a handful of
        # documents, not the whole store.
        chunks = self.collection.get(
            where={"$and": chunk_clauses}, include=["metadatas"]
        )
        # Archived copies, counted per slot they came from. The index is what a
        # session reads to decide where to look next, so it has to advertise
        # what the next level down actually holds — a slot rewritten five times
        # is worth knowing about, and without this the version chain is
        # invisible unless someone already suspects it exists.
        archived_where: dict = {"source": SUPERSEDED_SOURCE}
        if project:
            archived_where = {"$and": [archived_where, {"project": project}]}
        archived = self.collection.get(where=archived_where, include=["metadatas"])
        versions_per_slot: dict[str, int] = {}
        for meta in archived["metadatas"]:
            origin = meta.get("superseded_from")
            if origin:
                versions_per_slot[origin] = versions_per_slot.get(origin, 0) + 1

        projects: dict[str, dict] = {}

        def slot(name: str) -> dict:
            return projects.setdefault(
                name, {"summaries": {}, "brief_chars": 0, "history_chunks": 0}
            )

        # Slots archived outright have no live summary to hang a version count
        # on, so they would vanish from the map entirely. Counting them tells a
        # session that retired work exists here without listing what it was.
        live_ids = {
            self.summary_id(m.get("project"), m.get("category"), m.get("key"))
            for m in summaries["metadatas"]
        }
        # Distinct ORIGINS, not archived copies. A slot rewritten five times and
        # then archived is one archived slot, not five, and counting versions
        # here would inflate every long-lived project.
        archived_origins: dict[str, set] = {}
        for meta in archived["metadatas"]:
            origin = meta.get("superseded_from")
            if origin and origin not in live_ids:
                proj = meta.get("project") or "general"
                archived_origins.setdefault(proj, set()).add(origin)
        archived_only = {p: len(v) for p, v in archived_origins.items()}

        for doc, meta in zip(summaries["documents"], summaries["metadatas"]):
            entry = slot(meta.get("project") or "general")
            # Keyed slots are labelled "category/key" and unkeyed ones just
            # "category", so one flat mapping shows the whole shape of a project
            # — including which categories have been split and which haven't.
            label = meta.get("category")
            if meta.get("key"):
                label = f"{label}/{meta['key']}"
            slot_info = {
                "chars": len(doc),
                "updated": (meta.get("timestamp") or "")[:10],
            }
            prior = versions_per_slot.get(
                self.summary_id(meta.get("project"), meta.get("category"), meta.get("key"))
            )
            if prior:
                # Only present when there IS history, so the common case stays
                # as terse as it was and a version count means something.
                slot_info["prior_versions"] = prior
            entry["summaries"][label] = slot_info
            entry["brief_chars"] += len(doc)
            if meta.get("tier"):
                entry["tier"] = meta["tier"]

        for meta in chunks["metadatas"]:
            entry = slot(meta.get("project") or "general")
            entry["history_chunks"] += 1
            if meta.get("tier"):
                entry.setdefault("tier", meta["tier"])

        # A project whose slots were ALL archived has no live summary and may
        # have no live chunks either, so nothing above would have created an
        # entry for it — it would disappear from the map entirely, taking the
        # only pointer to its history with it. Give it an entry so the archive
        # is still findable.
        for name in archived_only:
            slot(name)

        for name, entry in projects.items():
            entry["summaries"] = dict(sorted(entry["summaries"].items()))
            if archived_only.get(name):
                entry["archived_slots"] = archived_only[name]

        if detail == "projects":
            # Counts and shape only. Enough to decide WHICH project to open and
            # roughly what it will cost, which is the whole job of a contents
            # page — anything more and it stops being cheap enough to open with.
            toc = {}
            for name, entry in sorted(projects.items()):
                row = {
                    "slots": len(entry["summaries"]),
                    "brief_chars": entry["brief_chars"],
                    "history_chunks": entry["history_chunks"],
                }
                if entry.get("tier"):
                    row["tier"] = entry["tier"]
                if entry.get("archived_slots"):
                    row["archived_slots"] = entry["archived_slots"]
                toc[name] = row
            return {
                "projects": toc,
                "total_summaries": len(summaries["ids"]),
                "total_history_chunks": len(chunks["ids"]),
                "next": "get_context(project) for one project's current state, "
                        "or get_index(project=...) for its individual slots.",
            }

        return {
            "projects": dict(sorted(projects.items())),
            "total_summaries": len(summaries["ids"]),
            "total_history_chunks": len(chunks["ids"]),
        }

    def update_summary(
        self,
        document: str,
        category: str,
        project: Optional[str] = None,
        tier: Optional[str] = None,
        source: str = "live",
        chat_title: Optional[str] = None,
        allow_shrink: bool = False,
        key: Optional[str] = None,
        create_key: bool = False,
    ):
        """
        Replace one summary slot in place, archiving whatever was there.

        This is the "whiteboard" write, as opposed to save(type="chunk") which
        is the "diary". Three safeguards, because replacing is destructive in a
        way appending never is:
          1. The previous content is returned, so a truncating write is visible
             immediately rather than discovered later.
          2. A replacement far shorter than what it replaces is refused
             (SHRINK_GUARD_RATIO).
          3. The old version is archived as a chunk before being overwritten, so
             nothing is ever actually lost — worst case the summary reads wrong
             and the previous version is one query away.
        """
        category, corrected_from = _normalize_category(category)
        key = _normalize_key(key)
        if key is None:
            # No keyless "main slot" — see decisions/no-keyless-slots. While a
            # category could hold both, get_context(project, category) was
            # ambiguous between "the category's own summary" and "everything
            # filed under it". Requiring a key makes that unrepresentable rather
            # than resolved by convention. Reads stay permissive: a keyless
            # lookup returns not-found, which is honest, since none can exist.
            raise MissingSummaryKey(
                project or "general", category, self.summary_keys(project, category)
            )
        project_key = project or "general"
        if not project:
            tier = None  # tier is meaningless without a project
        elif tier and tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS}, got '{tier}'")
        elif not tier:
            raise ValueError("tier is required when project is set")

        sid = self.summary_id(project, category, key)
        current = self.get_summary(project, category, with_embedding=True, key=key)

        # Guard only the creation of a NEW key, and only where something already
        # exists to duplicate. Writing to a key that is already there, or opening
        # the first slot in an empty category, has nothing to fragment.
        if key and current is None and not create_key:
            existing = self.summary_keys_ranked(project, category, document)
            if existing:
                raise UnknownSummaryKey(project_key, category, key, existing)

        archived_id = None
        archived_ids = None
        previous_doc = None
        if current is not None:
            previous_doc, previous_meta, previous_emb = current

            if not allow_shrink and len(document) < len(previous_doc) * SHRINK_GUARD_RATIO:
                raise SummaryShrinkRefused(sid, previous_doc, document)

            archived_ids = self._archive(sid, previous_doc, previous_meta, previous_emb)
            archived_id = archived_ids[0]

        metadata = {
            "project": project_key,
            "category": category,
            "type": "summary",
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # A full replacement always archives, so nothing is owed at this
            # point — see PATCH_ARCHIVE_EVERY.
            "unarchived_patches": 0,
        }
        if key:
            metadata["key"] = key
        if tier:
            metadata["tier"] = tier
        if chat_title:
            metadata["chat_title"] = chat_title
        if corrected_from:
            metadata["category_corrected_from"] = corrected_from

        self.collection.upsert(ids=[sid], documents=[document], metadatas=[metadata])
        return {
            "id": sid,
            "previous": previous_doc,
            "archived_id": archived_id,
            "archived_split_into": len(archived_ids) if archived_ids and len(archived_ids) > 1 else None,
            "corrected_from": corrected_from,
            **metadata,
        }

    def search(
        self,
        query: str,
        project: Optional[str] = None,  # None = search across all (incl. general)
        category: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        include_retired: bool = False,
    ):
        """
        Ranked similarity search over summaries and history.

        ONE exclusion, not two. Superseded copies used to be hidden alongside
        retired ones behind a separate include_superseded flag; since 2026-08-16
        they are ordinary history and rank like any other chunk — see
        SEARCH_HIDDEN_SOURCES for why hiding them was both inconsistent and, in
        the archive_slot case, actively harmful. Only material the store has been
        TOLD is wrong stays out by default.
        """
        corrected_from = None
        if category is not None:
            category, corrected_from = _normalize_category(category)
        top_k = min(top_k, MAX_TOP_K)

        where_clauses = []
        if project:
            where_clauses.append({"project": project})
        if category:
            where_clauses.append({"category": category})
        if not include_retired:
            where_clauses.append({"source": {"$ne": RETIRED_SOURCE}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        results = self.collection.query(
            query_texts=[query], n_results=top_k, where=where
        )

        # Truncate long documents so one call can't dump unbounded tokens
        # into the conversation. Full content should live in a proper
        # summary anyway if it's meant to be retrieved whole.
        docs = results.get("documents", [[]])[0]
        truncated_docs = []
        for d in docs:
            if len(d) > MAX_DOC_CHARS:
                truncated_docs.append(d[:MAX_DOC_CHARS] + " …[truncated]")
            else:
                truncated_docs.append(d)
        results["documents"] = [truncated_docs]
        # Visible signal, not silent — caller/tool-response can surface this
        # ("interpreted category 'desicions' as 'decisions'") instead of the
        # correction happening invisibly.
        results["category_corrected_from"] = corrected_from
        return results


class _OfflineHashEmbedding:
    """
    Deterministic, dependency-free embedding function used ONLY because this
    sandbox can't reach the CDN that hosts ChromaDB's default onnx model, and
    has no Voyage API key configured. It's a crude bag-of-words hash into a
    fixed-size vector — good enough to prove filtering + retrieval logic
    works, NOT something to actually use. On your machine, use
    VoyageRestEmbedding with a real API key instead.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "offline-hash-stub"

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            # numpy, not a list, for the same reason as VoyageRestEmbedding —
            # otherwise this stub works locally and breaks the moment it's
            # pointed at a Cloud collection.
            vectors.append(vec)
        return vectors

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)


if __name__ == "__main__":
    import json
    import shutil

    TEST_PATH = "./chroma_data_smoketest"
    shutil.rmtree(TEST_PATH, ignore_errors=True)

    # Smoke test: write tagged entries, prove upsert semantics (writing the
    # same summary twice should REPLACE it, not duplicate it), prove
    # category validation auto-corrects typos visibly, prove project-filtered
    # retrieval still respects boundaries, prove result truncation kicks in,
    # and prove the two embedding-function safety paths actually work
    # (mismatch guard raises; local-fallback reopen doesn't crash).
    # use_cloud=False on every construction below: without it, a developer with
    # Chroma Cloud credentials in .env would run this suite against the real
    # shared store and write fixture data into it.
    store = ContextStore(
        persist_path=TEST_PATH,
        embedding_function=_OfflineHashEmbedding(),
        embedding_function_name="offline-hash-stub",
        use_cloud=False,
    )

    store.save(
        document="The ticketing SaaS uses Postgres with a Redis cache. Backend is FastAPI, frontend is React.",
        category="architecture",
        type="summary",
        project="ticketing-saas",
        tier="client",
        source="backfill",
    )
    # Save the SAME summary again with different content — should overwrite,
    # not create a second "living" summary.
    result = store.save(
        document="UPDATED: The ticketing SaaS uses Postgres with a Redis cache and now also runs a Kafka queue for async jobs.",
        category="architecture",
        type="summary",
        project="ticketing-saas",
        tier="client",
        source="live",
    )
    print("=== Upsert test: same id both times? ===")
    print(result["id"])

    count = store.collection.count()
    print(f"Total docs in collection after two summary writes to the same slot: {count}")
    assert count == 1, f"expected 1 doc, got {count}"

    print("\n=== Typo correction test: 'desicions' should auto-correct to 'decisions', visibly ===")
    result = store.save(
        document="Decided to use Kafka for async jobs.",
        category="desicions",  # typo, on purpose
        type="summary",
        project="ticketing-saas",
        tier="client",
    )
    print(f"Saved under corrected category: '{result['category']}', corrected_from='{result['corrected_from']}'")
    assert result["category"] == "decisions"
    assert result["corrected_from"] == "desicions"

    r = store.search("kafka decision", project="ticketing-saas", category="desicions")
    print(f"Searching with the SAME typo still finds it: {r['documents']}")
    print(f"Search also surfaces the correction: category_corrected_from={r['category_corrected_from']}")
    assert r["category_corrected_from"] == "desicions"

    print("\n=== Nonsense category test: should still fail, no reasonable match ===")
    try:
        store.save(
            document="test",
            category="xyzabc123",
            type="summary",
            project="ticketing-saas",
            tier="client",
        )
        print("FAIL: nonsense category was accepted")
    except ValueError as e:
        print(f"OK, rejected: {e}")

    print("\n=== Retrieval reflects the UPDATED summary, not the original ===")
    r = store.search("what does the architecture look like", project="ticketing-saas")
    print(r["documents"])

    print("\n=== Truncation test ===")
    long_text = "This is a very long chunk. " * 60
    assert len(long_text) > MAX_DOC_CHARS, "the fixture must actually exceed the cap"
    store.save(document=long_text, category="note", type="chunk", project=None, source="live")
    r = store.search("very long chunk", category="note")
    doc = r["documents"][0][0]
    # Derived from the constant, not hardcoded: MAX_DOC_CHARS moved 800 -> 1000
    # with chunk-splitting, and a magic number here silently outlives the change.
    ceiling = MAX_DOC_CHARS + len(" …[truncated]")
    print(f"Returned doc length: {len(doc)} (cap {MAX_DOC_CHARS} + marker = {ceiling})")
    print(doc[-40:])
    assert len(doc) <= ceiling

    print("\n=== tier-stripped-for-general test ===")
    result = store.save(
        document="I prefer terse responses.",
        category="preference",
        type="summary",
        project=None,
        tier="personal",  # should be silently stripped, not stored
    )
    assert "tier" not in result, f"tier leaked into a general entry: {result}"
    print("OK, tier not present on general entry")

    print("\n=== Embedding-function mismatch guard test ===")
    try:
        ContextStore(
            persist_path=TEST_PATH,
            embedding_function=_OfflineHashEmbedding(),
            embedding_function_name="some-other-function",
            use_cloud=False,
        )
        print("FAIL: reopening with a different embedding_function_name did not raise")
    except RuntimeError as e:
        print(f"OK, rejected: {e}")

    print("\n=== Local-fallback reopen test (allow_local_fallback=True, no explicit function) ===")
    FALLBACK_PATH = "./chroma_data_smoketest_fallback"
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)
    fallback_store = ContextStore(
        persist_path=FALLBACK_PATH, allow_local_fallback=True, use_cloud=False
    )
    fallback_store.save(
        document="Testing the local fallback embedding path.",
        category="note",
        type="chunk",
        project=None,
        source="live",
    )
    # Reopen the SAME path with allow_local_fallback again — exercises the
    # get_collection(embedding_function=None) branch, not just creation.
    fallback_reopen = ContextStore(
        persist_path=FALLBACK_PATH, allow_local_fallback=True, use_cloud=False
    )
    r = fallback_reopen.search("local fallback embedding")
    print(f"Local-fallback reopen query returned: {r['documents']}")
    assert len(r["documents"][0]) >= 1, "local-fallback reopen returned no results"
    print("OK, local-fallback reopen works")

    print("\n=== Content-addressed chunk ids: writing the same chunk twice must not duplicate ===")
    before = store.collection.count()
    first = store.save_chunks(["A fact that gets written twice."], category="note")
    second = store.save_chunks(["A fact that gets written twice."], category="note")
    added = store.collection.count() - before
    print(f"ids equal across both writes: {first['ids'] == second['ids']}, collection grew by {added}")
    assert first["ids"] == second["ids"], "chunk id was not stable for identical content"
    assert added == 1, f"expected 1 new entry, got {added}"

    print("\n=== Chunk keys: same text under different keys does NOT collapse ===")
    assert store.chunk_id("x", "p", "cat") == f"chunk-{hashlib.sha1('p-cat-x'.encode()).hexdigest()[:16]}", \
        "an unkeyed chunk id must reproduce the pre-key formula byte for byte"
    unkeyed = store.save_chunks(["Rotation needs a republish."], category="note", project="keychunks", tier="personal")
    keyed_a = store.save_chunks(["Rotation needs a republish."], category="note", project="keychunks", tier="personal", key="alpha")
    keyed_b = store.save_chunks(["Rotation needs a republish."], category="note", project="keychunks", tier="personal", key="bravo")
    same_key_again = store.save_chunks(["Rotation needs a republish."], category="note", project="keychunks", tier="personal", key="alpha")
    ids = {unkeyed["ids"][0], keyed_a["ids"][0], keyed_b["ids"][0]}
    print(f"  unkeyed={unkeyed['ids'][0]} alpha={keyed_a['ids'][0]} bravo={keyed_b['ids'][0]}")
    assert len(ids) == 3, "identical text under three different keys (incl. no key) must get three distinct ids"
    assert keyed_a["ids"] == same_key_again["ids"], "same text under the same key must still be stable"
    stored_meta = store.collection.get(ids=[keyed_a["ids"][0]], include=["metadatas"])["metadatas"][0]
    assert stored_meta["key"] == "alpha", "chunk metadata must carry its key"
    print("  OK, keyed chunks get distinct, stable ids and carry key in metadata.")

    print("\n=== save_chunks: batch write, in-batch dedupe, oversized warning ===")
    batch = store.save_chunks(
        ["Fact one.", "Fact two.", "Fact one.", "x" * (MAX_DOC_CHARS + 100)],
        category="note",
    )
    print(f"count={batch['count']} duplicates_collapsed={batch['duplicates_collapsed']} "
          f"oversized={len(batch['oversized'])}")
    assert batch["count"] == 3, f"expected 3 distinct chunks, got {batch['count']}"
    assert batch["duplicates_collapsed"] == 1
    assert len(batch["oversized"]) == 1, "oversized chunk was not flagged at write time"

    print("\n=== patch_summary: the happy path ===")
    PATCH_DOC = (
        "Deploy status: PendingConfirmation. "
        "Slot A: alpha. Slot B: alpha. Slot C: alpha. Slot D: alpha. Slot E: alpha. "
        "The service runs on Lambda in eu-west-1, and the alarm topic is in eu-west-1. "
        + "Filler text keeping this document long enough that small patches stay "
          "comfortably under the archive ratio. " * 4
    )
    store.save(
        document=PATCH_DOC, category="config", type="summary",
        project="patch-test", tier="personal", source="live",
    )
    res = store.patch_summary(
        old_str="PendingConfirmation", new_str="Confirmed",
        category="config", project="patch-test",
    )
    expected_delta = len("Confirmed") - len("PendingConfirmation")
    print(f"chars {res['chars_before']} -> {res['chars_after']} (delta {res['delta']}, "
          f"expected {expected_delta}), archived={res['archived_id']}")
    assert res["delta"] == expected_delta, "something outside the match moved"
    assert res["archived_id"] is None, "a small patch should not archive a full copy"
    assert res["unarchived_patches"] == 1

    doc, meta, _ = store.get_summary("patch-test", "config")
    assert "Confirmed." in doc and "PendingConfirmation" not in doc
    assert "Slot A: alpha." in doc and doc.endswith(PATCH_DOC[-40:]), "unrelated text was disturbed"
    assert meta["tier"] == "personal", "tier was not inherited from the slot"
    print("Patch applied, rest of the document untouched, tier inherited.")

    print("\n=== patch_summary: every refusal hands back the stored text ===")
    for label, kwargs, exc in [
        ("no match", {"old_str": "text that is not there", "new_str": "x"}, PatchNoMatch),
        ("not unique", {"old_str": "eu-west-1", "new_str": "eu-west-2"}, PatchAmbiguous),
        ("no change", {"old_str": "Slot A", "new_str": "Slot A"}, PatchNoOp),
    ]:
        try:
            store.patch_summary(category="config", project="patch-test", **kwargs)
            print(f"FAIL: '{label}' was not refused")
            raise SystemExit(1)
        except exc as refusal:
            carries = refusal.current is not None
            extra = f", occurrences={refusal.count}" if isinstance(refusal, PatchAmbiguous) else ""
            print(f"OK, refused ({label}): carries current text={carries}{extra}")
            # A no-op is refused before the document is read, so it has nothing
            # to hand back; the other two must carry it or the caller can't retry.
            assert carries == (exc is not PatchNoOp)

    try:
        store.patch_summary(
            old_str="anything", new_str="x", category="config", project="no-such-project"
        )
        print("FAIL: patching an empty slot was not refused")
        raise SystemExit(1)
    except PatchSlotMissing as refusal:
        print(f"OK, refused (no summary yet): {refusal}")

    print("\n=== patch_summary: accumulated small patches force a checkpoint ===")
    archived_at = None
    for n, slot in enumerate(["A", "B", "C", "D"], start=2):
        res = store.patch_summary(
            old_str=f"Slot {slot}: alpha", new_str=f"Slot {slot}: beta",
            category="config", project="patch-test",
        )
        print(f"  patch {n}: unarchived_patches={res['unarchived_patches']} archived={res['archived_id']}")
        if res["archived_id"]:
            archived_at = n
    assert archived_at == PATCH_ARCHIVE_EVERY, (
        f"expected a forced archive on patch {PATCH_ARCHIVE_EVERY}, got {archived_at}"
    )
    assert store.get_summary("patch-test", "config")[1]["unarchived_patches"] == 0

    print("\n=== patch_summary: a large patch archives immediately ===")
    store.save(
        document="alpha beta gamma delta", category="note", type="summary",
        project="patch-test", tier="personal", source="live",
    )
    res = store.patch_summary(
        old_str="alpha", new_str="omega", category="note", project="patch-test"
    )
    print(f"touched {len('alpha')}/{len('alpha beta gamma delta')} chars, archived={res['archived_id']}")
    assert res["archived_id"] is not None, "a patch over the ratio should archive"

    print("\n=== patch_summary: an APPEND-shaped patch archives too ===")
    # The regression this guards: the ratio used to measure only len(old_str),
    # so swapping a short marker for a large block displaced almost nothing
    # while growing the document by half — and archived nothing. Live slots grew
    # 45-145% in one session with zero checkpoints before this was fixed.
    base = "HEADER. " + "Body text that makes this document a realistic length. " * 20
    store.save(document=base, category="tasks", type="summary",
               project="append-test", tier="personal", source="live")
    injected = "MASSIVE NEW SECTION. " + "Freshly appended detail. " * 30
    res = store.patch_summary(
        old_str="HEADER.", new_str=f"HEADER.\n\n{injected}",
        category="tasks", project="append-test",
    )
    displaced = len("HEADER.") / len(base)
    print(f"  displaced {displaced:.1%} of the document but grew it "
          f"{res['chars_before']} -> {res['chars_after']} ({res['delta']:+d}); "
          f"archived={res['archived_id'] is not None}")
    assert displaced < PATCH_ARCHIVE_RATIO, "fixture must displace less than the ratio"
    assert res["archived_id"] is not None, \
        "an append that injects more than the ratio must archive — growth is drift too"
    assert store.slot_history("append-test", "tasks")["versions"][0]["content"] == base, \
        "the checkpoint must hold the pre-append text"
    print("  checkpoint holds the pre-append version.")

    print("\n=== Archived copies are ordinary, visible history ===")
    r = store.search("alpha beta gamma", project="patch-test", top_k=10)
    assert any("alpha beta gamma delta" == d for d in r["documents"][0]), \
        "a superseded copy must rank in DEFAULT search — it is history, not error"
    assert any("omega" in d for d in r["documents"][0]), "the live value must still be there too"
    print("OK, superseded copies rank alongside the current value, no flag needed.")

    print("\n=== index(): the map, without the contents ===")
    idx = store.index()
    print(json.dumps(idx, indent=2, sort_keys=True))

    # Every project that has anything must appear, and sizes must be real.
    assert "patch-test" in idx["projects"] and "ticketing-saas" in idx["projects"]
    pt = idx["projects"]["patch-test"]
    assert pt["tier"] == "personal"
    stored_config = store.get_summary("patch-test", "config")[0]
    assert pt["summaries"]["config"]["chars"] == len(stored_config), "index size disagrees with the store"
    assert pt["brief_chars"] == sum(v["chars"] for v in pt["summaries"].values())

    # The whole point: the map costs a fraction of the content it describes.
    brief_chars = sum(len(e["content"]) for e in store.get_brief("patch-test"))
    map_chars = len(json.dumps(idx["projects"]["patch-test"]))
    print(f"\npatch-test: index {map_chars} chars vs get_brief {brief_chars} chars")
    assert map_chars < brief_chars

    print("\n=== index(): superseded archives are excluded, live chunks counted ===")
    live_general = store.collection.get(
        where={"$and": [{"project": "general"}, {"type": "chunk"},
                        {"source": {"$ne": SUPERSEDED_SOURCE}}]},
        include=[],
    )
    assert idx["projects"]["general"]["history_chunks"] == len(live_general["ids"])
    # patch-test accumulated superseded copies during the patch tests above;
    # none of them may show up as history.
    archived = store.collection.get(
        where={"$and": [{"project": "patch-test"}, {"source": SUPERSEDED_SOURCE}]}, include=[]
    )
    assert len(archived["ids"]) >= 2, "expected archives from the patch tests"
    assert idx["projects"]["patch-test"]["history_chunks"] == 0, "archives leaked into the index"
    print(f"OK, {len(archived['ids'])} archived copies present and none counted as history.")

    # Search visibility and index arithmetic are deliberately DIFFERENT sets.
    # Superseded copies became ordinary search results on 2026-08-16, but the
    # index still leaves them out of history_chunks because it reports them
    # separately as prior_versions — counting both would double-count.
    assert SUPERSEDED_SOURCE in INDEX_EXCLUDED_SOURCES
    assert SUPERSEDED_SOURCE not in SEARCH_HIDDEN_SOURCES
    visible = store.search("alpha beta gamma", project="patch-test", top_k=10)["ids"][0]
    assert any(i in visible for i in archived["ids"]), \
        "the same copies the index excludes must still be searchable"
    print("  and the same copies ARE searchable — the two exclusions are separate on purpose.")

    print("\n=== index(project=...) scopes, and an unknown project is empty not an error ===")
    scoped = store.index(project="patch-test")
    assert list(scoped["projects"]) == ["patch-test"]
    assert store.index(project="no-such-project")["projects"] == {}
    print("OK, scoping and empty results behave.")

    print("\n=== Summary ids: prefix-based, keyed extends unkeyed ===")
    assert store.summary_id("p", "config") == "summary-p-config"
    assert store.summary_id("p", "config", "lambda") == "summary-p-config-lambda"
    print("OK, summary- prefix on both forms, keyed just appends -{key}.")

    print("\n=== Key slugification ===")
    for raw, want in [("Lambda Settings", "lambda-settings"), ("lambda_settings", "lambda-settings"),
                      ("  COGNITO  ", "cognito"), ("api gateway!!", "api-gateway")]:
        got_key = _normalize_key(raw)
        print(f"  {raw!r:20} -> {got_key!r}")
        assert got_key == want
    for bad in ["", "!!!", "summary", "x" * (MAX_KEY_CHARS + 1)]:
        try:
            _normalize_key(bad)
            print(f"FAIL: {bad!r} was accepted as a key")
            raise SystemExit(1)
        except ValueError:
            pass
    print("OK, empty/punctuation-only/reserved/over-long keys rejected.")

    print("\n=== A keyless write is refused outright ===")
    try:
        store.update_summary(
            document="Runs on Lambda: python3.13, arm64, 512MB, 30s timeout, SnapStart on.",
            category="config", project="keytest", tier="personal",
        )
        print("FAIL: a keyless summary was created")
        raise SystemExit(1)
    except MissingSummaryKey as refusal:
        assert "key is required" in str(refusal)
        print(f"OK, refused: {refusal}")

    print("\n=== The FIRST key in an empty category needs no create_key ===")
    store.update_summary(
        document="Runs on Lambda: python3.13, arm64, 512MB, 30s timeout, SnapStart on.",
        category="config", project="keytest", tier="personal", key="lambda",
    )
    print("OK, keytest/config/lambda opened the category.")

    print("\n=== A SECOND key in a populated category is refused without create_key ===")
    try:
        store.update_summary(
            document="Cognito pool with two app clients, PKCE for the public one.",
            category="config", project="keytest", tier="personal", key="cognito",
        )
        print("FAIL: a new key was created without create_key")
        raise SystemExit(1)
    except UnknownSummaryKey as refusal:
        print(f"OK, refused. existing={refusal.existing}")
        assert "lambda" in refusal.existing, "the existing key must be offered as a candidate"

    print("\n=== create_key=True lets it through, and the slots are independent ===")
    store.update_summary(
        document="Cognito pool with two app clients, PKCE for the public one.",
        category="config", project="keytest", tier="personal", key="cognito", create_key=True,
    )
    store.update_summary(
        document="Alarm at 5 rejections per 300s, SNS topic context-mcp-alerts.",
        category="config", project="keytest", tier="personal", key="alerting", create_key=True,
    )
    assert store.summary_keys("keytest", "config") == ["alerting", "cognito", "lambda"]
    assert None not in store.summary_keys("keytest", "config"), "no keyless slot may exist"
    first = store.get_summary("keytest", "config", key="lambda")[0]
    assert "Lambda" in first and "Cognito" not in first, "writing a key touched another key's slot"
    print(f"OK, keys = {store.summary_keys('keytest', 'config')}, slots independent.")

    print("\n=== Writing to an EXISTING key needs no create_key, and patch is key-aware ===")
    store.update_summary(
        document="Cognito pool eu-west-1_x, two app clients, PKCE for the public one.",
        category="config", project="keytest", tier="personal", key="cognito",
    )
    res = store.patch_summary(
        old_str="two app clients", new_str="three app clients",
        category="config", project="keytest", key="cognito",
    )
    assert res["id"] == "summary-keytest-config-cognito"
    assert "three app clients" in store.get_summary("keytest", "config", key="cognito")[0]
    assert "Lambda" in store.get_summary("keytest", "config", key="lambda")[0], \
        "patching one key must leave its siblings alone"
    assert store.get_summary("keytest", "config") is None, \
        "a keyless lookup must be not-found, never a fallback to some other slot"
    print(f"OK, patched {res['id']} only.")

    print("\n=== Refusal ranks existing keys by CONTENT, closest first ===")
    ranked = store.summary_keys_ranked(
        "keytest", "config", "Raised the CloudWatch alarm threshold and re-pointed the SNS topic."
    )
    print(f"  incoming about alerting -> ranked {ranked}")
    assert ranked[0] == "alerting", f"expected 'alerting' first, got {ranked}"

    print("\n=== Brief and index expose keys ===")
    brief = store.get_brief("keytest")
    labels = [(e["category"], e["key"]) for e in brief]
    print(f"  brief slots: {labels}")
    assert ("config", "lambda") in labels and ("config", "cognito") in labels
    assert all(k for _, k in labels), "every brief entry must carry a key"
    keyed_index = store.index(project="keytest")["projects"]["keytest"]["summaries"]
    print(f"  index labels: {sorted(keyed_index)}")
    assert "config/lambda" in keyed_index and "config/cognito" in keyed_index
    assert "config" not in keyed_index, "a bare category label means a keyless slot exists"

    print("\n=== index(detail='projects'): the table of contents a session opens with ===")
    import json as _json
    full = store.index()
    toc = store.index(detail="projects")
    assert set(full["projects"]) == set(toc["projects"]), "the TOC must list every project"
    for name, row in toc["projects"].items():
        assert "summaries" not in row, "the TOC must not carry per-slot detail"
        assert row["slots"] == len(full["projects"][name]["summaries"])
        assert row["brief_chars"] == full["projects"][name]["brief_chars"]
    full_size, toc_size = len(_json.dumps(full)), len(_json.dumps(toc))
    assert toc_size < full_size
    print(f"  {len(toc['projects'])} projects: {full_size} chars -> {toc_size} chars "
          f"({100*toc_size/full_size:.0f}% of the full map)")
    assert "next" in toc, "the TOC must say what to call next — that is the point"
    try:
        store.index(detail="nonsense")
        raise AssertionError("an unknown detail should be refused")
    except ValueError:
        print("  refuses an unknown detail value.")

    print("\n=== get_brief(category=...): load one category, not the project ===")
    store.update_summary("Python 3.13, FastAPI, Chroma.", category="tech_stack",
                         project="tiers", tier="personal", key="overview")
    store.update_summary("OAuth only, no shared credential.", category="architecture",
                         project="tiers", tier="personal", key="auth")
    store.update_summary("Alarm at 5 per 300s.", category="config",
                         project="tiers", tier="personal", key="alerting")
    whole = store.get_brief("tiers")
    one = store.get_brief("tiers", category="architecture")
    assert len(whole) == 3 and len(one) == 1 and one[0]["category"] == "architecture"
    whole_chars = sum(len(e["content"]) for e in whole)
    one_chars = sum(len(e["content"]) for e in one)
    assert one_chars < whole_chars
    print(f"  whole project {whole_chars} chars -> one category {one_chars} chars")
    assert store.get_brief("tiers", category="architecure")[0]["category"] == "architecture", \
        "a near-miss category should still resolve"
    assert store.get_brief("tiers", category="tasks") == [], "a category with no slots is empty, not an error"
    print("  typo corrected, empty category returns cleanly.")

    print("\n=== slot_history: a known slot's past, without guessing at words ===")
    store.update_summary("Rotation is easy.", category="config", project="hist",
                         tier="personal", key="rotation", create_key=False)
    for text in ["Rotation needs a republish.", "Rotation needs a republish and an alias move."]:
        store.update_summary(text, category="config", project="hist", tier="personal", key="rotation")
    h = store.slot_history("hist", "config", "rotation")
    assert h["version_count"] == 2, f"expected 2 archived versions, got {h['version_count']}"
    assert h["current"]["content"].endswith("alias move.")
    stamps = [v["superseded_at"] for v in h["versions"]]
    assert stamps == sorted(stamps, reverse=True), "versions must come back newest first"
    assert h["versions"][-1]["content"] == "Rotation is easy.", "oldest version should be the original"
    assert h["archived_slot"] is False
    print(f"  {h['version_count']} versions, newest first, current value alongside.")

    idx_h = store.index(project="hist")["projects"]["hist"]["summaries"]["config/rotation"]
    assert idx_h["prior_versions"] == 2, "the index must advertise that history exists"
    print(f"  index advertises prior_versions={idx_h['prior_versions']} on the slot.")

    empty = store.slot_history("hist", "tech_stack")
    assert empty["version_count"] == 0 and empty["current"] is None
    print("  a slot with no history returns cleanly rather than erroring.")

    # archived_slots counts distinct slots, not archived copies. This slot has
    # two prior versions; archiving it must add ONE, not three.
    store.archive_slot(project="hist", category="config", key="rotation", reason="done")
    hist_idx = store.index(project="hist")["projects"]["hist"]
    assert hist_idx.get("archived_slots") == 1, \
        f"expected 1 archived slot, got {hist_idx.get('archived_slots')} — counting versions, not slots"
    assert "config/rotation" not in hist_idx["summaries"]
    assert hist_idx["summaries"] == {}, "that was the project's only slot"
    print("  archived_slots counts slots (1), not the 2 versions underneath it,")
    print("  and a project with nothing live still appears, so its history stays findable.")
    # The archived text itself must survive the slot disappearing.
    assert store.slot_history("hist", "config", "rotation")["version_count"] == 3, \
        "archiving adds the live value as a third version"

    # A superseded chunk (from _archive) must carry the key of the summary it
    # came from — it's redundant with superseded_from but keeps both queryable
    # on the same field instead of needing per-origin parsing logic.
    rotation_versions = store.collection.get(
        where={"superseded_from": store.summary_id("hist", "config", "rotation")},
        include=["metadatas"],
    )["metadatas"]
    assert all(m.get("key") == "rotation" for m in rotation_versions), \
        "every superseded copy of a keyed slot must carry that key in its metadata"
    print("  superseded copies of a keyed slot carry that key too.")

    print("\n=== archive_slot: finished work leaves the brief but stays retrievable ===")
    store.update_summary("PHASE X - do the thing. DONE, shipped as abc1234.",
                         category="tasks", project="archtest", tier="personal", key="phase-x")
    store.update_summary("PHASE Y - still outstanding.",
                         category="tasks", project="archtest", tier="personal", key="phase-y",
                         create_key=True)
    before = store.index(project="archtest")["projects"]["archtest"]
    assert "tasks/phase-x" in before["summaries"] and "tasks/phase-y" in before["summaries"]
    arch = store.archive_slot(project="archtest", category="tasks", key="phase-x",
                              reason="Completed 2026-08-05; detail is history, not current state.")
    after = store.index(project="archtest")["projects"]["archtest"]
    assert arch["archived"] and arch["chars_freed"] > 0
    assert "tasks/phase-x" not in after["summaries"], "archived slot must leave the index"
    assert "tasks/phase-y" in after["summaries"], "archiving one slot must not touch another"
    assert after["brief_chars"] < before["brief_chars"]
    assert not any(e["key"] == "phase-x" for e in store.get_brief("archtest")), \
        "archived slot must leave get_brief — that is the whole point"
    print(f"  gone from brief and index, freed {arch['chars_freed']} chars; sibling untouched.")

    # Archived as SUPERSEDED, not RETIRED: it was true when written. And this is
    # the case that drove the visibility collapse — archive_slot leaves NOTHING
    # behind, so if the archived copy were hidden, the only content that ever
    # existed on this topic would be undiscoverable by default. A search finding
    # nothing is a worse failure than a search finding an old answer.
    back = store.search(query="phase X do the thing", project="archtest", top_k=10)
    assert arch["archived_id"] in back["ids"][0], \
        "an archived-outright slot must be findable in DEFAULT search, or it is lost"
    amet = store.collection.get(ids=[arch["archived_id"]], include=["metadatas"])["metadatas"][0]
    assert amet["source"] == SUPERSEDED_SOURCE, "finished is superseded, not retired"
    assert amet["archived_reason"].startswith("Completed")
    assert amet["superseded_from"] == store.summary_id("archtest", "tasks", "phase-x"), \
        "provenance must survive — it stopped gating visibility, it did not go away"
    print("  findable by default, provenance intact, flagged superseded not retired.")

    try:
        store.archive_slot(project="archtest", category="tasks", key="never-existed")
        raise AssertionError("archiving a missing slot should have been refused")
    except PatchSlotMissing:
        print("  refuses a slot that does not exist.")

    print("\n=== retire_chunk: a disproved fact stops competing with the one that fixed it ===")
    saved = store.save_chunks(
        [
            "Rotation is easy: update the secret and the next cold start picks it up.",
            "The OIDC subject claim was the CI failure.",
        ],
        category="config", project="retiretest", tier="personal",
    )
    wrong, other = saved["ids"][0], saved["ids"][1]

    def _visible(**kw):
        return set(store.search(query="rotating a secret", project="retiretest", top_k=10, **kw)["ids"][0])

    assert wrong in _visible(), "chunk should be searchable before retirement"
    res = store.retire_chunk(
        wrong,
        reason="SnapStart means a cold start does NOT pick up a rotated secret.",
        superseded_by="config/rotation",
    )
    assert res["retired"] and res["previous_source"] == "live"
    assert wrong not in _visible(), "retired chunk must drop out of default search"
    assert other in _visible(), "retiring one chunk must not affect its neighbours"
    print("  hidden from search, neighbour untouched.")

    # Retirement is now the ONLY exclusion, which makes it the load-bearing one:
    # superseded history flows through default search freely, so nothing but
    # include_retired can bring back material known to be wrong.
    assert wrong in _visible(include_retired=True), "include_retired should surface it for auditing"
    print("  only include_retired brings it back.")

    meta = store.collection.get(ids=[wrong], include=["metadatas"])["metadatas"][0]
    assert meta["source"] == RETIRED_SOURCE and meta["superseded_by"] == "config/rotation"
    assert meta["archived_reason"].startswith("SnapStart")
    assert "retired_reason" not in meta, "retire_chunk must write the unified archived_reason field, not the old one"
    assert store.retire_chunk(wrong, reason="again")["already_retired"] is True
    assert store.index(project="retiretest")["projects"]["retiretest"]["history_chunks"] == 1, \
        "retired chunks must not be counted as live history"
    print("  metadata recorded, idempotent, excluded from the index count.")

    store.update_summary("A live summary.", category="config", project="retiretest",
                         tier="personal", key="alerting")
    for bad_id, label in [(store.summary_id("retiretest", "config", "alerting"), "a summary"),
                          ("chunk-nope", "a missing id")]:
        try:
            store.retire_chunk(bad_id, reason="x")
            raise AssertionError(f"retiring {label} should have been refused")
        except (ValueError, ChunkNotFound):
            pass
    try:
        store.retire_chunk(other, reason="   ")
        raise AssertionError("a blank reason should have been refused")
    except ValueError:
        pass
    print("  refuses summaries, unknown ids, and blank reasons.")

    print("\n=== _split_for_archive: the buffer, and what does NOT get split ===")
    assert _split_for_archive("short") == ["short"]
    # Inside the +10% buffer: over the display cap but not worth fragmenting.
    inside = "x" * (MAX_DOC_CHARS + 50)
    assert len(_split_for_archive(inside)) == 1, \
        "a document inside the buffer must survive whole — that is the whole point of the buffer"
    # A single oversized paragraph has no boundary to cut on, and inventing one
    # would mean slicing mid-sentence.
    assert len(_split_for_archive("y" * (SPLIT_THRESHOLD + 500))) == 1, \
        "an unparagraphed document must come back whole, not hard-cut"
    print(f"  cap {MAX_DOC_CHARS}, split trigger {SPLIT_THRESHOLD}; buffer and "
          "unparagraphed text both left intact.")

    para = "P{} " + "filler words to give this paragraph real length. " * 8
    long_doc = "\n\n".join(para.format(i) for i in range(6))
    pieces = _split_for_archive(long_doc)
    print(f"  {len(long_doc)}ch in 6 paragraphs -> {len(pieces)} pieces, "
          f"sizes {[len(p) for p in pieces]}")
    assert len(pieces) > 1, "a genuinely oversized multi-paragraph document must split"
    assert all(len(p) <= SPLIT_THRESHOLD for p in pieces), "no piece may exceed the threshold"
    assert "\n\n".join(pieces) == long_doc, "splitting must be lossless — every byte survives"
    assert all(p.strip() for p in pieces), "no empty pieces"
    # Greedy packing, not one-paragraph-per-piece: 6 paragraphs must not become 6.
    assert len(pieces) < 6, "paragraphs must be packed greedily, not split one per piece"

    print("\n=== Archiving an oversized slot splits it, and history stitches it back ===")
    store.update_summary(long_doc, category="note", project="splittest",
                         tier="personal", key="big")
    store.update_summary(long_doc + "\n\nP6 a new trailing paragraph.",
                         category="note", project="splittest", tier="personal", key="big")
    h = store.slot_history("splittest", "note", "big")
    assert h["version_count"] == 1, \
        f"one archival event is ONE version however many pieces it became, got {h['version_count']}"
    v = h["versions"][0]
    print(f"  archived as {v['pieces']} pieces, reassembled to {v['chars']}ch "
          f"(original {len(long_doc)}ch)")
    assert v["pieces"] > 1 and len(v["reassembled_from"]) == v["pieces"]
    assert v["content"] == long_doc, "the stitched version must equal the original byte for byte"
    assert "incomplete" not in v, "no piece should be missing"

    # Each piece is independently searchable — the actual point of splitting.
    # A blended embedding over the whole document is why a buried paragraph
    # would not rank; per-piece vectors are what fix that.
    hits = store.search("P4 filler words", project="splittest", top_k=10)
    assert any(m.get("split_count") for m in hits["metadatas"][0]), \
        "split pieces must be searchable in their own right"
    piece_meta = next(m for m in hits["metadatas"][0] if m.get("split_count"))
    assert piece_meta["split_count"] == v["pieces"]
    assert piece_meta.get("superseded_from") == store.summary_id("splittest", "note", "big"), \
        "a piece must still name the slot it came from"
    print(f"  pieces searchable individually, each carrying split_count="
          f"{piece_meta['split_count']} and its superseded_from.")

    # An unsplit archive must NOT gain split metadata — the common case stays
    # exactly as it was, including reusing its existing vector for free.
    store.update_summary("Short value one, long enough to clear the shrink guard.",
                         category="note", project="splittest", tier="personal",
                         key="small", create_key=True)
    store.update_summary("Short value two, long enough to clear the shrink guard.",
                         category="note", project="splittest", tier="personal", key="small")
    small = store.slot_history("splittest", "note", "small")["versions"][0]
    assert "pieces" not in small and "reassembled_from" not in small, \
        "an unsplit archive must carry no split bookkeeping at all"
    print("  an unsplit archive stays exactly as before, no split metadata added.")

    shutil.rmtree(TEST_PATH, ignore_errors=True)
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)
    print("\nAll smoke tests passed.")
