"""
The behavioural contract of ContextStore, as an executable specification.

This is the smoke suite that used to sit at the bottom of shared/store.py. It
moved here for one reason: it is the only complete statement of what this store
DOES, and while it constructed Chroma itself and reached into `store.collection`,
it could only ever run against Chroma. A storage rewrite verified by a suite that
only runs on the thing being replaced is not verified.

So run() takes a FACTORY, and asserts behaviour only through the public API plus
the three introspection methods (count / record / records). Point it at a second
backend and a passing run is the port's proof.

Deliberately NOT here: tests of Chroma's own machinery — the embedding-function
mismatch guard and the local-fallback reopen. Those assert things about the
ENGINE rather than about this store's semantics, so they stay beside the backend
they describe. DynamoDB will need its own equivalents, not these — see
tasks/dynamodb-migration on hand-rolling the mismatch guard.
"""
from __future__ import annotations

import hashlib
import json
import json as _json

from shared.store import (
    INDEX_EXCLUDED_SOURCES,
    MAX_DOC_CHARS,
    MAX_KEY_CHARS,
    PATCH_ARCHIVE_EVERY,
    PATCH_ARCHIVE_RATIO,
    RETIRED_SOURCE,
    SEARCH_HIDDEN_SOURCES,
    SPLIT_THRESHOLD,
    SUPERSEDED_SOURCE,
    ChunkNotFound,
    MissingSummaryKey,
    PatchAmbiguous,
    PatchNoMatch,
    PatchNoOp,
    PatchSlotMissing,
    SummaryShrinkRefused,
    UnknownSummaryKey,
    _normalize_key,
    _split_for_archive,
)


def run(make_store) -> None:
    """
    Exercise the whole contract against whatever make_store() returns.

    The factory must hand back an EMPTY store: several assertions count records
    store-wide, so fixtures left over from a previous run would fail them for
    the wrong reason.
    """
    store = make_store()

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

    count = store.count()
    print(f"Total records in the store after two summary writes to the same slot: {count}")
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

    print("\n=== Content-addressed chunk ids: writing the same chunk twice must not duplicate ===")
    before = store.count()
    first = store.save_chunks(["A fact that gets written twice."], category="note")
    second = store.save_chunks(["A fact that gets written twice."], category="note")
    added = store.count() - before
    print(f"ids equal across both writes: {first['ids'] == second['ids']}, store grew by {added}")
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
    stored_meta = store.record(keyed_a["ids"][0])
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
    live_general = store.count(project="general", type="chunk",
                               exclude_sources=(SUPERSEDED_SOURCE,))
    assert idx["projects"]["general"]["history_chunks"] == live_general
    # patch-test accumulated superseded copies during the patch tests above;
    # none of them may show up as history.
    archived = store.records(project="patch-test", source=SUPERSEDED_SOURCE)
    assert len(archived) >= 2, "expected archives from the patch tests"
    assert idx["projects"]["patch-test"]["history_chunks"] == 0, "archives leaked into the index"
    print(f"OK, {len(archived)} archived copies present and none counted as history.")

    # Search visibility and index arithmetic are deliberately DIFFERENT sets.
    # Superseded copies became ordinary search results on 2026-08-16, but the
    # index still leaves them out of history_chunks because it reports them
    # separately as prior_versions — counting both would double-count.
    assert SUPERSEDED_SOURCE in INDEX_EXCLUDED_SOURCES
    assert SUPERSEDED_SOURCE not in SEARCH_HIDDEN_SOURCES
    visible = store.search("alpha beta gamma", project="patch-test", top_k=10)["ids"][0]
    assert any(a["id"] in visible for a in archived), \
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
    rotation_versions = store.records(
        superseded_from=store.summary_id("hist", "config", "rotation"))
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
    amet = store.record(arch["archived_id"])
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

    meta = store.record(wrong)
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

    print("\n=== Archiving the SAME content twice does not duplicate it ===")
    # Content-addressed ids mean a repeated archive is the SAME record, and a
    # backend must collapse it. Missed by this suite until a DynamoDB sort key
    # containing a timestamp split one id across two items — which happens in
    # practice because patching a long slot leaves its tail split pieces
    # byte-identical between archival events.
    A = "Value A, long enough to clear the shrink guard comfortably here."
    B = "Value B, also long enough to clear the shrink guard comfortably."
    store.update_summary(A, category="note", project="dupetest", tier="personal", key="cycle")
    store.update_summary(B, category="note", project="dupetest", tier="personal", key="cycle")
    store.update_summary(A, category="note", project="dupetest", tier="personal", key="cycle")
    store.update_summary(B, category="note", project="dupetest", tier="personal", key="cycle")
    dupe_rows = store.records(project="dupetest")
    dupe_ids = [r["id"] for r in dupe_rows]
    print(f"  4 writes cycling A/B/A/B -> {len(dupe_ids)} records, "
          f"{len(set(dupe_ids))} distinct ids")
    assert len(dupe_ids) == len(set(dupe_ids)), (
        f"a record id appeared more than once: "
        f"{[i for i in dupe_ids if dupe_ids.count(i) > 1]}")
    hist = store.slot_history("dupetest", "note", "cycle")
    assert all(v["content"] in (A, B) for v in hist["versions"])
    print(f"  {hist['version_count']} archived versions, every id unique.")

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

    print("\n=== A slot read returns the caller's own last write ===")
    # Read-your-writes, asserted because a backend can violate it SILENTLY.
    # patch_summary reads the slot, matches old_str against what came back,
    # derives the archive copy and the patch counter from it, and writes the
    # result — so a read serving a stale version produces a patch computed
    # against superseded text whose every built-in check still passes, and the
    # intervening edit disappears with the call reporting success. The returned
    # chars_before/after/delta cannot catch it either: they are measured from
    # the same stale read, so they agree with each other.
    #
    # Chained on purpose. Each iteration's old_str is the PREVIOUS iteration's
    # new_str, so a stale read cannot quietly match — it raises PatchNoMatch and
    # names the backend. Looped because a propagation race that loses only
    # sometimes is still a broken contract, and one round would usually win it.
    store.update_summary(
        "Consistency probe, revision r0, long enough to clear the shrink guard.",
        category="config", project="ryw", tier="personal", key="probe", create_key=True,
    )
    for n in range(8):
        store.patch_summary(old_str=f"revision r{n},", new_str=f"revision r{n + 1},",
                            category="config", project="ryw", key="probe")
        doc, meta, _ = store.get_summary("ryw", "config", key="probe")
        assert f"revision r{n + 1}," in doc, (
            f"read-after-write violated on round {n}: the slot came back as {doc!r}, "
            f"without the revision just written"
        )
    assert "revision r8," in store.get_summary("ryw", "config", key="probe")[0]
    print("  8 chained patches, each read back immediately, no stale version served.")

    print("\nContract satisfied.")
