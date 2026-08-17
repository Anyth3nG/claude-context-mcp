"""
Runs the contract in shared/conformance.py against Chroma, then Chroma's own
engine tests.

WHY THIS IS A SEPARATE MODULE and not shared/store.py's __main__: running
`python -m shared.store` executes store.py as `__main__`, and conformance.py
imports `shared.store` — so Python holds TWO copies of the module and two sets
of exception classes. `except PatchNoMatch` then fails to catch the
`PatchNoMatch` that was raised, because they are different objects. Keeping the
entry point out of store.py means exactly one copy exists.

Add a sibling (smoke_dynamodb.py) to run the same contract against a second
backend; that is the shape the migration needs.
"""
from __future__ import annotations

import shutil

from shared import conformance
from shared.store import ContextStore, _OfflineHashEmbedding

TEST_PATH = "./chroma_data_smoketest"
FALLBACK_PATH = "./chroma_data_smoketest_fallback"


def make_chroma_store() -> ContextStore:
    """
    A fresh, EMPTY Chroma store.

    use_cloud=False is load-bearing rather than tidy: without it a developer with
    Chroma Cloud credentials in .env runs this whole suite against the real
    shared store and writes fixture data into it.
    """
    return ContextStore(
        persist_path=TEST_PATH,
        embedding_function=_OfflineHashEmbedding(),
        embedding_function_name="offline-hash-stub",
        use_cloud=False,
    )


def main() -> None:
    shutil.rmtree(TEST_PATH, ignore_errors=True)
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)

    conformance.run(make_chroma_store)

    # Chroma-specific from here down. These assert things about the ENGINE, not
    # about this store's semantics, which is why they are not in the portable
    # contract. A DynamoDB backend will need its own versions — a hand-rolled
    # mismatch guard — rather than these.
    store = make_chroma_store()
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

    shutil.rmtree(TEST_PATH, ignore_errors=True)
    shutil.rmtree(FALLBACK_PATH, ignore_errors=True)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
