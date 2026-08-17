#!/usr/bin/env python3
"""
Throwaway spike: does DynamoDB vector search do what shared/store.py needs?

Run BEFORE porting anything. decisions/dynamodb-over-chroma accepted the move on
an access-pattern argument, but the evaluation's own strongest objection was that
AWS's filtering docs were incomplete — so the filtering semantics are exactly
what has to be established by experiment rather than by reading.

Five questions, each answered PASS/FAIL against a throwaway table that is
deleted at the end:

  Q1  Does an exact GetItem read-your-write immediately? (get_summary depends on it)
  Q2  Does vector search rank a known-nearest item first?
  Q3  Do filter AND similarity combine in ONE call, as Chroma's where+query_texts do?
  Q4  How stale is the vector index after a write, in seconds?
  Q5  Can you search across ALL partitions, or does a HASH element force scoping?

    python3 spike_dynamodb_vectors.py            # create, test, delete
    python3 spike_dynamodb_vectors.py --keep     # leave the table for poking
"""
from __future__ import annotations

import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "eu-west-1"
TABLE = "context-mcp-vector-spike"
INDEX = "vec"
DIMS = 1024          # matches voyage-3.5, so this is not a toy-dimension result
KEEP = "--keep" in sys.argv

ddb = boto3.client("dynamodb", region_name=REGION)
results: list[tuple[str, bool, str]] = []


def record(q: str, ok: bool, detail: str) -> None:
    results.append((q, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {q} — {detail}")


def vec(slot: int, width: int = 64) -> list[float]:
    """A 1024-dim vector distinguishable by which block is hot."""
    v = [0.0] * DIMS
    for i in range(slot * width, slot * width + width):
        v[i] = 1.0
    return v


def av_vec(v: list[float]) -> dict:
    return {"L": [{"N": str(x)} for x in v]}


def search_vec(v: list[float]) -> list[dict]:
    return [{"N": str(x)} for x in v]


def create() -> None:
    print(f"creating {TABLE} (PK/SK + 1024-dim COSINE vector index)…")
    ddb.create_table(
        TableName=TABLE,
        BillingMode="PAY_PER_REQUEST",  # vector indexes are on-demand only
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            # FINDING: every attribute named in SearchSchema must also be
            # declared here, exactly like a GSI key. CreateTable fails with
            # "One element in SearchSchema is not defined in attribute
            # definitions" otherwise. So the set of filterable attributes is
            # fixed at table-definition time, not per query.
            {"AttributeName": "project", "AttributeType": "S"},
            {"AttributeName": "category", "AttributeType": "S"},
            {"AttributeName": "source", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        VectorIndexes=[
            {
                "IndexName": INDEX,
                "VectorAttribute": {"AttributeName": "embedding"},
                # The crux. Chroma filters on arbitrary metadata; here the
                # filterable attributes must be declared. No HASH element is
                # declared on purpose — Q5 tests whether search then spans all
                # partitions, which search_context(project=None) requires.
                "SearchSchema": [
                    {"AttributeName": "project", "SearchSchemaElementType": "INLINE_FILTER"},
                    {"AttributeName": "category", "SearchSchemaElementType": "INLINE_FILTER"},
                    {"AttributeName": "source", "SearchSchemaElementType": "INLINE_FILTER"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "Dimensions": DIMS,
                "DistanceFunction": "COSINE",
            }
        ],
    )
    for _ in range(60):
        d = ddb.describe_table(TableName=TABLE)["Table"]
        if d["TableStatus"] == "ACTIVE":
            vidx = d.get("VectorIndexes", [])
            states = {i["IndexName"]: i.get("IndexStatus", "?") for i in vidx}
            if all(s == "ACTIVE" for s in states.values()) and states:
                print(f"  table ACTIVE, vector index {states}")
                return
        time.sleep(5)
    raise SystemExit("table did not become ACTIVE in 5 minutes")


def put(pk: str, sk: str, project: str, category: str, source: str, slot: int, doc: str) -> None:
    ddb.put_item(TableName=TABLE, Item={
        "pk": {"S": pk}, "sk": {"S": sk},
        "project": {"S": project}, "category": {"S": category}, "source": {"S": source},
        "document": {"S": doc},
        "embedding": av_vec(vec(slot)),
    })


def seed() -> None:
    print("\nseeding 4 items across 2 projects…")
    put("context-mcp", "config#lambda",   "context-mcp", "config",    "live", 0, "Lambda runs arm64.")
    put("context-mcp", "decisions#voyage","context-mcp", "decisions", "live", 1, "Voyage for embeddings.")
    put("context-mcp", "config#old",      "context-mcp", "config", "superseded", 2, "An archived value.")
    put("ticketing",   "config#domains",  "ticketing",   "config",    "live", 3, "Domains for ticketing.")


def q1_read_your_write() -> None:
    print("\nQ1 exact lookup, read-your-write")
    put("probe", "probe#1", "probe", "note", "live", 5, "written just now")
    got = ddb.get_item(TableName=TABLE,
                       Key={"pk": {"S": "probe"}, "sk": {"S": "probe#1"}},
                       ConsistentRead=True)
    ok = "Item" in got and got["Item"]["document"]["S"] == "written just now"
    record("Q1 GetItem is immediately consistent", ok,
           "strongly-consistent read returned the write" if ok else "write not visible")


def q2_ranking() -> None:
    print("\nQ2 nearest-first ranking")
    r = ddb.search_vectors(TableName=TABLE, IndexName=INDEX,
                           SearchVector=search_vec(vec(1)), TopK=3)
    hits = [(h["Item"]["sk"]["S"], round(h["Score"], 4)) for h in r["SearchResults"]]
    ok = bool(hits) and hits[0][0] == "decisions#voyage"
    record("Q2 exact-match vector ranks first", ok, f"top{len(hits)}={hits}")


def q3_filter_plus_similarity() -> None:
    print("\nQ3 filter + similarity in one call")
    try:
        r = ddb.search_vectors(
            TableName=TABLE, IndexName=INDEX,
            SearchVector=search_vec(vec(0)), TopK=10,
            SearchConditionExpression="category = :c AND #s <> :sup",
            ExpressionAttributeNames={"#s": "source"},
            ExpressionAttributeValues={":c": {"S": "config"}, ":sup": {"S": "superseded"}},
        )
    except ClientError as e:
        record("Q3 filter+similarity in one call", False,
               f"{e.response['Error']['Code']}: {e.response['Error']['Message'][:150]}")
        return
    sks = [h["Item"]["sk"]["S"] for h in r["SearchResults"]]
    ok = "config#lambda" in sks and "config#old" not in sks and "decisions#voyage" not in sks
    record("Q3 filter+similarity in one call", ok,
           f"got {sks} — category filter and source exclusion both applied" if ok
           else f"got {sks}, expected only config# items excluding superseded")


def q4_staleness() -> None:
    print("\nQ4 vector index freshness after a write")
    put("context-mcp", "note#fresh", "context-mcp", "note", "live", 6, "freshly written")
    t0 = time.time()
    seen, waited = False, 0.0
    for _ in range(40):
        r = ddb.search_vectors(TableName=TABLE, IndexName=INDEX,
                               SearchVector=search_vec(vec(6)), TopK=5)
        if any(h["Item"]["sk"]["S"] == "note#fresh" for h in r["SearchResults"]):
            seen, waited = True, time.time() - t0
            break
        time.sleep(0.5)
    record("Q4 new write becomes searchable", seen,
           f"visible after {waited:.1f}s" if seen else "not searchable within 20s")


def q5_cross_partition() -> None:
    print("\nQ5 search across all partitions (no HASH element declared)")
    r = ddb.search_vectors(TableName=TABLE, IndexName=INDEX,
                           SearchVector=search_vec(vec(3)), TopK=10)
    projects = {h["Item"]["project"]["S"] for h in r["SearchResults"]}
    ok = len(projects) > 1 or "ticketing" in projects
    record("Q5 unscoped search spans partitions", ok,
           f"hits spanned projects {sorted(projects)}")


def teardown() -> None:
    if KEEP:
        print(f"\n--keep: leaving {TABLE} in place. Delete it with:")
        print(f"  aws dynamodb delete-table --table-name {TABLE} --region {REGION}")
        return
    print(f"\ndeleting {TABLE}…")
    ddb.delete_table(TableName=TABLE)
    print("  deleted")


def main() -> None:
    try:
        ddb.describe_table(TableName=TABLE)
        raise SystemExit(f"{TABLE} already exists — delete it first, or run with --keep off a clean slate.")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    create()
    seed()
    try:
        q1_read_your_write()
        q2_ranking()
        q3_filter_plus_similarity()
        q4_staleness()
        q5_cross_partition()
    finally:
        teardown()

    print("\n" + "=" * 68)
    for q, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {q}")
    failed = [q for q, ok, _ in results if not ok]
    print("=" * 68)
    print("\nALL FIVE ANSWERED — the port can proceed on this API."
          if not failed else
          f"\n{len(failed)} QUESTION(S) UNRESOLVED — resolve before porting:\n  - " + "\n  - ".join(failed))


if __name__ == "__main__":
    main()
