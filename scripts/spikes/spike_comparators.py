#!/usr/bin/env python3
"""
Follow-up spike: which comparators does SearchConditionExpression accept?

The first spike established that filter and similarity DO combine in one call,
but that `<>` is rejected. That single rejection matters more than it looks:
ContextStore.search() excludes retired chunks with {"source": {"$ne": "retired"}},
so if not-equal is unavailable the default read path cannot be expressed as
written and needs a schema change rather than a translation.

This enumerates what actually parses, so the port is designed against the real
grammar instead of a guess.
"""
from __future__ import annotations

import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION, TABLE, INDEX, DIMS = "eu-west-1", "context-mcp-comparator-spike", "vec", 1024
ddb = boto3.client("dynamodb", region_name=REGION)


def vec(slot, width=64):
    v = [0.0] * DIMS
    for i in range(slot * width, slot * width + width):
        v[i] = 1.0
    return v


def sv(v):
    return [{"N": str(x)} for x in v]


def create():
    print(f"creating {TABLE}…")
    ddb.create_table(
        TableName=TABLE, BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": n, "AttributeType": "S"}
            for n in ("pk", "sk", "project", "category", "source", "visible")
        ],
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                   {"AttributeName": "sk", "KeyType": "RANGE"}],
        VectorIndexes=[{
            "IndexName": INDEX,
            "VectorAttribute": {"AttributeName": "embedding"},
            "SearchSchema": [
                {"AttributeName": "project", "SearchSchemaElementType": "INLINE_FILTER"},
                {"AttributeName": "category", "SearchSchemaElementType": "INLINE_FILTER"},
                {"AttributeName": "source", "SearchSchemaElementType": "INLINE_FILTER"},
                {"AttributeName": "visible", "SearchSchemaElementType": "INLINE_FILTER"},
            ],
            "Projection": {"ProjectionType": "ALL"},
            "Dimensions": DIMS, "DistanceFunction": "COSINE",
        }],
    )
    for _ in range(60):
        d = ddb.describe_table(TableName=TABLE)["Table"]
        if d["TableStatus"] == "ACTIVE" and all(
                i.get("IndexStatus") == "ACTIVE" for i in d.get("VectorIndexes", [{}])):
            print("  ACTIVE"); return
        time.sleep(5)
    raise SystemExit("table never went ACTIVE")


def seed():
    rows = [
        ("context-mcp", "config#lambda",    "config",    "live",       "y", 0),
        ("context-mcp", "decisions#voyage", "decisions", "live",       "y", 1),
        ("context-mcp", "config#old",       "config",    "superseded", "y", 2),
        ("context-mcp", "config#wrong",     "config",    "retired",    "n", 3),
        ("ticketing",   "config#domains",   "config",    "live",       "y", 4),
    ]
    for pk, sk, cat, src, vis, slot in rows:
        ddb.put_item(TableName=TABLE, Item={
            "pk": {"S": pk}, "sk": {"S": sk}, "project": {"S": pk},
            "category": {"S": cat}, "source": {"S": src}, "visible": {"S": vis},
            "embedding": {"L": [{"N": str(x)} for x in vec(slot)]},
        })
    print(f"  seeded {len(rows)} items (one retired, marked visible=n)")


CASES = [
    ("equality",            "category = :cfg",                          {":cfg": {"S": "config"}}, None),
    ("not-equal <>",        "#s <> :ret",                               {":ret": {"S": "retired"}}, {"#s": "source"}),
    ("NOT equality",        "NOT #s = :ret",                            {":ret": {"S": "retired"}}, {"#s": "source"}),
    ("IN list",             "#s IN (:a, :b, :c)",                       {":a": {"S": "live"}, ":b": {"S": "backfill"}, ":c": {"S": "superseded"}}, {"#s": "source"}),
    ("AND of equalities",   "category = :cfg AND project = :p",         {":cfg": {"S": "config"}, ":p": {"S": "context-mcp"}}, None),
    ("OR of equalities",    "category = :cfg OR category = :dec",       {":cfg": {"S": "config"}, ":dec": {"S": "decisions"}}, None),
    ("begins_with",         "begins_with(category, :pre)",              {":pre": {"S": "conf"}}, None),
    ("attribute_exists",    "attribute_exists(category)",               {}, None),
    ("positive flag",       "visible = :y",                             {":y": {"S": "y"}}, None),
    ("flag AND category",   "visible = :y AND category = :cfg",         {":y": {"S": "y"}, ":cfg": {"S": "config"}}, None),
]


def probe():
    print("\n%-20s %-8s %s" % ("expression", "result", "hits / error"))
    print("-" * 92)
    supported = []
    for label, expr, vals, names in CASES:
        kw = dict(TableName=TABLE, IndexName=INDEX, SearchVector=sv(vec(0)), TopK=10,
                  SearchConditionExpression=expr)
        if vals:
            kw["ExpressionAttributeValues"] = vals
        if names:
            kw["ExpressionAttributeNames"] = names
        try:
            r = ddb.search_vectors(**kw)
            sks = [h["Item"]["sk"]["S"] for h in r["SearchResults"]]
            print("%-20s %-8s %s" % (label, "OK", sks))
            supported.append(label)
        except ClientError as e:
            msg = e.response["Error"]["Message"]
            print("%-20s %-8s %s" % (label, "REJECT", msg[:64]))
    return supported


def main():
    try:
        ddb.describe_table(TableName=TABLE)
        raise SystemExit(f"{TABLE} exists — delete it first.")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    create(); seed()
    try:
        ok = probe()
    finally:
        print(f"\ndeleting {TABLE}…")
        ddb.delete_table(TableName=TABLE)
        print("  deleted")
    print("\nSUPPORTED:", ", ".join(ok) or "none")


if __name__ == "__main__":
    main()
