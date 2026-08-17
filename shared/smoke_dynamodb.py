"""
Runs the SAME contract as shared/smoke.py, against DynamoDB.

This is the port's proof. If shared/conformance.py passes here, the DynamoDB
backend is behaviourally equivalent to Chroma on every rule the store has —
which is the reason the contract was extracted in the first place.

Creates a throwaway table, runs, and deletes it. The offline hash embedder is
used rather than Voyage: the contract exercises hundreds of writes and only
cares that similar text ranks together, so paying for real embeddings would buy
nothing but latency. Table dimensions therefore follow the embedder, not
production's 1024.

    python3 -m shared.smoke_dynamodb            # create, run, delete
    python3 -m shared.smoke_dynamodb --keep     # leave the table for poking
"""
from __future__ import annotations

import sys
import time

import boto3
from botocore.exceptions import ClientError

from shared import conformance
from shared.dynamo_driver import GSI_BY_ID, GSI_BY_TYPE, VECTOR_INDEX, DynamoDriver
from shared.store import ContextStore, _OfflineHashEmbedding

REGION = "eu-west-1"
TABLE = "context-mcp-conformance"
DIMS = 384  # _OfflineHashEmbedding's width


def create_table(ddb, table: str, dims: int) -> None:
    print(f"creating {table} (dims={dims}, COSINE)…")
    ddb.create_table(
        TableName=table,
        BillingMode="PAY_PER_REQUEST",  # vector indexes are on-demand only
        AttributeDefinitions=[
            # Key attributes, GSI keys, and every SearchSchema attribute must all
            # be declared — CreateTable rejects a SearchSchema naming anything
            # absent from here, which is why the filterable set is fixed now.
            {"AttributeName": "project", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "id", "AttributeType": "S"},
            {"AttributeName": "type", "AttributeType": "S"},
            {"AttributeName": "type_sk", "AttributeType": "S"},
            {"AttributeName": "category", "AttributeType": "S"},
            {"AttributeName": "searchable", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "project", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        GlobalSecondaryIndexes=[
            {
                # Serves get_index() with no project — the one access pattern
                # that does not decompose per-project.
                "IndexName": GSI_BY_TYPE,
                "KeySchema": [
                    {"AttributeName": "type", "KeyType": "HASH"},
                    {"AttributeName": "type_sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                # Ids are opaque handles: search results carry them and
                # archive(id)/record(id) consume them, but they cannot be parsed
                # back into an address.
                "IndexName": GSI_BY_ID,
                "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        VectorIndexes=[
            {
                "IndexName": VECTOR_INDEX,
                "VectorAttribute": {"AttributeName": "embedding"},
                "SearchSchema": [
                    {"AttributeName": "project", "SearchSchemaElementType": "INLINE_FILTER"},
                    {"AttributeName": "category", "SearchSchemaElementType": "INLINE_FILTER"},
                    {"AttributeName": "type", "SearchSchemaElementType": "INLINE_FILTER"},
                    {"AttributeName": "searchable", "SearchSchemaElementType": "INLINE_FILTER"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "Dimensions": dims,
                "DistanceFunction": "COSINE",
            }
        ],
    )
    for _ in range(90):
        d = ddb.describe_table(TableName=table)["Table"]
        gsis = {i["IndexName"]: i["IndexStatus"] for i in d.get("GlobalSecondaryIndexes", [])}
        vecs = {i["IndexName"]: i.get("IndexStatus", "?") for i in d.get("VectorIndexes", [])}
        if (d["TableStatus"] == "ACTIVE"
                and all(s == "ACTIVE" for s in gsis.values())
                and vecs and all(s == "ACTIVE" for s in vecs.values())):
            print(f"  ACTIVE — gsis={gsis} vector={vecs}")
            return
        time.sleep(5)
    raise SystemExit("table did not become ACTIVE")


def main() -> None:
    keep = "--keep" in sys.argv
    ddb = boto3.client("dynamodb", region_name=REGION)

    try:
        ddb.describe_table(TableName=TABLE)
        raise SystemExit(f"{TABLE} already exists — delete it first:\n"
                         f"  aws dynamodb delete-table --table-name {TABLE} --region {REGION}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    create_table(ddb, TABLE, DIMS)
    try:
        embedder = _OfflineHashEmbedding(dim=DIMS)

        def make_dynamo_store() -> ContextStore:
            return ContextStore(driver=DynamoDriver(TABLE, embedder, region=REGION))

        conformance.run(make_dynamo_store)
        print("\nDynamoDB satisfies the contract.")
    finally:
        if keep:
            print(f"\n--keep: {TABLE} left in place. Delete with:")
            print(f"  aws dynamodb delete-table --table-name {TABLE} --region {REGION}")
        else:
            print(f"\ndeleting {TABLE}…")
            ddb.delete_table(TableName=TABLE)
            print("  deleted")


if __name__ == "__main__":
    main()
