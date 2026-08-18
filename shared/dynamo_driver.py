"""
DynamoDB implementation of StorageDriver, per docs/dynamodb-schema.md.

Every rule still lives in ContextStore. This file knows only how a record turns
into an item and how a neutral filter turns into a query.

TWO THINGS THAT LIVE HERE AND NOWHERE ELSE, because they are consequences of
DynamoDB rather than of the store's design:

  `searchable`  — SearchConditionExpression accepts only conjunctions of
                  equality, so search()'s "source is not retired" cannot be
                  expressed. It becomes a stored positive flag, derived from
                  `source` on every write. ContextStore never mentions it.

  PK / SK       — derived from METADATA, never by parsing the id.
                  `summary-context-mcp-config-api-gateway` has hyphens in the
                  project and in the key, so recovering the parts would mean
                  scanning for a token that happens to be a valid category. Ids
                  stay opaque handles, addressed through the by_id index.
"""
from __future__ import annotations

import os
from decimal import Decimal
from typing import Optional

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

from shared.store import RETIRED_SOURCE, SUPERSEDED_SOURCE

GSI_BY_TYPE = "by_type"
GSI_BY_ID = "by_id"
VECTOR_INDEX = "semantic"

# The embedding-guard item lives in its own partition and carries NO `type`, so
# it is invisible to every read: count() and scan() reach records through the
# by_type index or a typed filter, and a project-scoped query never names this
# partition. That is deliberate — a bookkeeping row that showed up in count()
# would break the store's own arithmetic.
GUARD_PK = "__meta__"
GUARD_SK = "embedding"

# DynamoDB caps a BatchWriteItem at 25 items.
BATCH_LIMIT = 25

# Every attribute a record item may carry, EXCEPT the embedding. Non-embedding
# reads project exactly this set, because the alternative — fetching whole
# items and discarding the vector client-side — moves and parses a 1024-float
# list per record that the caller never asked for. That is what took the /map
# page from milliseconds of data to a Lambda timeout: an atlas build reads
# every record in the store, and the vectors were ~95% of the bytes.
#
# DynamoDB projections are include-only, so this list must be COMPLETE: an
# attribute missing from it would silently vanish from projected reads. That
# failure mode is why _item() refuses to write any attribute not listed here —
# adding a new metadata field fails loudly at write time until this tuple
# learns about it.
KNOWN_ATTRIBUTES = (
    "project", "sk", "id", "type_sk", "document", "chars", "searchable",
    "category", "key", "type", "source", "timestamp", "tier",
    "chat_title", "category_corrected_from", "backfilled",
    "superseded_from", "superseded_at", "superseded_by", "superseded_reason",
    "split_index", "split_count", "unarchived_patches",
    "archived_reason", "retired_at", "retired_from_source", "retired_reason",
)
_WRITABLE_ATTRIBUTES = frozenset(KNOWN_ATTRIBUTES) | {"embedding"}

# Attribute names DynamoDB reserves; any expression naming one needs an alias.
_RESERVED = {"project", "type", "source", "key", "timestamp", "document"}

_ser = TypeSerializer()
_deser = TypeDeserializer()


def _alias(name: str) -> str:
    return f"#{name}" if name in _RESERVED else name


def _projection_names() -> dict:
    """
    ExpressionAttributeNames for a non-embedding read: every known attribute,
    each behind an alias. Aliasing all of them, not just the ones in _RESERVED,
    because DynamoDB's reserved-word list runs to hundreds of entries and a
    future attribute that happens to land on one would fail only at query time.
    """
    return {f"#pr{n}": attr for n, attr in enumerate(KNOWN_ATTRIBUTES)}


def _plain(value):
    """DynamoDB hands numbers back as Decimal; the store expects int/float."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class DynamoDriver:
    """
    One table, partitioned by project, with type-prefixed sort keys.

    `embedder` is called with a list of texts and must return one vector each —
    the same duck-typed contract Chroma's embedding functions use. The driver
    owns embedding so that a whole put() batch costs ONE call, which is the
    property that makes splitting free.
    """

    def __init__(self, table_name: str, embedder, region: Optional[str] = None,
                 client=None):
        self.table = table_name
        self.embedder = embedder
        self.ddb = client or boto3.client(
            "dynamodb", region_name=region or os.environ.get("AWS_REGION", "eu-west-1")
        )

    def assert_embedding_space(self, model_name: str) -> None:
        """
        Refuse to open a table that was written with a different embedding model.

        Chroma does this natively — a collection remembers the embedding function
        that created it and refuses a mismatched reopen — because mixing
        embedding spaces corrupts similarity search silently: every write
        succeeds, every search returns plausible nonsense, and nothing errors.
        DynamoDB has no equivalent, so it is hand-rolled here.

        Deliberately NOT part of the StorageDriver contract: it is a property of
        this backend, not of the store, and Chroma's version is likewise tested
        beside Chroma rather than in shared/conformance.py.
        """
        key = {"project": _ser.serialize(GUARD_PK), "sk": _ser.serialize(GUARD_SK)}
        resp = self.ddb.get_item(TableName=self.table, Key=key, ConsistentRead=True)
        recorded = resp.get("Item", {}).get("embedding_model", {}).get("S")
        if recorded is None:
            # First open of this table: claim the space rather than assume it.
            self.ddb.put_item(TableName=self.table, Item={
                **key, "embedding_model": _ser.serialize(model_name)})
            return
        if recorded != model_name:
            raise RuntimeError(
                f"Table '{self.table}' was written with embedding model "
                f"'{recorded}', but you are opening it with '{model_name}'. "
                "Mixing embedding spaces corrupts similarity search with no "
                "error — every write succeeds and every search returns plausible "
                "nonsense. Match the original model, or use a fresh table."
            )

    # ---- addressing ------------------------------------------------------
    @staticmethod
    def _summary_sk(category: Optional[str], key: Optional[str]) -> str:
        """
        A summary's sort key, from its natural key components.

        Factored out because it is now built in two places — when writing an
        item, and when reading one back by natural key — and the two silently
        disagreeing would turn every consistent read into a miss.
        """
        return f"summary#{category or '_'}#{key or '_'}"

    @staticmethod
    def _sort_key(record_id: str, meta: dict) -> str:
        category = meta.get("category") or "_"
        if meta.get("type") == "summary":
            return DynamoDriver._summary_sk(category, meta.get("key"))
        # A superseded copy sorts under its ORIGIN slot, then by its own id.
        #
        # The id and NOT superseded_at, which is what an earlier version used.
        # Chunk ids are content-addressed, so archiving identical text twice
        # produces the SAME id — and it recurs in practice, because when a long
        # slot is patched only the head changes and the tail split pieces are
        # byte-identical across archival events. With a timestamp in the sort
        # key those two archives became two items sharing one id: Chroma
        # collapses them on id, DynamoDB cannot, since uniqueness is (PK, SK).
        # Real data found this after the contract missed it.
        #
        # Nothing is lost by dropping the timestamp: slot_history already sorts
        # on superseded_at in code, so the "free ordering" a timestamped key
        # appeared to give was never actually being used.
        if meta.get("superseded_from"):
            return f"superseded#{category}#{meta.get('key') or '_'}#{record_id}"
        return f"chunk#{category}#{record_id}"

    @staticmethod
    def _searchable(meta: dict) -> str:
        """Retired material is the only thing held back from default search."""
        return "n" if meta.get("source") == RETIRED_SOURCE else "y"

    def _item(self, record: dict, embedding) -> dict:
        meta = {k: v for k, v in record["metadata"].items() if v is not None}
        project = meta.get("project") or "general"
        sk = self._sort_key(record["id"], meta)
        item = {
            **meta,
            "project": project,
            "sk": sk,
            "id": record["id"],
            "type_sk": f"{project}#{sk}",
            "document": record["document"],
            "chars": len(record["document"] or ""),
            "searchable": self._searchable(meta),
        }
        unknown = set(item) - _WRITABLE_ATTRIBUTES
        if unknown:
            raise ValueError(
                f"record would write attributes {sorted(unknown)} that are not in "
                "KNOWN_ATTRIBUTES. Projected (non-embedding) reads return only "
                "listed attributes, so an unlisted one would be stored but "
                "silently absent from every scan/fetch. Add it to "
                "KNOWN_ATTRIBUTES in shared/dynamo_driver.py."
            )
        serialized = {k: _ser.serialize(v) for k, v in item.items()}
        # Built directly rather than through TypeSerializer, which refuses
        # floats outright ("use Decimal instead"). A vector is a list of N, and
        # routing 1024 floats through Decimal to get the same wire format would
        # be conversion for its own sake.
        serialized["embedding"] = {"L": [{"N": str(float(x))} for x in embedding]}
        return serialized

    @staticmethod
    def _record(item: dict) -> dict:
        plain = {k: _plain(_deser.deserialize(v)) for k, v in item.items()}
        # sk / type_sk / searchable / chars are physical bookkeeping, not part of
        # the record contract, so they never surface to ContextStore.
        embedding = plain.pop("embedding", None)
        for physical in ("sk", "type_sk", "searchable", "chars"):
            plain.pop(physical, None)
        out = {
            "id": plain.pop("id"),
            "document": plain.pop("document", None),
            "metadata": plain,
        }
        if embedding is not None:
            out["embedding"] = embedding
        return out

    # ---- writes ----------------------------------------------------------
    def put(self, records: list[dict]) -> None:
        if not records:
            return
        # One embedding call for the whole batch, however many records it holds.
        needs = [r for r in records if r.get("embedding") is None]
        if needs:
            vectors = self.embedder([r["document"] for r in needs])
            for r, v in zip(needs, vectors):
                r = r  # bind for clarity
                r["embedding"] = v
        items = [self._item(r, r["embedding"]) for r in records]
        for start in range(0, len(items), BATCH_LIMIT):
            chunk = items[start:start + BATCH_LIMIT]
            request = {self.table: [{"PutRequest": {"Item": i}} for i in chunk]}
            resp = self.ddb.batch_write_item(RequestItems=request)
            # BatchWriteItem can decline part of a batch under throttling and
            # reports it rather than failing, so an unretried write is a silently
            # lost record.
            unprocessed = resp.get("UnprocessedItems") or {}
            while unprocessed.get(self.table):
                resp = self.ddb.batch_write_item(RequestItems=unprocessed)
                unprocessed = resp.get("UnprocessedItems") or {}

    def update_metadata(self, record_id: str, metadata: dict) -> None:
        """
        Metadata-only write. Keeps the document and vector untouched, and keeps
        the item where it is: the sort key was fixed when the record was created,
        and recomputing it here would create a second item instead of editing one.
        """
        located = self._locate(record_id)
        if located is None:
            return
        project, sk = located
        meta = {k: v for k, v in metadata.items() if v is not None}
        meta["searchable"] = self._searchable(meta)
        names, values, sets = {}, {}, []
        for n, (field, value) in enumerate(meta.items()):
            if field in ("project", "sk", "id", "type_sk", "document", "embedding"):
                continue
            names[f"#f{n}"] = field
            values[f":v{n}"] = _ser.serialize(value)
            sets.append(f"#f{n} = :v{n}")
        if not sets:
            return
        self.ddb.update_item(
            TableName=self.table,
            Key={"project": _ser.serialize(project), "sk": _ser.serialize(sk)},
            UpdateExpression="SET " + ", ".join(sets),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def remove(self, ids: list[str]) -> None:
        for record_id in ids:
            located = self._locate(record_id)
            if located is None:
                continue
            project, sk = located
            self.ddb.delete_item(
                TableName=self.table,
                Key={"project": _ser.serialize(project), "sk": _ser.serialize(sk)},
            )

    # ---- reads -----------------------------------------------------------
    def _locate(self, record_id: str) -> Optional[tuple]:
        """id -> (project, sk), via the by_id index. Ids are opaque by design."""
        resp = self.ddb.query(
            TableName=self.table, IndexName=GSI_BY_ID,
            KeyConditionExpression="id = :i",
            ExpressionAttributeValues={":i": _ser.serialize(record_id)},
            ProjectionExpression="#p, sk",
            ExpressionAttributeNames={"#p": "project"},
        )
        items = resp.get("Items") or []
        if not items:
            return None
        return (_deser.deserialize(items[0]["project"]),
                _deser.deserialize(items[0]["sk"]))

    def fetch(self, ids: list[str], *, with_embeddings: bool = False) -> list[dict]:
        out = []
        for record_id in ids:
            kwargs = {
                "TableName": self.table, "IndexName": GSI_BY_ID,
                "KeyConditionExpression": "id = :i",
                "ExpressionAttributeValues": {":i": _ser.serialize(record_id)},
            }
            if not with_embeddings:
                names = _projection_names()
                kwargs["ProjectionExpression"] = ", ".join(names)
                kwargs["ExpressionAttributeNames"] = names
            resp = self.ddb.query(**kwargs)
            for item in resp.get("Items") or []:
                rec = self._record(item)
                if not with_embeddings:
                    rec.pop("embedding", None)
                out.append(rec)
        return out

    def fetch_slot(self, record_id: str, *, project: Optional[str], category: str,
                   key: Optional[str],
                   with_embeddings: bool = False) -> Optional[dict]:
        """
        Read one summary slot by its NATURAL KEY, strongly consistently.

        Not fetch(): that resolves an id through the by_id GSI, and DynamoDB
        refuses ConsistentRead on any global secondary index ("Consistent reads
        are not supported on global secondary indexes"), because an index is
        maintained asynchronously after the base-table write returns. A summary
        read that lands inside that window returns the previous version, and
        patch_summary derives its match, its archive copy and its patch counter
        from it — so the write that follows silently discards whatever landed in
        between, while every built-in check still passes.

        The components sidestep the index entirely: PK and SK are exactly what a
        summary item is stored under, so this is a GetItem on the base table,
        where ConsistentRead is available and read-your-writes holds.
        """
        kwargs = {
            "TableName": self.table,
            "Key": {
                "project": _ser.serialize(project or "general"),
                "sk": _ser.serialize(self._summary_sk(category, key)),
            },
            "ConsistentRead": True,
        }
        if not with_embeddings:
            names = _projection_names()
            kwargs["ProjectionExpression"] = ", ".join(names)
            kwargs["ExpressionAttributeNames"] = names
        item = self.ddb.get_item(**kwargs).get("Item")
        if not item:
            return None
        rec = self._record(item)
        if not with_embeddings:
            rec.pop("embedding", None)
        return rec

    def scan(self, filt: dict, *, with_documents: bool = True,
             with_embeddings: bool = False) -> list[dict]:
        items = self._query(filt, with_embeddings=with_embeddings)
        recs = [self._record(i) for i in items]
        for r in recs:
            if not with_embeddings:
                r.pop("embedding", None)
            if not with_documents:
                r["document"] = None
        return recs

    def count(self, filt: Optional[dict] = None) -> int:
        if not filt:
            # No Scan: every record carries a type, so the two type partitions
            # of the by_type index cover the store. This also excludes any
            # non-record bookkeeping item by construction.
            return (len(self._query({"type": "summary"}))
                    + len(self._query({"type": "chunk"})))
        return len(self._query(filt))

    def similar(self, text: str, filt: dict, top_k: int) -> list[dict]:
        vector = self.embedder([text])[0]
        exclude = tuple(filt.get("exclude_sources") or ())
        if exclude and set(exclude) != {RETIRED_SOURCE}:
            # Only the retired exclusion has a stored flag behind it. Anything
            # else would need an inequality the vector filter grammar does not
            # have, and silently ignoring it would return known-wrong material.
            raise ValueError(
                f"vector search cannot express exclude_sources={exclude!r}; only "
                f"({RETIRED_SOURCE!r},) is supported, via the searchable flag"
            )
        conditions, names, values = [], {}, {}
        for field in ("project", "category", "type"):
            if filt.get(field) is not None:
                conditions.append(f"{_alias(field)} = :{field}")
                if field in _RESERVED:
                    names[f"#{field}"] = field
                values[f":{field}"] = _ser.serialize(filt[field])
        if exclude:
            conditions.append("searchable = :searchable")
            values[":searchable"] = _ser.serialize("y")

        kwargs = {
            "TableName": self.table, "IndexName": VECTOR_INDEX,
            "SearchVector": [{"N": str(float(x))} for x in vector],
            "TopK": top_k,
        }
        if conditions:
            kwargs["SearchConditionExpression"] = " AND ".join(conditions)
            kwargs["ExpressionAttributeValues"] = values
            if names:
                kwargs["ExpressionAttributeNames"] = names

        resp = self.ddb.search_vectors(**kwargs)
        out = []
        for hit in resp.get("SearchResults") or []:
            rec = self._record(hit["Item"])
            rec.pop("embedding", None)
            # Score is a DISTANCE under COSINE — 0.0 is identical. Passed
            # through under the name the store already uses.
            rec["distance"] = hit.get("Score")
            out.append(rec)
        return out

    # ---- query planning --------------------------------------------------
    def _sk_prefix(self, filt: dict) -> Optional[str]:
        """
        The narrowest sort-key prefix a filter implies, or None.

        type=chunk deliberately yields no prefix: both `chunk#` and
        `superseded#` items are chunks, so a prefix would silently drop the
        archived half.
        """
        if filt.get("type") == "summary":
            if filt.get("category"):
                return f"summary#{filt['category']}#"
            return "summary#"
        if filt.get("source") == SUPERSEDED_SOURCE or filt.get("superseded_from"):
            if filt.get("category") and filt.get("key"):
                return f"superseded#{filt['category']}#{filt['key']}#"
            if filt.get("category"):
                return f"superseded#{filt['category']}#"
            return "superseded#"
        return None

    def _query(self, filt: dict, *, with_embeddings: bool = False) -> list[dict]:
        filt = dict(filt or {})
        # Fields consumed by the key condition must not be re-applied as a
        # filter on an attribute that may be absent (a keyless legacy chunk).
        equality = {f: filt[f] for f in
                    ("project", "category", "type", "source", "key", "superseded_from")
                    if filt.get(f) is not None}
        exclude = tuple(filt.get("exclude_sources") or ())

        if filt.get("project") is not None:
            key_expr = "#p = :p"
            names = {"#p": "project"}
            values = {":p": _ser.serialize(filt["project"])}
            prefix = self._sk_prefix(filt)
            if prefix:
                key_expr += " AND begins_with(sk, :skp)"
                values[":skp"] = _ser.serialize(prefix)
            equality.pop("project", None)
            return self._run(None, key_expr, names, values, equality, exclude,
                             with_embeddings=with_embeddings)

        if filt.get("type") is not None:
            key_expr = "#t = :t"
            names = {"#t": "type"}
            values = {":t": _ser.serialize(filt["type"])}
            equality.pop("type", None)
            return self._run(GSI_BY_TYPE, key_expr, names, values, equality, exclude,
                             with_embeddings=with_embeddings)

        # No project and no type: walk both type partitions rather than Scan.
        out = []
        for record_type in ("summary", "chunk"):
            out.extend(self._query({**filt, "type": record_type},
                                   with_embeddings=with_embeddings))
        return out

    def _run(self, index: Optional[str], key_expr: str, names: dict, values: dict,
             equality: dict, exclude: tuple, *, with_embeddings: bool = False) -> list[dict]:
        conditions = []
        for n, (field, value) in enumerate(equality.items()):
            names[f"#e{n}"] = field
            values[f":e{n}"] = _ser.serialize(value)
            conditions.append(f"#e{n} = :e{n}")
        if exclude:
            names["#src"] = "source"
            parts = []
            for n, src in enumerate(exclude):
                values[f":x{n}"] = _ser.serialize(src)
                parts.append(f"#src <> :x{n}")
            # A FilterExpression is NOT SearchConditionExpression: <> is fine here.
            conditions.append("(" + " AND ".join(parts) + ")")

        kwargs = {
            "TableName": self.table,
            "KeyConditionExpression": key_expr,
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": values,
        }
        if not with_embeddings:
            projection = _projection_names()
            kwargs["ProjectionExpression"] = ", ".join(projection)
            names.update(projection)
        if index:
            kwargs["IndexName"] = index
        if conditions:
            kwargs["FilterExpression"] = " AND ".join(conditions)

        items: list[dict] = []
        while True:
            resp = self.ddb.query(**kwargs)
            items.extend(resp.get("Items") or [])
            # Explicit, unlike Chroma's silent cap: you always know there is more.
            last = resp.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
        return items
