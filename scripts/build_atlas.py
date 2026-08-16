#!/usr/bin/env python3
"""Build the context atlas: a self-contained page over the whole store.

Two views on one selection — a catalogue of slot cards with history beneath
each category, and a horizontal band of skill-tree constellations, one per
project. See decisions/map-design and decisions/map-honesty in the store for
the form and for what it deliberately refuses to draw.

This bakes a STATIC snapshot into the file, which is why it needs no auth and
why it is stale the moment anything is written. Re-run it to refresh.

    python3 scripts/build_atlas.py [output.html]

Output defaults to ~/context-atlas.html, deliberately outside this repo: the
page embeds every stored entry and the repo is public.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from shared.store import ContextStore
# The transform is shared with the served /map route, so the page this script
# writes and the page the server renders cannot drift apart. See shared/atlas.py
# for what it refuses to compute.
from shared.atlas import build_atlas_data, render_page

s = ContextStore()
atlas = build_atlas_data(s)
projects = atlas["projects"]

# Default output is OUTSIDE the repo: the page embeds the whole store, and this
# repo is public. Pass a path to put it elsewhere.
out = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "context-atlas.html"
out.write_text(render_page(atlas), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size:,} bytes)")
print(f"{len(projects)} constellations · {sum(len(p['slots']) for p in projects)} stars "
      f"· {sum(len(p['history']) for p in projects)} history motes")
for p in projects[:4]:
    cats = len({s['cat'] for s in p['slots']})
    print(f"   {p['name']:<28} {cats} branches, {len(p['slots'])} stars")
