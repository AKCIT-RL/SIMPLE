"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

------------------------------------------------------------------------------
Generate LEFT/RIGHT shelf variants of the MobileShelvingCart attach MJCF.

The sorting task needs two shelf instances in the same scene. The engine attaches
each furniture MJCF via MjSpec.from_file + attach *without* a name prefix, so
attaching the same file twice fails with "repeated name". This script clones the
cart's attach XML into `MobileShelvingCart_C05_left_attach.xml` and
`..._right_attach.xml`, renaming only the identifiers (model / body `name=` and
`mesh=` references) while keeping the `file=` paths untouched — both variants
share the exact same OBJ meshes on disk, so there is no asset duplication.

Usage:
    .venv/bin/python scripts/industrial/make_shelf_variants.py
"""

from __future__ import annotations

import os
import re

_BASE = "MobileShelvingCart_C05_01"
_DIR = os.path.join("data", "assets", "industrial", _BASE)
_VARIANTS = ["MobileShelvingCart_C05_left", "MobileShelvingCart_C05_right"]


def make_variant(src_xml: str, variant: str) -> str:
    with open(src_xml) as f:
        xml = f.read()
    # Rename identifiers only: model="...", name="...", mesh="..." — never file="...".
    xml = re.sub(rf'model="{_BASE}', f'model="{variant}', xml)
    xml = re.sub(rf'name="{_BASE}', f'name="{variant}', xml)
    xml = re.sub(rf'mesh="{_BASE}', f'mesh="{variant}', xml)
    out = os.path.join(_DIR, f"{variant}_attach.xml")
    with open(out, "w") as f:
        f.write(xml)
    return out


def main() -> None:
    src = os.path.join(_DIR, f"{_BASE}_attach.xml")
    assert os.path.exists(src), f"missing {src} — run extract_furniture_mesh.py first"
    for v in _VARIANTS:
        out = make_variant(src, v)
        # sanity: no dangling old identifiers outside file= attributes
        with open(out) as f:
            txt = f.read()
        leftovers = [
            m for m in re.findall(rf'(?:name|mesh|model)="({_BASE}[^"]*)"', txt)
        ]
        assert not leftovers, f"rename incomplete in {out}: {leftovers[:3]}"
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
