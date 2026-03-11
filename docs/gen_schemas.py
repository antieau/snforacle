"""Generate docs/schemas.md at MkDocs build time.

Imports every public Pydantic model, calls model_json_schema(), and writes
a single Markdown page with collapsible JSON blocks for each type.
"""

from __future__ import annotations

import json

import mkdocs_gen_files

from snforacle.schema import (
    DenseIntMatrix,
    ElementaryDivisorsResult,
    HNFResult,
    HNFWithTransformResult,
    SNFResult,
    SNFWithTransformsResult,
    SparseEntry,
    SparseIntMatrix,
)
from snforacle.ff_schema import (
    DenseFFMatrix,
    DenseFFMatrixOut,
    FFHNFResult,
    FFHNFWithTransformResult,
    FFRankResult,
    FFSNFResult,
    FFSNFWithTransformsResult,
    SparseFFEntry,
    SparseFFMatrix,
)
from snforacle.poly_schema import (
    DensePolyMatrix,
    DensePolyMatrixOut,
    PolyElementaryDivisorsResult,
    PolyHNFResult,
    PolyHNFWithTransformResult,
    PolySNFResult,
    PolySNFWithTransformsResult,
    SparsePolyEntry,
    SparsePolyMatrix,
)

_SECTIONS: list[tuple[str, list[tuple[str, type]]]] = [
    (
        "Integer matrices",
        [
            ("SparseEntry", SparseEntry),
            ("DenseIntMatrix", DenseIntMatrix),
            ("SparseIntMatrix", SparseIntMatrix),
            ("SNFResult", SNFResult),
            ("SNFWithTransformsResult", SNFWithTransformsResult),
            ("HNFResult", HNFResult),
            ("HNFWithTransformResult", HNFWithTransformResult),
            ("ElementaryDivisorsResult", ElementaryDivisorsResult),
        ],
    ),
    (
        "Finite-field matrices (F_p)",
        [
            ("SparseFFEntry", SparseFFEntry),
            ("DenseFFMatrix", DenseFFMatrix),
            ("SparseFFMatrix", SparseFFMatrix),
            ("DenseFFMatrixOut", DenseFFMatrixOut),
            ("FFSNFResult", FFSNFResult),
            ("FFSNFWithTransformsResult", FFSNFWithTransformsResult),
            ("FFHNFResult", FFHNFResult),
            ("FFHNFWithTransformResult", FFHNFWithTransformResult),
            ("FFRankResult", FFRankResult),
        ],
    ),
    (
        "Polynomial matrices (F_p[x])",
        [
            ("SparsePolyEntry", SparsePolyEntry),
            ("DensePolyMatrix", DensePolyMatrix),
            ("SparsePolyMatrix", SparsePolyMatrix),
            ("DensePolyMatrixOut", DensePolyMatrixOut),
            ("PolySNFResult", PolySNFResult),
            ("PolySNFWithTransformsResult", PolySNFWithTransformsResult),
            ("PolyHNFResult", PolyHNFResult),
            ("PolyHNFWithTransformResult", PolyHNFWithTransformResult),
            ("PolyElementaryDivisorsResult", PolyElementaryDivisorsResult),
        ],
    ),
]

lines: list[str] = [
    "# JSON Schemas",
    "",
    "These schemas are auto-generated from the Pydantic models at documentation",
    "build time, so they always reflect the current code.",
    "",
    "Each input and output type is a valid [JSON Schema (draft 2020-12)](https://json-schema.org/).",
    "You can use them to validate requests and responses programmatically:",
    "",
    "```python",
    "import jsonschema",
    "from snforacle.schema import DenseIntMatrix",
    "",
    "schema = DenseIntMatrix.model_json_schema()",
    "jsonschema.validate(instance=my_dict, schema=schema)",
    "```",
    "",
]

for section_title, models in _SECTIONS:
    lines.append(f"## {section_title}")
    lines.append("")
    for name, model in models:
        schema = model.model_json_schema()
        schema_json = json.dumps(schema, indent=2)
        lines.append(f"### {name}")
        lines.append("")
        # Include the docstring as a brief description if present
        doc = (model.__doc__ or "").strip()
        if doc:
            # Take only the first paragraph
            first_para = doc.split("\n\n")[0].replace("\n", " ").strip()
            lines.append(first_para)
            lines.append("")
        lines.append('??? note "JSON Schema"')
        lines.append("")
        lines.append("    ```json")
        for schema_line in schema_json.splitlines():
            lines.append(f"    {schema_line}")
        lines.append("    ```")
        lines.append("")

with mkdocs_gen_files.open("schemas.md", "w") as f:
    f.write("\n".join(lines))
