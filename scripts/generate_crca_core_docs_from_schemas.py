"""Generate lightweight Markdown docs from exported JSON schemas.

This intentionally documents *structured contracts* (schemas), not narratives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_schema(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_title(schema: Dict[str, Any], fallback: str) -> str:
    return schema.get("title") or fallback


def _collect_properties(schema: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    rows: List[Tuple[str, str, str]] = []
    for name, p in props.items():
        typ = p.get("type") or p.get("$ref") or "unknown"
        req = "required" if name in required else "optional"
        desc = (p.get("description") or "").replace("\n", " ").strip()
        rows.append((name, str(typ), f"{req}. {desc}".strip()))
    return rows


def _render_md(name: str, schema: Dict[str, Any]) -> str:
    title = _schema_title(schema, name)
    rows = _collect_properties(schema)

    lines: List[str] = []
    lines.append(f"## `{title}`")
    lines.append("")
    lines.append("This page is generated from the JSON schema (contract-first).")
    lines.append("")
    if not rows:
        lines.append("_No top-level properties found in schema._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Field | Type | Notes |")
    lines.append("|---|---|---|")
    for field, typ, notes in rows:
        lines.append(f"| `{field}` | `{typ}` | {notes} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    schema_dir = REPO_ROOT / "schemas_export" / "crca_core"
    if not schema_dir.exists():
        raise SystemExit(f"Missing {schema_dir}. Run scripts/export_crca_core_schemas.py first.")

    out_dir = REPO_ROOT / "docs_generated" / "crca_core"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "# crca_core schema contracts (generated)",
        "",
        "These documents are generated from exported Pydantic JSON schemas.",
        "",
    ]

    for schema_path in sorted(schema_dir.glob("*.schema.json")):
        name = schema_path.name.replace(".schema.json", "")
        schema = _load_schema(schema_path)
        md = _render_md(name, schema)
        (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
        index_lines.append(f"- `{name}`: `{name}.md`")

    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()

