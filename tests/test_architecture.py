"""Enforce the layering rules from docs/ARCHITECTURE.md (section 3).

Each module may import only from its own layer or layers above it (lower numbers).
The maths layers (2 and 3) may not import networking or UI libraries.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "riskengine"

LAYERS = {
    "config": 0,
    "conventions": 0,
    "data": 1,
    "download": 1,
    "risk": 2,
    "backtest": 3,
    "attribution": 3,
    "stress": 3,
    "pipeline": 4,
    "dashboard": 5,
}
FORBIDDEN_IN_MATHS = {"urllib", "requests", "http", "socket", "streamlit", "plotly"}


def _layer_of(module: str) -> int | None:
    parts = module.split(".")
    if parts[0] != "riskengine" or len(parts) < 2:
        return None
    return LAYERS.get(parts[1])


def _module_name(path: Path) -> str:
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
    return found


def _source_files():
    return [p for p in PKG.rglob("*.py") if _layer_of(_module_name(p)) is not None]


def test_every_top_level_module_has_a_layer():
    for p in PKG.iterdir():
        name = p.stem if p.suffix == ".py" else p.name
        if name.startswith("__") or name.startswith("."):
            continue
        assert name in LAYERS, f"{name} is not assigned to a layer in this test"


def test_imports_only_flow_downward():
    violations = []
    for path in _source_files():
        mod = _module_name(path)
        own = _layer_of(mod)
        for imp in _imports(path):
            target = _layer_of(imp)
            if target is not None and target > own:
                violations.append(f"{mod} (layer {own}) imports {imp} (layer {target})")
    assert not violations, "\n".join(violations)


def test_maths_layers_do_no_network_or_ui_work():
    violations = []
    for path in _source_files():
        mod = _module_name(path)
        if _layer_of(mod) not in (2, 3):
            continue
        for imp in _imports(path):
            if imp.split(".")[0] in FORBIDDEN_IN_MATHS:
                violations.append(f"{mod} imports {imp}")
    assert not violations, "\n".join(violations)
