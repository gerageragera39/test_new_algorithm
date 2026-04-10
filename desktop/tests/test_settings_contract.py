from __future__ import annotations

import ast
from pathlib import Path


_KNOWN_SETTINGS_MEMBERS = {
    "ensure_directories",
    "model_copy",
    "resolved_embedding_mode",
}


def _settings_fields(config_path: Path) -> set[str]:
    module = ast.parse(config_path.read_text(encoding="utf-8-sig"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    msg = f"Could not find Settings class in {config_path}"
    raise AssertionError(msg)


def _collect_settings_attribute_usage(project_root: Path) -> set[str]:
    used: set[str] = set()
    for package_dir in (project_root / "app", project_root / "desktop", project_root / "tests"):
        for path in package_dir.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))

            class Visitor(ast.NodeVisitor):
                def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
                    value = node.value
                    if (
                        isinstance(value, ast.Attribute)
                        and isinstance(value.value, ast.Name)
                        and value.value.id == "self"
                        and value.attr == "settings"
                    ):
                        used.add(node.attr)
                    elif isinstance(value, ast.Name) and value.id == "settings":
                        used.add(node.attr)
                    self.generic_visit(node)

            Visitor().visit(tree)
    return used


def test_all_settings_attributes_used_in_desktop_code_are_declared() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "app" / "core" / "config.py"

    declared = _settings_fields(config_path)
    used = _collect_settings_attribute_usage(project_root)
    missing = sorted(name for name in used if name not in declared and name not in _KNOWN_SETTINGS_MEMBERS)

    assert missing == []


def test_desktop_settings_match_main_project_when_available() -> None:
    desktop_root = Path(__file__).resolve().parents[1]
    main_config_path = desktop_root.parent / "app" / "core" / "config.py"
    desktop_config_path = desktop_root / "app" / "core" / "config.py"

    if not main_config_path.exists():
        return

    main_fields = _settings_fields(main_config_path)
    desktop_fields = _settings_fields(desktop_config_path)

    assert desktop_fields == main_fields
