import ast
from pathlib import Path


RUNTIME_DIR = Path(__file__).parents[2] / "runtime"
# `weight_utils.py` already gates loading bars by torch distributed rank.
ALLOWED_TQDM_FILES = {
    Path("loader/weight_utils.py"),
    Path("utils/progress.py"),
}


def _is_tqdm_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module is not None and node.module.startswith("tqdm")
    if isinstance(node, ast.Import):
        return any(
            alias.name == "tqdm" or alias.name.startswith("tqdm.")
            for alias in node.names
        )
    return False


def _is_tqdm_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "tqdm"
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "tqdm"
    return False


def test_runtime_progress_bars_are_rank_gated():
    offenders = []

    for path in sorted(RUNTIME_DIR.rglob("*.py")):
        rel_path = path.relative_to(RUNTIME_DIR)
        if rel_path in ALLOWED_TQDM_FILES:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_tqdm_import(node):
                offenders.append(f"{rel_path}:{node.lineno} imports tqdm directly")
            elif _is_tqdm_call(node):
                offenders.append(f"{rel_path}:{node.lineno} creates a raw tqdm")

    assert not offenders, "Use rank_zero_tqdm for runtime progress bars:\n" + "\n".join(
        offenders
    )
