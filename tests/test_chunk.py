from pathlib import Path

from repopilot.ingest.chunk import chunk_file, chunk_repo

PY = '''import os

CONST = 1


def alpha(x):
    return x + 1


@decorator
class Beta:
    def method(self):
        return 2
'''


def test_python_chunks_follow_definitions(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text(PY)
    chunks = chunk_file(f, "mod.py")
    symbols = [c.symbol for c in chunks]
    assert "def alpha" in symbols
    assert "class Beta" in symbols
    beta = next(c for c in chunks if c.symbol == "class Beta")
    assert beta.text.startswith("@decorator")  # decorators belong to their definition
    lines = PY.splitlines()
    assert lines[beta.start_line - 1] == "@decorator"
    assert lines[beta.end_line - 1].strip() == "return 2"


def test_line_numbers_match_source(tmp_path: Path):
    f = tmp_path / "notes.md"
    lines = [f"line {i}" for i in range(1, 151)]
    f.write_text("\n".join(lines))
    for c in chunk_file(f, "notes.md"):
        assert c.text.splitlines()[0] == lines[c.start_line - 1]
        assert c.text.splitlines()[-1] == lines[c.end_line - 1]


def test_repo_walk_skips_vendored_and_binary(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("var x = 1;")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "bundle.min.js").write_text("var a=1;")
    paths = {c.path for c in chunk_repo(tmp_path)}
    assert paths == {"src/app.py"}
