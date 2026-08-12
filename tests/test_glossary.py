"""Persistent known-terms glossary: read / append-with-dedup / default path."""

from meetscribe import glossary


def test_load_missing_file_is_empty(tmp_path):
    assert glossary.load(tmp_path / "glossary.txt") == []


def test_load_skips_blanks_and_comments(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("Borderless\n\n# a comment\n  Georg  \n", encoding="utf-8")
    assert glossary.load(p) == ["Borderless", "Georg"]


def test_append_dedups_case_insensitively(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("Borderless\n", encoding="utf-8")
    merged = glossary.append(p, ["georg", "BORDERLESS", "Christian"])
    assert merged == ["Borderless", "georg", "Christian"]
    assert glossary.load(p) == ["Borderless", "georg", "Christian"]


def test_append_creates_missing_file(tmp_path):
    p = tmp_path / "sub" / "g.txt"
    glossary.append(p, ["Ana"])
    assert glossary.load(p) == ["Ana"]


def test_append_empty_terms_is_noop(tmp_path):
    p = tmp_path / "g.txt"
    assert glossary.append(p, ["", "   "]) == []
    assert not p.exists()


def test_default_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert glossary.default_path() == tmp_path / "meetscribe" / "glossary.txt"
