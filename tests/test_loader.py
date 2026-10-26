"""Тесты загрузки корпуса (kbrag.ingest.loader).

Главное свойство: ``Document.text`` совпадает с телом файла символ в символ. Иначе смещения
фрагментов и эталонные цитаты «поедут», и оценка качества поиска станет нечестной.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kbrag.ingest import load_corpus, load_document, parse_front_matter

# ---------------------------------------------------------------- parse_front_matter


def test_parse_front_matter_splits_meta_and_body() -> None:
    meta, body = parse_front_matter("---\nid: it-vpn\ntitle: VPN\n---\n# VPN\n\nТекст.\n")
    assert meta == {"id": "it-vpn", "title": "VPN"}
    assert body == "# VPN\n\nТекст.\n"


def test_parse_front_matter_converts_dates_to_iso_strings() -> None:
    raw = "---\nid: a\nupdated: 2025-03-01\nreviewed: 2024-12-31 10:30:00\nhistory: [2023-02-10]\n---\nТекст"
    meta, _ = parse_front_matter(raw)
    assert meta["updated"] == "2025-03-01"
    assert meta["reviewed"] == "2024-12-31"
    assert meta["history"] == ["2023-02-10"]
    assert all(isinstance(value, str) for value in (meta["updated"], meta["reviewed"]))


def test_parse_front_matter_keeps_other_yaml_types() -> None:
    meta, _ = parse_front_matter("---\nid: a\nversion: 3\ndraft: false\ntags: [hr, vpn]\n---\n")
    assert meta == {"id": "a", "version": 3, "draft": False, "tags": ["hr", "vpn"]}


@pytest.mark.parametrize("raw", ["# Просто markdown\n\nБез шапки.", "", "Текст\n---\nid: x\n---\n"])
def test_parse_front_matter_without_header_returns_raw(raw: str) -> None:
    assert parse_front_matter(raw) == ({}, raw)


def test_parse_front_matter_empty_header() -> None:
    assert parse_front_matter("---\n---\nТекст") == ({}, "Текст")


def test_parse_front_matter_handles_crlf_and_bom() -> None:
    meta, body = parse_front_matter("﻿---\r\nid: a\r\ntitle: T\r\n---\r\nТекст\r\n")
    assert meta == {"id": "a", "title": "T"}
    assert body == "Текст\r\n"


def test_parse_front_matter_body_may_contain_horizontal_rules() -> None:
    # Вторая строка '---' закрывает шапку, а следующие '---' — это уже часть текста.
    meta, body = parse_front_matter("---\nid: a\n---\nАбзац\n---\nЕщё абзац\n")
    assert meta == {"id": "a"}
    assert body == "Абзац\n---\nЕщё абзац\n"


def test_parse_front_matter_unclosed_header_raises() -> None:
    with pytest.raises(ValueError, match="не закрыт"):
        parse_front_matter("---\nid: a\ntitle: T\n# Текст без закрывающей строки\n")


@pytest.mark.parametrize("yaml_text", ["id: [незакрытый список", "- просто\n- список"])
def test_parse_front_matter_bad_yaml_raises(yaml_text: str) -> None:
    with pytest.raises(ValueError):
        parse_front_matter(f"---\n{yaml_text}\n---\nТекст")


# ---------------------------------------------------------------- load_document


def test_load_document_matches_mini_docs(mini_corpus_dir: Path, mini_docs) -> None:
    for expected in mini_docs:
        dept = expected.metadata["department"].lower()
        doc = load_document(mini_corpus_dir / dept / f"{expected.doc_id}.md", root=mini_corpus_dir)
        assert doc.doc_id == expected.doc_id
        assert doc.title == expected.title
        assert doc.text == expected.text  # тело как есть, включая финальный перевод строки
        assert doc.metadata["source_path"] == f"{dept}/{expected.doc_id}.md"
        assert {k: v for k, v in doc.metadata.items() if k != "source_path"} == expected.metadata


def test_load_document_metadata_excludes_id_and_title(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: d1\ntitle: T\nowner: HR-отдел\nstatus: active\n---\nТекст\n", encoding="utf-8")
    doc = load_document(path)
    assert doc.metadata == {"owner": "HR-отдел", "status": "active", "source_path": path.as_posix()}


def test_load_document_strips_only_leading_newlines(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: d1\ntitle: T\n---\n\n\n  Отступ сохраняется.\n\n", encoding="utf-8")
    assert load_document(path).text == "  Отступ сохраняется.\n\n"


def test_load_document_title_falls_back_to_first_h1(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: d1\n---\nВступление\n## Раздел\n# Настоящий заголовок\n", encoding="utf-8")
    assert load_document(path).title == "Настоящий заголовок"


def test_load_document_title_falls_back_to_doc_id(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: d1\ntitle: ''\n---\nТекст без заголовков\n", encoding="utf-8")
    assert load_document(path).title == "d1"


@pytest.mark.parametrize("header", ["title: Без id", "id: ''", "id:   "])
def test_load_document_requires_id(tmp_path: Path, header: str) -> None:
    path = tmp_path / "doc.md"
    path.write_text(f"---\n{header}\n---\nТекст\n", encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_document(path)


def test_load_document_without_front_matter_raises(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Документ без шапки\n", encoding="utf-8")
    with pytest.raises(ValueError, match="doc.md"):  # в сообщении есть путь к файлу
        load_document(path)


def test_load_document_numeric_id_becomes_string(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: 42\ntitle: T\n---\nТекст\n", encoding="utf-8")
    assert load_document(path).doc_id == "42"


def test_load_document_normalizes_crlf_and_bom(tmp_path: Path) -> None:
    # Файл, сохранённый в Блокноте Windows: BOM + переводы строк \r\n.
    path = tmp_path / "doc.md"
    path.write_bytes("﻿---\r\nid: d1\r\ntitle: T\r\n---\r\n# T\r\n\r\nТекст.\r\n".encode())
    doc = load_document(path)
    assert doc.doc_id == "d1"
    assert doc.text == "# T\n\nТекст.\n"


def test_load_document_source_path_outside_root(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("---\nid: d1\n---\nТекст\n", encoding="utf-8")
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    assert load_document(path, root=other_root).metadata["source_path"] == path.as_posix()


# ---------------------------------------------------------------- load_corpus


def test_load_corpus_loads_mini_corpus(mini_corpus_dir: Path, mini_docs) -> None:
    docs = load_corpus(mini_corpus_dir)
    assert [doc.doc_id for doc in docs] == sorted(doc.doc_id for doc in mini_docs)
    for loaded, expected in zip(docs, mini_docs, strict=True):
        assert (loaded.doc_id, loaded.title, loaded.text) == (expected.doc_id, expected.title, expected.text)


def test_load_corpus_accepts_str_path(mini_corpus_dir: Path) -> None:
    assert len(load_corpus(str(mini_corpus_dir))) == 4


def test_load_corpus_skips_service_files_and_other_extensions(mini_corpus_dir: Path) -> None:
    (mini_corpus_dir / "README.md").write_text("# Описание корпуса\n", encoding="utf-8")
    (mini_corpus_dir / "hr" / "_template.md").write_text("---\nid: template\n---\nШаблон\n", encoding="utf-8")
    (mini_corpus_dir / "it" / "notes.txt").write_text("---\nid: notes\n---\nНе markdown\n", encoding="utf-8")
    ids = [doc.doc_id for doc in load_corpus(mini_corpus_dir)]
    assert ids == ["hr-remote-2023", "hr-remote-2025", "hr-vacation", "it-vpn"]


def test_load_corpus_finds_nested_directories(tmp_path: Path) -> None:
    deep = tmp_path / "corpus" / "eng" / "runbooks"
    deep.mkdir(parents=True)
    (deep / "rb.md").write_text("---\nid: eng-rb\ntitle: Runbook\n---\nТекст\n", encoding="utf-8")
    (tmp_path / "corpus" / "a.md").write_text("---\nid: a-doc\ntitle: A\n---\nТекст\n", encoding="utf-8")
    docs = load_corpus(tmp_path / "corpus")
    assert [doc.doc_id for doc in docs] == ["a-doc", "eng-rb"]
    assert docs[1].metadata["source_path"] == "eng/runbooks/rb.md"


def test_load_corpus_duplicate_ids_raise(mini_corpus_dir: Path) -> None:
    (mini_corpus_dir / "it" / "copy.md").write_text(
        "---\nid: it-vpn\ntitle: Копия\n---\nТекст\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="it-vpn"):
        load_corpus(mini_corpus_dir)


def test_load_corpus_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "нет-такого-каталога")


def test_load_corpus_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_corpus(tmp_path) == []
