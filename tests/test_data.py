"""Tests for data loading, validation, profiling, cleaning, and splitting."""
import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import TourismDataLoader, TourismRecord
from src.data.validators import DatasetValidator
from src.data.cleaners import DataCleaner
from src.data.splitters import DataSplitter


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RECORDS = [
    {"Input": "What are the best beaches in Bali?", "CoT": "Identify top beaches.", "Output": "Kuta, Seminyak, Nusa Dua."},
    {"Input": "Plan a 3-day trip to Seoul.", "CoT": "Plan itinerary.", "Output": "Day 1: Gyeongbokgung. Day 2: Hongdae. Day 3: N Seoul Tower."},
    {"Input": "Best halal restaurants in Bangkok.", "CoT": "Find halal options.", "Output": "Several halal restaurants near Pratunam."},
    {"Input": "What are the best beaches in Bali?", "CoT": "Identify top beaches.", "Output": "Kuta, Seminyak, Nusa Dua."},  # duplicate
]


@pytest.fixture
def sample_json_file(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(SAMPLE_RECORDS), encoding="utf-8")
    return p


@pytest.fixture
def records(sample_json_file):
    loader = TourismDataLoader(sample_json_file)
    return loader.load()


# ── Loader Tests ──────────────────────────────────────────────────────────────

def test_loader_count(records):
    assert len(records) == 4


def test_loader_types(records):
    for r in records:
        assert isinstance(r, TourismRecord)


def test_loader_fields(records):
    r = records[0]
    assert r.input_text == "What are the best beaches in Bali?"
    assert r.cot == "Identify top beaches."
    assert r.output_text == "Kuta, Seminyak, Nusa Dua."
    assert r.record_id == 0


def test_loader_iter_batches(records, sample_json_file):
    loader = TourismDataLoader(sample_json_file)
    loader.load()
    batches = list(loader.iter_batches(batch_size=2))
    assert len(batches) == 2
    assert all(len(b) <= 2 for b in batches)


# ── Validator Tests ───────────────────────────────────────────────────────────

def test_validator_all_valid(records):
    v = DatasetValidator()
    result = v.validate_all(records)
    assert result.total == 4
    assert result.valid == 4
    assert len(result.issues) == 0


def test_validator_empty_input():
    rec = TourismRecord(0, "", "some cot", "some output")
    v = DatasetValidator()
    issues = v.validate_record(rec)
    assert "EMPTY_INPUT" in issues


def test_validator_short_output():
    rec = TourismRecord(0, "Valid input query here", "cot", "short")
    v = DatasetValidator(min_output_len=20)
    issues = v.validate_record(rec)
    assert any("OUTPUT_TOO_SHORT" in i for i in issues)


# ── Cleaner Tests ─────────────────────────────────────────────────────────────

def test_cleaner_normalizes_whitespace(records):
    dirty = TourismRecord(0, "  Hello   world  ", "  cot  ", "  output  ")
    cleaner = DataCleaner()
    cleaned = cleaner.clean_record(dirty)
    assert cleaned.input_text == "Hello world"
    assert cleaned.cot == "cot"
    assert cleaned.output_text == "output"


def test_cleaner_dedup(records):
    cleaner = DataCleaner()
    unique, removed = cleaner.deduplicate(records)
    assert len(unique) == 3  # 4 records, 1 exact duplicate
    assert len(removed) == 1


def test_dedup_removes_exact_dupe(records):
    cleaner = DataCleaner()
    unique, removed = cleaner.deduplicate(records)
    removed_ids = {r["record_id"] for r in removed}
    assert 3 in removed_ids  # 4th record (index 3) is duplicate of 0


# ── Splitter Tests ────────────────────────────────────────────────────────────

def test_splitter_ratios():
    records = [TourismRecord(i, f"input {i}", "cot", f"output {i}") for i in range(100)]
    splitter = DataSplitter(0.8, 0.1, 0.1, seed=42)
    train, val, test = splitter.split(records)
    assert len(train) + len(val) + len(test) == 100
    assert abs(len(train) - 80) <= 1
    assert abs(len(val) - 10) <= 1


def test_splitter_deterministic():
    records = [TourismRecord(i, f"input {i}", "cot", f"output {i}") for i in range(50)]
    splitter = DataSplitter(seed=42)
    t1, v1, te1 = splitter.split(records)
    t2, v2, te2 = splitter.split(records)
    assert [r.record_id for r in t1] == [r.record_id for r in t2]


def test_splitter_save(tmp_path):
    records = [TourismRecord(i, f"input {i}", "cot", f"output {i}") for i in range(20)]
    splitter = DataSplitter(seed=42)
    train, val, test = splitter.split(records)
    paths = splitter.save_splits(train, val, test, output_dir=tmp_path)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "split_manifest.json").exists()
