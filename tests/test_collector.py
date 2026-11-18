"""Tests for the Seoul subway collector (no network calls)."""
import csv
from datetime import datetime

from src.collectors import seoul_subway as s


def test_service_hours_skip_the_night():
    assert s.in_service(datetime(2026, 11, 18, 8, 0, tzinfo=s.KST))
    assert s.in_service(datetime(2026, 11, 18, 0, 30, tzinfo=s.KST))
    assert not s.in_service(datetime(2026, 11, 18, 3, 0, tzinfo=s.KST))


def test_append_rows_writes_header_once(tmp_path):
    row = {"subwayNm": "2호선", "statnNm": "강남", "trainNo": "2001", "unexpected": "x"}
    now = datetime(2026, 11, 18, 9, 0, tzinfo=s.KST)
    path = s.append_rows(tmp_path, [row], now)
    s.append_rows(tmp_path, [row], now)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert path.name == "positions_2026-11-18.csv"
    assert len(rows) == 2 and rows[0]["statnNm"] == "강남"
    assert rows[0]["polled_at_kst"].startswith("2026-11-18T09:00:00")


def test_key_from_environment_wins(monkeypatch):
    monkeypatch.setenv("SEOUL_SUBWAY_KEY", "abc123")
    assert s.load_key() == "abc123"


def test_require_key_skips_when_only_sample(monkeypatch, tmp_path):
    monkeypatch.setenv("SEOUL_SUBWAY_KEY", "")
    monkeypatch.setattr(s, "load_key", lambda: "sample")
    assert s.main(["--minutes", "1", "--require-key", "--out", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())


def test_every_default_line_has_a_transport_mode(tmp_path):
    assert set(s.MODE_BY_LINE.values()) == {"Metro", "Train", "Light rail"}
    now = datetime(2026, 11, 18, 9, 0, tzinfo=s.KST)
    path = s.append_rows(tmp_path, [{"subwayNm": "수인분당선"}, {"subwayNm": "신림선"}], now)
    with path.open(encoding="utf-8") as fh:
        modes = [r["mode"] for r in csv.DictReader(fh)]
    assert modes == ["Train", "Light rail"]
