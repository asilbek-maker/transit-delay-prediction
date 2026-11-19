"""Tests for the Seoul bus collector (no network calls)."""
import csv
import gzip
import json
from datetime import datetime

import pytest

from src.collectors import seoul_bus as b


def _ok(items):
    return json.dumps({"msgHeader": {"headerCd": "0", "headerMsg": "정상적으로 처리되었습니다."},
                       "msgBody": {"itemList": items}})


def test_parse_ok_response():
    rows = b.parse_response(_ok([{"rtNm": "146", "staOrd": "1"}]))
    assert rows == [{"rtNm": "146", "staOrd": "1"}]


def test_parse_no_results_is_empty_not_error():
    body = json.dumps({"msgHeader": {"headerCd": "4", "headerMsg": "결과가 없습니다."},
                       "msgBody": {"itemList": None}})
    assert b.parse_response(body) == []


def test_parse_quota_error_raises():
    with pytest.raises(b.QuotaExceeded):
        b.parse_response("<returnAuthMsg>LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR</returnAuthMsg>")


def test_parse_unauthorized_raises_value_error():
    with pytest.raises(ValueError):
        b.parse_response(json.dumps({"error": "Unauthorized", "message": "등록되지 않은 서비스키"}))


def test_append_gzip_members_read_back_as_one_csv(tmp_path):
    now = datetime(2026, 11, 19, 8, 0, tzinfo=b.KST)
    row = {"rtNm": "146", "staOrd": "61", "stNm": "선릉역", "vehId1": "110058325"}
    path = b.append_rows(tmp_path, [row], now)
    b.append_rows(tmp_path, [row, row], now)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert path.name == "arrivals_2026-11-19.csv.gz"
    assert len(rows) == 3 and rows[2]["stNm"] == "선릉역"
    assert rows[0]["polled_at_kst"].startswith("2026-11-19T08:00:00")


def test_routes_are_all_known_trunk_ids():
    assert set(b.ROUTES.values()) == {"146", "152", "370", "753"}
    assert all(r.startswith("1001") and len(r) == 9 for r in b.ROUTES)


def test_missing_key_with_require_key_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "load_key", lambda: "")
    assert b.main(["--minutes", "1", "--require-key", "--out", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
