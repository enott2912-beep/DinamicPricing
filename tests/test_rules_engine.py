import json
from pathlib import Path
from unittest.mock import patch

import pytest

from model.rules_engine import RuleEngine
from model.rules_validator import validate_rules_list

SAMPLE_RULE = {
    "name": "test_rule",
    "description": "Test",
    "condition": "price > 0",
    "action": "price * 1.01",
}


def test_rule_engine_evaluate_first_matching_rule(rule_engine: RuleEngine):
    price, name = rule_engine.evaluate(
        {
            "price": 100.0,
            "comp_1": 105.0,
            "comp_2": 95.0,
            "comp_price": 95.0,
            "sales": 200.0,
            "avg_sales_7d": 210.0,
            "cogs": 60.0,
        }
    )
    assert name == "always_match"
    assert price == 105.0


def test_rule_engine_evaluate_hold_when_no_match(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json")
    engine.rules = [
        {"name": "never", "condition": "price < 0", "action": "1.0"},
    ]
    price, name = engine.evaluate(
        {
            "price": 88.5,
            "comp_1": 90.0,
            "comp_2": 85.0,
            "comp_price": 85.0,
            "sales": 100.0,
            "avg_sales_7d": 100.0,
            "cogs": 50.0,
        }
    )
    assert name == "hold"
    assert price == 88.5


def test_rule_engine_evaluate_no_rules(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json")
    engine.rules = []
    price, name = engine.evaluate({"price": 77.0})
    assert name == "no_rules"
    assert price == 77.0


def test_rule_engine_margin_available_in_condition(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json")
    engine.rules = [
        {
            "name": "low_margin",
            "condition": "margin < 0.2",
            "action": "price * 1.1",
        },
    ]
    price, name = engine.evaluate(
        {
            "price": 100.0,
            "comp_1": 100.0,
            "comp_2": 100.0,
            "comp_price": 100.0,
            "sales": 50.0,
            "avg_sales_7d": 50.0,
            "cogs": 90.0,
        }
    )
    assert name == "low_margin"
    assert price == 110.0


def test_rule_engine_skips_broken_rule(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json")
    engine.rules = [
        {"name": "broken", "condition": "undefined_xyz > 0", "action": "price"},
        {"name": "ok", "condition": "price > 0", "action": "price + 2"},
    ]
    price, name = engine.evaluate(
        {
            "price": 50.0,
            "comp_1": 50.0,
            "comp_2": 50.0,
            "comp_price": 50.0,
            "sales": 10.0,
            "avg_sales_7d": 10.0,
            "cogs": 30.0,
        }
    )
    assert name == "ok"
    assert price == 52.0


def test_rule_engine_save_rules_rejects_invalid(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json")
    invalid = [{"name": "", "condition": "price > 0", "action": "price"}]
    is_valid, _ = validate_rules_list(invalid)
    assert is_valid is False
    with pytest.raises(ValueError, match="валидац"):
        engine.save_rules(invalid)


def test_save_rules_without_session_writes_file(tmp_path: Path):
    rules_path = tmp_path / "rules.json"
    engine = RuleEngine(rules_path=rules_path)
    engine.save_rules([SAMPLE_RULE])
    saved = json.loads(rules_path.read_text(encoding="utf-8"))
    assert saved[0]["name"] == "test_rule"


def test_save_rules_session_db_failure_does_not_touch_global_file(tmp_path: Path):
    rules_path = tmp_path / "rules.json"
    global_rules = [{"name": "global_default", "condition": "price > 0", "action": "price"}]
    rules_path.write_text(json.dumps(global_rules), encoding="utf-8")

    engine = RuleEngine(rules_path=rules_path, session_id="sess-fail-test")
    user_rules = [{"name": "user_private", "condition": "price > 1", "action": "price * 2"}]

    with patch("ui.session_store.save_rules", side_effect=OSError("db locked")):
        with pytest.raises(RuntimeError, match="базу данных"):
            engine.save_rules(user_rules)

    assert json.loads(rules_path.read_text(encoding="utf-8")) == global_rules


def test_load_rules_session_db_failure_raises(tmp_path: Path):
    engine = RuleEngine(rules_path=tmp_path / "rules.json", session_id="sess-load-fail")

    with patch("ui.session_store.load_rules", side_effect=OSError("db unavailable")):
        with pytest.raises(RuntimeError, match="загрузить правила сессии"):
            engine.load_rules()


def test_save_rules_session_success_uses_db(tmp_path: Path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text("[]", encoding="utf-8")
    session_id = "sess-ok-test"

    from ui import session_store

    session_store.init_db()
    session_store._ensure_session_row(session_id)

    engine = RuleEngine(rules_path=rules_path, session_id=session_id)
    engine.save_rules([SAMPLE_RULE])

    loaded = session_store.load_rules(session_id)
    assert loaded is not None
    assert loaded[0]["name"] == "test_rule"
    assert json.loads(rules_path.read_text(encoding="utf-8")) == []
