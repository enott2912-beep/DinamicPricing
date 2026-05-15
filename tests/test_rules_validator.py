from model.rules_validator import validate_rule, validate_rules_list


def test_validate_rule_valid_minimal():
    rule = {
        "name": "test_rule",
        "condition": "price > comp_price",
        "action": "price * 0.99",
    }
    ok, msg = validate_rule(rule)
    assert ok is True
    assert msg == ""


def test_validate_rule_missing_name():
    rule = {"condition": "price > 0", "action": "price"}
    ok, msg = validate_rule(rule)
    assert ok is False
    assert "назван" in msg.lower() or "name" in msg.lower() or msg


def test_validate_rule_missing_condition():
    rule = {"name": "x", "action": "price"}
    ok, _ = validate_rule(rule)
    assert ok is False


def test_validate_rule_missing_action():
    rule = {"name": "x", "condition": "price > 0"}
    ok, _ = validate_rule(rule)
    assert ok is False


def test_validate_rule_unknown_variable():
    rule = {
        "name": "bad",
        "condition": "unknown_var > 0",
        "action": "price",
    }
    ok, msg = validate_rule(rule)
    assert ok is False
    assert msg


def test_validate_rule_negative_action_price():
    rule = {
        "name": "bad",
        "condition": "price > 0",
        "action": "-5",
    }
    ok, msg = validate_rule(rule)
    assert ok is False
    assert "отрицатель" in msg.lower() or "negative" in msg.lower() or msg


def test_validate_rules_list_too_many():
    rules = [
        {"name": f"r{i}", "condition": "price > 0", "action": "price"}
        for i in range(51)
    ]
    ok, errors = validate_rules_list(rules)
    assert ok is False
    assert any("50" in e for e in errors)


def test_validate_rules_list_collects_multiple_errors():
    rules = [
        {"name": "", "condition": "price > 0", "action": "price"},
        {"name": "ok", "condition": "bad_var > 0", "action": "price"},
    ]
    ok, errors = validate_rules_list(rules)
    assert ok is False
    assert len(errors) >= 2
