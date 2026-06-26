import pandas as pd
import pytest

from model.utils import parse_is_oos_series, parse_is_oos_value


@pytest.mark.parametrize(
    "raw,expected",
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        ("false", False),
        ("False", False),
        ("no", False),
        ("Нет", False),
        ("0", False),
        ("", False),
        ("true", True),
        ("yes", True),
        ("Да", True),
        ("1", True),
        ("отсутствует", True),
        ("в наличии", False),
    ],
)
def test_parse_is_oos_value(raw, expected: bool):
    assert parse_is_oos_value(raw) is expected


def test_parse_is_oos_series_mixed_strings():
    series = pd.Series(["no", "Да", "0", "отсутствует", False])
    result = parse_is_oos_series(series)
    assert result.tolist() == [False, True, False, True, False]


def test_parse_is_oos_value_unknown_raises():
    with pytest.raises(ValueError, match="is_oos"):
        parse_is_oos_value("maybe")


@pytest.mark.parametrize("is_oos_raw", ["Нет", "no", "0", "false"])
def test_validate_loaded_data_is_oos_false_strings(is_oos_raw: str):
    from unittest.mock import MagicMock, patch

    from ui.data_manager import _validate_loaded_data

    row = {
        "date": "2025-01-01",
        "product_id": 1,
        "product": "Товар",
        "our_price": 100.0,
        "competitor_price": 102.0,
        "sales": 10.0,
        "revenue": 1000.0,
        "cogs": 60.0,
        "profit": 400.0,
        "is_oos": is_oos_raw,
    }
    with patch("ui.data_manager.st") as mock_st:
        result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is not None
    assert result["is_oos"].iloc[0] == False
    mock_st.error.assert_not_called()


@pytest.mark.parametrize("is_oos_raw", ["Да", "yes", "1", "отсутствует"])
def test_validate_loaded_data_is_oos_true_strings(is_oos_raw: str):
    from unittest.mock import MagicMock, patch

    from ui.data_manager import _validate_loaded_data

    row = {
        "date": "2025-01-01",
        "product_id": 1,
        "product": "Товар",
        "our_price": 100.0,
        "competitor_price": 102.0,
        "sales": 10.0,
        "revenue": 1000.0,
        "cogs": 60.0,
        "profit": 400.0,
        "is_oos": is_oos_raw,
    }
    with patch("ui.data_manager.st") as mock_st:
        result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is not None
    assert result["is_oos"].iloc[0] == True
    mock_st.error.assert_not_called()
