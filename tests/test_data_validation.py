from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ui.data_manager import _validate_loaded_data


@pytest.fixture
def valid_row() -> dict:
    return {
        "date": "2025-01-01",
        "product_id": 1,
        "product": "Молоко",
        "our_price": 80.0,
        "competitor_price": 82.0,
        "sales": 100.0,
        "revenue": 8000.0,
        "cogs": 60.0,
        "profit": 2000.0,
    }


@patch("ui.data_manager.st")
def test_validate_loaded_data_valid_minimal(mock_st: MagicMock, valid_row: dict):
    df = pd.DataFrame([valid_row])
    result = _validate_loaded_data(df)
    assert result is not None
    assert len(result) == 1
    mock_st.error.assert_not_called()


@patch("ui.data_manager.st")
def test_validate_loaded_data_missing_column(mock_st: MagicMock, valid_row: dict):
    row = {k: v for k, v in valid_row.items() if k != "profit"}
    result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is None
    mock_st.error.assert_called()


@patch("ui.data_manager.st")
def test_validate_loaded_data_revenue_mismatch(mock_st: MagicMock, valid_row: dict):
    row = dict(valid_row)
    row["revenue"] = 1.0
    result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is None
    mock_st.error.assert_called()


@patch("ui.data_manager.st")
def test_validate_loaded_data_negative_price(mock_st: MagicMock, valid_row: dict):
    row = dict(valid_row)
    row["our_price"] = -1.0
    result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is None
    mock_st.error.assert_called()


@patch("ui.data_manager.st")
def test_validate_loaded_data_bad_date(mock_st: MagicMock, valid_row: dict):
    row = dict(valid_row)
    row["date"] = "not-a-date"
    result = _validate_loaded_data(pd.DataFrame([row]))
    assert result is None
    mock_st.error.assert_called()
