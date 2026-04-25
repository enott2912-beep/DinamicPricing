import atexit
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICT_PATH = DATA_DIR / "predict_sales.csv"
SALES_HISTORY_PATH = DATA_DIR / "sales_history.csv"
GEN_MODE_PATH = DATA_DIR / "generation_mode.txt"

REQUIRED_COLS = {
    "date",
    "product_id",
    "product",
    "our_price",
    "competitor_price",
    "sales",
    "revenue",
    "cogs",
    "profit",
}

OPTIONAL_NUMERIC_COLS = ["competitor_1_price", "competitor_2_price"]
NUMERIC_COLS = ["our_price", "competitor_price", "sales", "revenue", "cogs", "profit"] + OPTIONAL_NUMERIC_COLS


def _cleanup_data_files_on_exit() -> None:
    """Удаляет локальные CSV при завершении процесса Streamlit."""
    try:
        if SALES_HISTORY_PATH.exists():
            SALES_HISTORY_PATH.unlink()
        if PREDICT_PATH.exists():
            PREDICT_PATH.unlink()
    except OSError:
        pass


atexit.register(_cleanup_data_files_on_exit)


def df_fingerprint(df: pd.DataFrame) -> tuple:
    """Компактная сигнатура набора данных для инвалидации прогноза."""
    if df is None or len(df) == 0:
        return (0,)
    sub = df[["date", "product", "sales", "revenue", "profit", "our_price"]].copy()
    h = int(pd.util.hash_pandas_object(sub, index=True).sum())
    return (len(df), str(df["date"].min()), str(df["date"].max()), h)


def _validate_loaded_data(df: pd.DataFrame) -> pd.DataFrame | None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"В CSV не хватает колонок: {', '.join(sorted(missing))}")
        return None

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        st.error("В CSV есть некорректные значения даты. Исправьте колонку `date`.")
        return None

    for col in NUMERIC_COLS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            st.error(f"В колонке `{col}` есть нечисловые или пустые значения.")
            return None

    negative_checks = {
        "our_price": "наша цена",
        "competitor_price": "цена конкурента",
        "sales": "продажи",
        "revenue": "выручка",
        "cogs": "себестоимость",
    }
    for col, label in negative_checks.items():
        if (out[col] < 0).any():
            st.error(f"В CSV обнаружены отрицательные значения: `{label}` ({col}).")
            return None

    if out["product"].astype(str).str.strip().eq("").any():
        st.error("В CSV есть пустые значения в колонке `product`.")
        return None

    expected_revenue = out["sales"] * out["our_price"]
    mismatch_rev = (out["revenue"] - expected_revenue).abs() > 0.01
    if mismatch_rev.any():
        st.error(
            "В CSV есть несогласованные строки: `revenue` не совпадает с `sales * our_price`."
        )
        return None

    expected_profit = out["revenue"] - out["sales"] * out["cogs"]
    mismatch_prof = (out["profit"] - expected_profit).abs() > 0.01
    if mismatch_prof.any():
        st.error(
            "В CSV есть несогласованные строки: `profit` не совпадает с `revenue - sales * cogs`."
        )
        return None

    if "is_oos" in out.columns:
        out["is_oos"] = out["is_oos"].astype(bool)
    sort_cols = [c for c in ["date", "store", "brand", "product"] if c in out.columns]
    if not sort_cols:
        sort_cols = ["date", "product"]
    return out.sort_values(sort_cols).reset_index(drop=True)


@st.cache_data
def load_data(uploaded_file=None, use_uploaded: bool = False):
    try:
        if use_uploaded and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            if not SALES_HISTORY_PATH.exists():
                return None
            df = pd.read_csv(SALES_HISTORY_PATH)
    except EmptyDataError:
        return None

    return _validate_loaded_data(df)


def clear_predict_file(columns: list[str] | None = None) -> None:
    """Очищает файл прогнозов (predict_sales.csv)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = columns or [
        "date",
        "store_id",
        "store",
        "store_profile",
        "brand_id",
        "brand",
        "product_id",
        "product",
        "our_price",
        "competitor_1_price",
        "competitor_2_price",
        "competitor_price",
        "is_oos",
        "sales",
        "revenue",
        "cogs",
        "profit",
    ]
    pd.DataFrame(columns=cols).to_csv(PREDICT_PATH, index=False)
    st.cache_data.clear()


def get_generated_mode() -> str | None:
    """Возвращает режим последней генерации: baseline | experimental."""
    if not GEN_MODE_PATH.exists():
        return None
    try:
        mode = GEN_MODE_PATH.read_text(encoding="utf-8").strip().lower()
        return mode or None
    except OSError:
        return None


def save_predict_file(df: pd.DataFrame) -> None:
    """Полностью перезаписывает predict_sales.csv текущим прогнозом (без слияния)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if out.empty:
        return
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[out["date"].notna()].copy()
    if out.empty:
        return
    sort_cols = [c for c in ["date", "store", "brand", "product"] if c in out.columns]
    if not sort_cols:
        sort_cols = ["date", "product"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(PREDICT_PATH, index=False)
    st.cache_data.clear()


@st.cache_data
def load_predict_data() -> pd.DataFrame | None:
    """Загружает прогнозы из отдельного файла predict_sales.csv."""
    if not PREDICT_PATH.exists():
        return None
    pred_df = pd.read_csv(PREDICT_PATH)
    if pred_df.empty:
        return pred_df
    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
        pred_df = pred_df[pred_df["date"].notna()].copy()
    sort_cols = [c for c in ["date", "store", "brand", "product"] if c in pred_df.columns]
    if not sort_cols:
        sort_cols = ["date", "product"]
    return pred_df.sort_values(sort_cols).reset_index(drop=True)
