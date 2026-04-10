import sys
import calendar
from datetime import date
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError

# Добавляем корень проекта в sys.path для импортов
sys.path.append(str(Path(__file__).parent.parent))
from model.pricing import (
    PRODUCTS,
    SEED,
    apply_rules,
    fit_regression,
    forecast,
    forecast_from_regression,
    simulate,
)
from generator.generate_data import main as run_data_generation

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICT_PATH = DATA_DIR / "predict_sales.csv"

# ==========================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==========================================
st.set_page_config(page_title="Dynamic Pricing MVP", page_icon="💰", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
@st.cache_data
def load_data(uploaded_file=None, use_uploaded: bool = False):
    try:
        if use_uploaded and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            path = DATA_DIR / "sales_history.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
    except EmptyDataError:
        return None

    required_cols = {
        "date",
        "product_id",
        "product",
        "our_price",
        "competitor_price",
        "sales",
        "revenue",
    }
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"В CSV не хватает колонок: {', '.join(sorted(missing))}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        st.error("В CSV есть некорректные значения даты. Исправьте колонку `date`.")
        return None

    # Принудительная сортировка пользовательских и локальных CSV по дате.
    df = df.sort_values(["date", "product"]).reset_index(drop=True)
    return df


def clear_predict_file(columns: list[str] | None = None) -> None:
    """Очищает файл прогнозов (predict_sales.csv)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = columns or [
        "date",
        "product_id",
        "product",
        "our_price",
        "competitor_price",
        "sales",
        "revenue",
    ]
    pd.DataFrame(columns=cols).to_csv(PREDICT_PATH, index=False)
    st.cache_data.clear()


def save_predict_file(df: pd.DataFrame) -> None:
    """Добавляет прогнозные строки в отдельный файл predict_sales.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if out.empty:
        return
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[out["date"].notna()].copy()
    if out.empty:
        return
    out = out.sort_values(["date", "product"]).reset_index(drop=True)

    if PREDICT_PATH.exists():
        try:
            old = pd.read_csv(PREDICT_PATH)
        except EmptyDataError:
            old = pd.DataFrame(columns=out.columns)
    else:
        old = pd.DataFrame(columns=out.columns)

    if not old.empty and "date" in old.columns:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old = old[old["date"].notna()].copy()

    combined = pd.concat([old, out], ignore_index=True)
    combined = combined.sort_values(["date", "product"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined.to_csv(PREDICT_PATH, index=False)
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
    return pred_df.sort_values(["date", "product"]).reset_index(drop=True)


CHART_LABELS = [
    "Точечный",
    "Линейный",
    "Столбчатая диаграмма",
    "Гистограмма",
]
CHART_LABELS_TIME = ["Точечный", "Линейный", "Столбчатая диаграмма", "Гистограмма"]


_MONTHS_RU = (
    "",
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
)


def _init_overview_session(product_key: str, available_dates: set) -> None:
    """Инициализирует состояние календаря при смене товара или первом запуске."""
    min_d, max_d = min(available_dates), max(available_dates)
    
    # Если продукт изменился, запоминаем новый ключ
    if st.session_state.get("ov_product_key") != product_key:
        st.session_state.ov_product_key = product_key
        # Если диапазон уже выбран, пробуем его сохранить (или подрезать под новый товар)
        if "ov_range_start" in st.session_state and st.session_state.ov_range_start is not None:
            # "Подрезаем" текущий выбор под границы нового товара, чтобы избежать ошибок
            st.session_state.ov_range_start = max(min_d, min(max_d, st.session_state.ov_range_start))
            if st.session_state.ov_range_end:
                st.session_state.ov_range_end = max(min_d, min(max_d, st.session_state.ov_range_end))
        else:
            # Если выбора еще нет, ставим на весь доступный период
            st.session_state.ov_range_start = min_d
            st.session_state.ov_range_end = max_d
            st.session_state.ov_pending_second = False

        # Обновляем год/месяц календаря для отображения новой начальной даты
        st.session_state.ov_cal_year = st.session_state.ov_range_start.year
        st.session_state.ov_cal_month = st.session_state.ov_range_start.month


def _on_overview_day_click(clicked: date) -> None:
    """Первый клик — начало периода, второй — конец (даты из таблицы)."""
    if st.session_state.get("ov_pending_second"):
        a, b = st.session_state.ov_range_start, clicked
        if b < a:
            a, b = b, a
        st.session_state.ov_range_start = a
        st.session_state.ov_range_end = b
        st.session_state.ov_pending_second = False
    else:
        st.session_state.ov_range_start = clicked
        st.session_state.ov_range_end = None
        st.session_state.ov_pending_second = True


def _nav_overview_month(delta: int) -> None:
    y, m = st.session_state.ov_cal_year, st.session_state.ov_cal_month
    m += delta
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    st.session_state.ov_cal_year = y
    st.session_state.ov_cal_month = m


def render_overview_date_calendar(available_dates: set) -> None:
    """Календарь: активны только дни, присутствующие в таблице."""
    min_d, max_d = min(available_dates), max(available_dates)
    y, m = st.session_state.ov_cal_year, st.session_state.ov_cal_month
    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    pending = st.session_state.ov_pending_second
    range_end = max_d if (pending and re is None) else re
    st.caption(
        "Выберите **начальную** дату кликом, затем **конечную**. "
        "Доступны только дни, для которых есть строки в данных."
    )
    if st.session_state.ov_pending_second and st.session_state.ov_range_end is None:
        st.info(
            f"Выбрано начало: **{st.session_state.ov_range_start}**. "
            "Кликните конечную дату."
        )

    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if st.button("◀", key="ov_cal_prev", help="Предыдущий месяц"):
            _nav_overview_month(-1)
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center'><b>{_MONTHS_RU[m]} {y}</b></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("▶", key="ov_cal_next", help="Следующий месяц"):
            _nav_overview_month(1)
            st.rerun()

    weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    header = st.columns(7)
    for i, w in enumerate(weekdays):
        header[i].markdown(
            f"<div style='text-align:center;font-size:0.85em'>{w}</div>",
            unsafe_allow_html=True,
        )

    first_weekday, days_in_month = calendar.monthrange(y, m)
    # monthrange: Monday=0
    grid = []
    row = [None] * first_weekday
    for d in range(1, days_in_month + 1):
        day_date = date(y, m, d)
        row.append(day_date)
        if len(row) == 7:
            grid.append(row)
            row = []
    if row:
        while len(row) < 7:
            row.append(None)
        grid.append(row)

    for row in grid:
        cols = st.columns(7)
        for col, day_date in zip(cols, row):
            if day_date is None:
                col.empty()
                continue
            in_data = day_date in available_dates
            if not in_data:
                col.markdown(
                    f"<div style='text-align:center;color:#ccc;padding:6px'>{day_date.day}</div>",
                    unsafe_allow_html=True,
                )
                continue
            label = str(day_date.day)
            is_in_range = (
                rs is not None and range_end is not None and rs <= day_date <= range_end
            )
            col.button(
                label,
                key=f"ovday_{day_date.isoformat()}",
                on_click=_on_overview_day_click,
                args=(day_date,),
                type="primary" if is_in_range else "secondary",
                use_container_width=True,
            )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Сбросить на весь период", key="ov_reset_range"):
            st.session_state.ov_range_start = min_d
            st.session_state.ov_range_end = max_d
            st.session_state.ov_pending_second = False
            st.rerun()
    with b2:
        st.caption(f"В данных: **{min_d}** — **{max_d}**")


def apply_overview_date_filter(prod_df: pd.DataFrame) -> pd.DataFrame:
    """Фильтр по выбранному периоду (пока не выбран конец — от выбранного начала до конца ряда)."""
    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    pending = st.session_state.ov_pending_second
    dcol = prod_df["date"].dt.date
    if pending and re is None:
        return prod_df[dcol >= rs].copy()
    if rs is not None and re is not None:
        return prod_df[(dcol >= rs) & (dcol <= re)].copy()
    return prod_df.copy()


def get_product_df_with_period(df: pd.DataFrame, product: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает пару (исходные_данные, отфильтрованные_по_календарю).
    Если product == "Все товары", возвращается весь датасет.
    """
    if product == "Все товары":
        prod_df_raw = df.sort_values("date")
    else:
        prod_df_raw = df[df["product"] == product].sort_values("date")
        
    if len(prod_df_raw) == 0:
        return prod_df_raw, prod_df_raw
        
    available_dates = set(prod_df_raw["date"].dt.date)
    # Инициализируем сессию календаря (общая для всех вкладок)
    _init_overview_session(product, available_dates)
    prod_df_period = apply_overview_date_filter(prod_df_raw)
    
    return prod_df_raw, prod_df_period


def plot_price_vs_sales(ax, prod_df: pd.DataFrame, kind: str) -> None:
    """График «цена — продажи» в зависимости от типа."""
    x = prod_df["our_price"].values
    y = prod_df["sales"].values
    if len(prod_df) == 0:
        ax.text(
            0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes
        )
        return

    if kind == "Точечный":
        ax.scatter(x, y, alpha=0.6, c="#1f77b4")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
    elif kind == "Линейный":
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker=".", alpha=0.8, color="#1f77b4")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Продажи (шт)")
    elif kind == "Столбчатая диаграмма":
        # Чтобы цифры не налазили друг на друга, группируем цены в бины (шаг побольше)
        # Если уникальных цен много (> 10), используем 8 корзин. Иначе - как есть.
        unique_prices = np.sort(prod_df["our_price"].unique())
        if len(unique_prices) > 10:
            bins = np.linspace(unique_prices.min(), unique_prices.max(), 9)
            labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins)-1)]
            prod_df['price_bin'] = pd.cut(prod_df['our_price'], bins=bins, labels=labels, include_lowest=True)
            g = prod_df.groupby('price_bin', as_index=False)['sales'].sum()
            ax.bar(g['price_bin'].astype(str), g['sales'], color="#1f77b4", alpha=0.85)
            ax.set_xlabel("Диапазон цены (₽)")
        else:
            g = prod_df.groupby("our_price", as_index=False)["sales"].sum().sort_values("our_price")
            ax.bar(g["our_price"].astype(str), g["sales"], color="#1f77b4", alpha=0.85)
            ax.set_xlabel("Цена (₽)")
        
        ax.set_ylabel("Продажи (шт)")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Гистограмма":
        bins = min(20, max(5, len(prod_df) // 3))
        ax.hist(x, bins=bins, weights=y, color="#1f77b4", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Сумма продаж (вес по дням)")
    
    ax.grid(True, alpha=0.3)


def plot_prices_over_time(ax, prod_df: pd.DataFrame, kind: str) -> None:
    """Динамика цен по датам."""
    if len(prod_df) == 0:
        ax.text(
            0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes
        )
        return
    t = prod_df["date"].values
    p1 = prod_df["our_price"].values
    p2 = prod_df["competitor_price"].values

    if kind == "Точечный":
        ax.scatter(t, p1, label="Наша цена", alpha=0.8, c="#1f77b4")
        ax.scatter(t, p2, label="Цена конкурента", alpha=0.6, c="#ff7f0e", marker="s")
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Линейный":
        ax.plot(t, p1, label="Наша цена", marker=".", alpha=0.8)
        ax.plot(t, p2, label="Цена конкурента", ls="--", alpha=0.6)
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    elif kind == "Столбчатая диаграмма":
        n = len(prod_df)
        
        # Если данных больше 30, агрегируем по периодам
        if n > 30:
            # Копируем DataFrame, чтобы не менять исходный
            df_copy = prod_df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            
            # Определяем период агрегации
            days_range = (df_copy['date'].max() - df_copy['date'].min()).days
            if days_range <= 60:
                # Меньше 2 месяцев — группируем по 3 дня
                freq = '3D'
                freq_name = "по 3 дня"
                label_fmt = '%m-%d'
            elif days_range <= 180:
                # Меньше полугода — по неделям
                freq = 'W'
                freq_name = "по неделям"
                label_fmt = '%m-%d'
            else:
                # Больше полугода — по месяцам
                freq = 'ME'
                freq_name = "по месяцам"
                label_fmt = '%Y-%m'
            
            # Агрегируем: берем среднюю цену за период
            df_agg = df_copy.groupby(pd.Grouper(key='date', freq=freq)).agg({
                'our_price': 'mean',
                'competitor_price': 'mean'
            }).dropna()
            
            if len(df_agg) == 0:
                ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center", transform=ax.transAxes)
                return
            
            x = np.arange(len(df_agg))
            p1_agg = df_agg['our_price'].values
            p2_agg = df_agg['competitor_price'].values
            date_labels = [d.strftime(label_fmt) for d in df_agg.index]
            
            ax.bar(x - 0.2, p1_agg, width=0.4, label="Наша цена (средняя)", alpha=0.8)
            ax.bar(x + 0.2, p2_agg, width=0.4, label="Конкурент (средний)", alpha=0.8)
            
            # Добавляем подпись об агрегации с человеко-читаемым текстом
            ax.set_title(f"Средние цены (сгруппировано {freq_name})", fontsize=9)
            
        else:
            # Данных мало — показываем все столбцы
            x = np.arange(n)
            w = 0.35
            ax.bar(x - w/2, p1, width=w, label="Наша цена", alpha=0.8)
            ax.bar(x + w/2, p2, width=w, label="Конкурент", alpha=0.8)
            date_labels = [pd.Timestamp(ti).strftime("%m-%d") for ti in t]
        
        # Настройка подписей оси X
        ax.set_xticks(x)
        
        # Если меток всё ещё много, прореживаем
        if len(date_labels) > 15:
            step = max(1, len(date_labels) // 12)
            visible_labels = [date_labels[i] if i % step == 0 else "" for i in range(len(date_labels))]
            ax.set_xticklabels(visible_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xticks(list(range(0, len(date_labels), step)))
        else:
            ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9)
        
        ax.set_ylabel("Цена (₽)")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    elif kind == "Гистограмма":
        ax.hist(
            p1,
            bins=min(15, max(5, len(prod_df) // 2)),
            alpha=0.7,
            label="Наша цена",
            color="#1f77b4",
        )
        ax.hist(
            p2,
            bins=min(15, max(5, len(prod_df) // 2)),
            alpha=0.5,
            label="Конкурент",
            color="#ff7f0e",
        )
        ax.set_xlabel("Цена (₽)")
        ax.set_ylabel("Число дней")
        ax.legend()


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Управление")

# Загрузка файла
uploaded_file = st.sidebar.file_uploader(
    "Загрузить свой CSV (sales_history)", type=["csv"]
)
use_uploaded_data = False
if uploaded_file is not None:
    source_mode = st.sidebar.radio(
        "Источник данных для построения",
        ["Сгенерированные данные", "Свой CSV"],
        index=1,
        help="Выберите, на каком наборе строить графики и расчеты.",
    )
    use_uploaded_data = source_mode == "Свой CSV"

# Новая функциональность: Генерация данных из UI
st.sidebar.markdown("---")
st.sidebar.write("Нет своих данных?")
n_generate_days = st.sidebar.number_input(
    "Дней для генерации истории",
    min_value=7,
    max_value=365,
    value=100,
    step=1,
    help="100 дней x 5 товаров = 500 строк в sales_history.csv",
)
if st.sidebar.button("✨ Сгенерировать историю", help="Полностью перезапишет файл sales_history.csv"):
    with st.spinner("Генерация данных..."):
        run_data_generation(int(n_generate_days))
        # Очищаем кэш и перезагружаем страницу
        st.cache_data.clear()
        clear_predict_file()
        st.sidebar.success("История сгенерирована!")
        st.rerun()

st.sidebar.markdown("---")

df = load_data(uploaded_file, use_uploaded=use_uploaded_data)

if df is None:
    st.warning(
        "⚠️ Данные не найдены. Похоже, `data/sales_history.csv` отсутствует и файл не загружен."
    )
    if st.button("Сгенерировать демо-данные"):
        from generator.generate_data import main as gen_main

        gen_main()
        clear_predict_file()
        st.rerun()
    st.stop()

st.sidebar.divider()

# Выбор товара и метода
product_list = sorted(list(df["product"].unique()))
selected_product = st.sidebar.selectbox("Выберите товар", product_list)
nav = st.sidebar.radio(
    "Раздел",
    ["📊 Обзор", "💡 Рекомендации", "🔮 Симуляция"],
)

# ==========================================
# ОСНОВНОЙ КОНТЕНТ
# ==========================================

if nav == "📊 Обзор":
    st.title(f"📊 Анализ продаж: {selected_product}")

    prod_df_raw, _ = get_product_df_with_period(df, selected_product)
    if len(prod_df_raw) == 0:
        st.warning("Нет данных по выбранному товару.")
        st.stop()
    available_dates = set(prod_df_raw["date"].dt.date)
    _init_overview_session(selected_product, available_dates)

    with st.expander(
        "📅 Период анализа (календарь по датам из таблицы)", expanded=True
    ):
        render_overview_date_calendar(available_dates)
        rs = st.session_state.ov_range_start
        re = st.session_state.ov_range_end
        if re is not None:
            st.success(f"Выбран период: **{rs}** — **{re}**")
        elif st.session_state.ov_pending_second:
            st.caption(
                f"Показаны данные с **{rs}** до последней даты в таблице (ожидается выбор конца периода)."
            )

    prod_df = apply_overview_date_filter(prod_df_raw)

    if len(prod_df) == 0:
        st.warning(
            "За выбранный период нет строк. Расширьте диапазон или сбросьте период."
        )
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Средняя цена", f"{prod_df['our_price'].mean():.2f} ₽")
    c2.metric("Общие продажи", f"{prod_df['sales'].sum():,.0f} шт")
    c3.metric("Выручка", f"{prod_df['revenue'].sum():,.0f} ₽")

    st.subheader("Зависимость продаж от нашей цены")
    kind1 = st.selectbox("Тип графика", CHART_LABELS, key="ov_chart_price_sales")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_price_vs_sales(ax, prod_df, kind1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Динамика нашей цены и цены конкурента")
    kind2 = st.selectbox("Тип графика", CHART_LABELS_TIME, key="ov_chart_time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_prices_over_time(ax2, prod_df, kind2)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

elif nav == "💡 Рекомендации":
    st.title(f"💡 Рекомендации по цене: {selected_product}")

    _, prod_df = get_product_df_with_period(df, selected_product)
    if len(prod_df) == 0:
        st.warning("За выбранный период нет данных по товару.")
        st.stop()
    rs = st.session_state.ov_range_start
    re = st.session_state.ov_range_end
    if re is not None:
        st.caption(f"Период расчета рекомендаций: **{rs}** — **{re}**")
    elif st.session_state.ov_pending_second:
        st.caption(
            f"Период расчета рекомендаций: с **{rs}** до последней даты в таблице."
        )

    last_row = prod_df.iloc[-1]
    # Для эвристики берем среднее по предыдущим дням (без текущего),
    # иначе условие "падение спроса" срабатывает заметно реже.
    prev_sales = prod_df["sales"].iloc[:-1]
    avg7 = prev_sales.tail(7).mean() if len(prev_sales) > 0 else last_row["sales"]

    # 1. Эвристика
    rec_price_rules, rule_name = apply_rules(last_row, avg7)
    fc_rules = forecast(selected_product, rec_price_rules, last_row["revenue"])

    # 2. Регрессия
    a, b, opt_price_reg, is_reg_reliable = fit_regression(prod_df, selected_product)
    fc_reg = forecast_from_regression(a, b, opt_price_reg, last_row["revenue"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.metric("Наша цена", f"{last_row['our_price']:.2f} ₽")
        st.metric("Цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
        st.metric("Выручка (день)", f"{last_row['revenue']:.2f} ₽")

    with c2:
        st.subheader("Эвристика")
        st.metric(
            "Новая цена",
            f"{rec_price_rules:.2f} ₽",
            f"{fc_rules['growth_pct']:+.1f}% выручки",
        )
        st.caption(f"Правило: {rule_name}")

    with c3:
        st.subheader("Регрессия")
        st.metric(
            "Новая цена",
            f"{opt_price_reg:.2f} ₽",
            f"{fc_reg['growth_pct']:+.1f}% выручки",
        )
        st.caption(f"Формула: Revenue = P * ({a:.1f} - {b:.2f}*P)")
        if not is_reg_reliable:
            st.warning(
                "Наклон регрессии не отрицательный: оценка эластичности ненадежна, "
                "поэтому оптимальная цена может быть неточной."
            )

    # Визуализация функции выручки
    st.divider()
    st.subheader("Анализ эластичности выручки (Regression)")

    p_min = max(1, opt_price_reg * 0.5)
    p_max = opt_price_reg * 1.5
    prices = np.linspace(p_min, p_max, 100)
    revenues = prices * (a - b * prices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices, revenues, label="Прогноз выручки", color="gray", alpha=0.5)
    ax.axvline(
        last_row["our_price"],
        color="red",
        ls="--",
        label=f"Текущая ({last_row['our_price']:.2f})",
    )
    ax.axvline(
        opt_price_reg, color="green", ls="-", label=f"Оптимальная ({opt_price_reg:.2f})"
    )
    ax.set_xlabel("Цена (₽)")
    ax.set_ylabel("Прогнозируемая выручка")
    ax.legend()
    st.pyplot(fig)

    st.info(
        f"👉 **Итог по {selected_product}**: старая цена {last_row['our_price']:.2f} ₽ → новая цена {opt_price_reg:.2f} ₽ | прогноз выручки: {fc_reg['growth_pct']:+.1f}%"
    )

elif nav == "🔮 Симуляция":
    st.title("🔮 Симуляция будущего")

    # Отображение информации о периоде обучения (из календаря на вкладке Обзор)
    rs = st.session_state.get("ov_range_start")
    re = st.session_state.get("ov_range_end")
    if rs:
        period_text = f"**{rs}** — **{re if re else '...'}**"
        st.info(f"💡 Модели будут обучаться на историческом периоде: {period_text}. "
                "Вы можете изменить его в календаре на вкладке '📊 Обзор'.")
    
    col1, col2, col3 = st.columns(3)
    
    sim_scope_options = ["Все товары"] + sorted(list(df["product"].unique()))
    sim_scope = col1.selectbox("Область симуляции", sim_scope_options)
    
    n_steps = col2.slider("Горизонт симуляции (дней)", 7, 30, 14)
    method = col3.selectbox("Метод принятия решений", ["regression", "rules"])

    if st.button("Запустить симуляцию", type="primary"):
        # Подготовка данных периода (Знания для модели)
        _, prod_df_period = get_product_df_with_period(df, sim_scope)
        
        if len(prod_df_period) < 2:
            st.error("⛔ Слишком короткий период для обучения. Выберите диапазон пошире в календаре.")
            st.stop()

        with st.spinner("Рынок просчитывается..."):
            # Вызов ядра симуляции
            simulated_df = simulate(prod_df_period, n_steps, method, target_product=sim_scope)

        st.success(f"✅ Симуляция завершена!")

        # Визуализация результатов (используем только выбранный период истории)
        if sim_scope == "Все товары":
            hist_rev = prod_df_period.groupby("date")["revenue"].sum()
            sim_rev = simulated_df.groupby("date")["revenue"].sum()
            title_suffix = "по всем товарам"
        else:
            hist_rev = prod_df_period[prod_df_period["product"] == sim_scope].groupby("date")["revenue"].sum()
            sim_rev = simulated_df[simulated_df["product"] == sim_scope].groupby("date")["revenue"].sum()
            title_suffix = f"по товару '{sim_scope}'"
        max_period_date = prod_df_period["date"].max()
        future_sim = sim_rev[sim_rev.index > max_period_date]
        predict_rows = simulated_df[simulated_df["date"] > max_period_date].copy()
        save_predict_file(predict_rows)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_rev.index, hist_rev.values, label="Выбранная история", color="blue", linewidth=1.5)
        if not future_sim.empty:
            ax.plot(
                future_sim.index,
                future_sim.values,
                label=f"Прогноз ({method})",
                color="green",
                ls="--",
                linewidth=2
            )
        ax.axvline(max_period_date, color="black", alpha=0.3, linestyle=':')
        ax.set_ylabel(f"Выручка {title_suffix} (₽)")
        ax.set_title(f"Сценарный прогноз на {n_steps} дней (от {max_period_date.date()})")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

        # Сводные показатели
        avg_hist = hist_rev.mean()
        avg_sim = future_sim.mean()
        delta = (avg_sim - avg_hist) / avg_hist * 100 if avg_hist and not np.isnan(avg_sim) else 0.0
        first_day_rev = float(future_sim.iloc[0]) if not future_sim.empty else np.nan
        last_day_rev = float(future_sim.iloc[-1]) if not future_sim.empty else np.nan

        m1, m2, m3 = st.columns(3)
        if future_sim.empty or np.isnan(avg_sim):
            m1.warning(
                "Не удалось вычислить ожидаемую выручку: в симуляции нет корректных будущих точек "
                "относительно выбранного периода."
            )
            m2.metric("Ожидаемая выручка (1-й день)", "—")
            m3.metric("Ожидаемая выручка (последний день)", "—")
        else:
            m1.metric(
                f"Ожидаемая выручка/день",
                f"{avg_sim:,.0f} ₽",
                f"{delta:+.1f}%",
                help="Среднее значение выручки за период симуляции в сравнении с историческим средним."
            )
            m2.metric(
                "Ожидаемая выручка (1-й день)",
                f"{first_day_rev:,.0f} ₽",
            )
            m3.metric(
                "Ожидаемая выручка (последний день)",
                f"{last_day_rev:,.0f} ₽",
            )
        
        st.subheader("📋 Детализация прогноза (из predict_sales.csv)")
        pred_df = load_predict_data()
        if pred_df is None or pred_df.empty:
            st.info("Файл predict_sales.csv пока пуст. Запустите симуляцию, чтобы увидеть прогнозные строки.")
        else:
            if sim_scope != "Все товары":
                pred_df = pred_df[pred_df["product"] == sim_scope].copy()

            if pred_df.empty:
                st.info("В predict_sales.csv нет строк для выбранной области симуляции.")
            else:
                pred_min = pred_df["date"].min().date()
                pred_max = pred_df["date"].max().date()
                picked = st.date_input(
                    "Период данных прогноза",
                    value=(pred_min, pred_max),
                    min_value=pred_min,
                    max_value=pred_max,
                    key="predict_period_range",
                    help="Выберите начальную и конечную даты для фильтрации predict_sales.csv",
                )
                if isinstance(picked, tuple) and len(picked) == 2:
                    p_start, p_end = picked
                else:
                    p_start, p_end = pred_min, pred_max

                mask = (pred_df["date"].dt.date >= p_start) & (pred_df["date"].dt.date <= p_end)
                pred_filtered = pred_df[mask].copy()
                st.dataframe(pred_filtered, use_container_width=True)

