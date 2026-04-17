import calendar
from datetime import date

import pandas as pd
import streamlit as st

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
    min_d, max_d = min(available_dates), max(available_dates)
    if st.session_state.get("ov_product_key") != product_key:
        st.session_state.ov_product_key = product_key
        if "ov_range_start" in st.session_state and st.session_state.ov_range_start is not None:
            st.session_state.ov_range_start = max(
                min_d, min(max_d, st.session_state.ov_range_start)
            )
            if st.session_state.ov_range_end:
                st.session_state.ov_range_end = max(
                    min_d, min(max_d, st.session_state.ov_range_end)
                )
        else:
            st.session_state.ov_range_start = min_d
            st.session_state.ov_range_end = max_d
            st.session_state.ov_pending_second = False
        st.session_state.ov_cal_year = st.session_state.ov_range_start.year
        st.session_state.ov_cal_month = st.session_state.ov_range_start.month


def _on_overview_day_click(clicked: date) -> None:
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
    if pending and re is None:
        st.info(f"Выбрано начало: **{rs}**. Кликните конечную дату.")

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

    for week_row in grid:
        cols = st.columns(7)
        for col, day_date in zip(cols, week_row):
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
            is_in_range = rs is not None and range_end is not None and rs <= day_date <= range_end
            col.button(
                str(day_date.day),
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
    if product == "Все товары":
        prod_df_raw = df.sort_values("date")
    else:
        prod_df_raw = df[df["product"] == product].sort_values("date")

    if len(prod_df_raw) == 0:
        return prod_df_raw, prod_df_raw

    available_dates = set(prod_df_raw["date"].dt.date)
    _init_overview_session(product, available_dates)
    return prod_df_raw, apply_overview_date_filter(prod_df_raw)


def _init_predict_calendar_session(scope_key: str, available_dates: set) -> None:
    if not available_dates:
        return
    min_d, max_d = min(available_dates), max(available_dates)
    dates_sig = (min_d, max_d, len(available_dates))
    if (
        st.session_state.get("pd_scope_key") != scope_key
        or st.session_state.get("pd_dates_sig") != dates_sig
    ):
        st.session_state.pd_scope_key = scope_key
        st.session_state.pd_dates_sig = dates_sig
        st.session_state.pd_range_start = min_d
        st.session_state.pd_range_end = max_d
        st.session_state.pd_pending_second = False
        st.session_state.pd_cal_year = min_d.year
        st.session_state.pd_cal_month = min_d.month


def _on_predict_day_click(clicked: date) -> None:
    if st.session_state.get("pd_pending_second"):
        a, b = st.session_state.pd_range_start, clicked
        if b < a:
            a, b = b, a
        st.session_state.pd_range_start = a
        st.session_state.pd_range_end = b
        st.session_state.pd_pending_second = False
    else:
        st.session_state.pd_range_start = clicked
        st.session_state.pd_range_end = None
        st.session_state.pd_pending_second = True


def _nav_predict_month(delta: int) -> None:
    y, m = st.session_state.pd_cal_year, st.session_state.pd_cal_month
    m += delta
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    st.session_state.pd_cal_year = y
    st.session_state.pd_cal_month = m


def render_predict_period_calendar(available_dates: set, scope_key: str) -> None:
    min_d, max_d = min(available_dates), max(available_dates)
    _init_predict_calendar_session(scope_key, available_dates)
    y, m = st.session_state.pd_cal_year, st.session_state.pd_cal_month
    rs = st.session_state.get("pd_range_start")
    re = st.session_state.get("pd_range_end")
    pending = st.session_state.get("pd_pending_second", False)
    range_end = max_d if (pending and re is None) else re
    st.caption(
        "Выберите **начало** и **конец** периода кликами по датам (доступны только дни из прогноза)."
    )
    if pending and re is None and rs is not None:
        st.info(f"Начало периода: **{rs}**. Кликните конечную дату.")

    nav1, nav2, nav3 = st.columns([1, 4, 1])
    with nav1:
        if st.button("◀", key="pd_cal_prev", help="Предыдущий месяц"):
            _nav_predict_month(-1)
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center'><b>{_MONTHS_RU[m]} {y}</b></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("▶", key="pd_cal_next", help="Следующий месяц"):
            _nav_predict_month(1)
            st.rerun()

    weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    header = st.columns(7)
    for i, w in enumerate(weekdays):
        header[i].markdown(
            f"<div style='text-align:center;font-size:0.85em'>{w}</div>",
            unsafe_allow_html=True,
        )

    first_weekday, days_in_month = calendar.monthrange(y, m)
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

    for cal_row in grid:
        cols = st.columns(7)
        for col, day_date in zip(cols, cal_row):
            if day_date is None:
                col.empty()
                continue
            if day_date not in available_dates:
                col.markdown(
                    f"<div style='text-align:center;color:#ccc;padding:6px'>{day_date.day}</div>",
                    unsafe_allow_html=True,
                )
                continue
            is_in_range = rs is not None and range_end is not None and rs <= day_date <= range_end
            col.button(
                str(day_date.day),
                key=f"pdday_{day_date.isoformat()}",
                on_click=_on_predict_day_click,
                args=(day_date,),
                type="primary" if is_in_range else "secondary",
                use_container_width=True,
            )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Сбросить на весь период прогноза", key="pd_reset_range"):
            st.session_state.pd_range_start = min_d
            st.session_state.pd_range_end = max_d
            st.session_state.pd_pending_second = False
            st.rerun()
    with b2:
        st.caption(f"В прогнозе: **{min_d}** — **{max_d}**")


def apply_predict_period_filter(pred_df: pd.DataFrame) -> pd.DataFrame:
    rs = st.session_state.get("pd_range_start")
    re = st.session_state.get("pd_range_end")
    pending = st.session_state.get("pd_pending_second", False)
    if len(pred_df) == 0:
        return pred_df
    dcol = pred_df["date"].dt.date
    if pending and re is None and rs is not None:
        return pred_df[dcol >= rs].copy()
    if rs is not None and re is not None:
        return pred_df[(dcol >= rs) & (dcol <= re)].copy()
    return pred_df.copy()
