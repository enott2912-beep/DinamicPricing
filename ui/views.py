import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from model.pricing import (
    simulate,
    PRODUCTS,
)
from model.analytics import get_recommendations_all_products, get_recommendations_single_product
from ui.calendars import (
    apply_overview_date_filter,
    apply_predict_period_filter,
    get_product_df_with_period,
    render_overview_date_calendar,
    render_predict_period_calendar,
)
from ui.charts import (
    CHART_LABELS,
    CHART_LABELS_TIME,
    plot_price_vs_sales,
    plot_prices_over_time,
    render_stored_simulation,
)
from ui.data_manager import load_predict_data, save_predict_file

RU_COL_MAP = {
    "date": "Дата",
    "store_id": "ID магазина",
    "store": "Магазин",
    "store_profile": "Профиль магазина",
    "brand_id": "ID бренда",
    "brand": "Бренд",
    "product_id": "ID товара",
    "product": "Товар",
    "our_price": "Наша цена, ₽",
    "competitor_1_price": "Цена конкурента 1, ₽",
    "competitor_2_price": "Цена конкурента 2, ₽",
    "competitor_price": "Цена конкурента, ₽",
    "is_oos": "Нет в наличии",
    "sales": "Продажи, шт",
    "revenue": "Выручка, ₽",
    "cogs": "Себестоимость, ₽",
    "profit": "Прибыль, ₽",
    "oos_days": "SKU-дней без наличия",
}


def _ru_table(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает технические имена колонок в русские для UI-таблиц."""
    return df.rename(columns={c: RU_COL_MAP[c] for c in df.columns if c in RU_COL_MAP})


def _disagreement_level(value_pct: float) -> str:
    if value_pct < 10:
        return "низкое"
    if value_pct < 25:
        return "среднее"
    return "высокое"


def _render_model_disagreement_hint(
    *,
    lin_price: float,
    lgbm_price: float,
    lin_growth_pct: float,
    lgbm_growth_pct: float,
    n_rows: int,
    n_days: int,
    unique_prices: int,
    price_std: float,
    nl_warnings: list[str] | None = None,
    context_label: str = "",
) -> None:
    nl_warnings = nl_warnings or []
    price_gap_pct = abs(lin_price - lgbm_price) / max(abs(lin_price), 1.0) * 100
    growth_gap_pp = abs(lin_growth_pct - lgbm_growth_pct)
    level = _disagreement_level(max(price_gap_pct, growth_gap_pp))

    reasons: list[str] = []
    if n_rows < 120:
        reasons.append(f"мало наблюдений ({n_rows})")
    if n_days < 45:
        reasons.append(f"короткий период ({n_days} дн.)")
    if unique_prices < 8:
        reasons.append(f"низкая вариативность цены (уникальных цен: {unique_prices})")
    if price_std < 1.0:
        reasons.append(f"узкий ценовой диапазон (std={price_std:.2f})")
    if nl_warnings:
        reasons.append("сработали диагностические предупреждения LightGBM")
    if not reasons:
        reasons.append("разные допущения моделей (линейная vs нелинейная)")

    ctx = f" ({context_label})" if context_label else ""
    header = (
        f"Расхождение моделей{ctx}: **{level}**. "
        f"Цена: **{price_gap_pct:.1f}%**, прогноз прибыли: **{growth_gap_pp:.1f} п.п.**"
    )
    details = "Вероятные причины: " + "; ".join(reasons) + "."
    advice = (
        "Рекомендация: увеличьте период, проверьте вариативность цен, "
        "при высоком расхождении используйте линейную как baseline."
    )
    if level == "высокое":
        st.warning(f"{header}\n\n{details}\n\n{advice}")
    elif level == "среднее":
        st.info(f"{header}\n\n{details}\n\n{advice}")
    else:
        st.caption(f"{header} {details}")


def _render_welcome_demo_charts() -> None:
    """Иллюстративные графики без загрузки данных (синтетика)."""
    rng = np.random.default_rng(42)
    n = 42
    t = np.linspace(0, 3.3, n)
    price = 70 + 16 * np.sin(t) + rng.normal(0, 2.0, n)
    sales = np.clip(620 - 5.5 * price + rng.normal(0, 32, n), 35, None)
    comp = price * 1.03 + rng.normal(0, 2.2, n)
    dates = pd.date_range(periods=n, freq="D", end=pd.Timestamp("2026-03-15"))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=price, y=sales, mode='markers', name="Точки", marker=dict(color="#2b6fcf", size=8)))
    order = np.argsort(price)
    fig1.add_trace(go.Scatter(x=price[order], y=sales[order], mode='lines', name="Тренд", line=dict(color="#153a7a", width=2), opacity=0.42))
    
    fig1.update_layout(
        title="Обзор: цена и спрос",
        xaxis_title="Наша цена (₽)",
        yaxis_title="Продажи (шт)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        height=300
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=price, mode='lines+markers', name="Наша цена", line=dict(color="#2b6fcf", width=2)))
    fig2.add_trace(go.Scatter(x=dates, y=comp, mode='lines', name="Конкурент", line=dict(color="#c45c12", width=2, dash="dash")))

    fig2.update_layout(
        title="Динамика относительно рынка",
        yaxis_title="Цена (₽)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300
    )

    c_a, c_b = st.columns(2, gap="medium")
    with c_a:
        st.markdown(
            """
            <div class="welcome-demo-wrap">
                <p class="welcome-demo-title">Пример: «цена → продажи»</p>
                <p class="welcome-demo-desc">После загрузки данных здесь будут ваши точки; для «всех товаров»
                графики можно сводить по дням.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig1, width='stretch')
    with c_b:
        st.markdown(
            """
            <div class="welcome-demo-wrap">
                <p class="welcome-demo-title">Пример: цены во времени</p>
                <p class="welcome-demo-desc">В приложении доступен календарь периода и отдельный блок симуляции
                прогноза выручки.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, width='stretch')


def render_welcome_screen() -> None:
    st.markdown(
        """
        <div class="welcome-hero">
            <h1>Динамическое ценообразование</h1>
            <p class="welcome-hero-sub">
                Учебный прототип для оценки ценовых решений: загрузите историю продаж
                (или сгенерируйте демо-данные), выберите режим и получите аналитический
                обзор, рекомендации по цене и сценарный прогноз прибыли.
            </p>
            <div class="welcome-hero-badges">
                <span class="welcome-pill">🧭 2 режима: baseline и experimental</span>
                <span class="welcome-pill">📊 Обзор, рекомендации, симуляция</span>
                <span class="welcome-pill">📁 CSV или генерация данных в 1 клик</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="welcome-section-title">Возможности приложения</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">📊</div>
                <h3>Обзор продаж</h3>
                <p>Метрики по периоду, графики «цена — спрос» и динамика относительно конкурента.
                Удобный календарь, чтобы сузить интервал анализа.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">💡</div>
                <h3>Рекомендации</h3>
                <p>В проверенном режиме используются rules + линейная регрессия,
                в тестовом — нелинейная модель LightGBM с отдельным генератором данных.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">🔮</div>
                <h3>Симуляция</h3>
                <p>Пошаговый прогноз рынка на выбранный горизонт: по одному SKU или сразу по всем товарам,
                с сохранением прогноза в таблицу.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="welcome-section-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-section-title">Режимы работы</p>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">🧭</div>
                <h3>Проверенный режим (baseline)</h3>
                <p>Стабильный сценарий для операционной работы: базовый генератор данных,
                рекомендации на правилах и линейной регрессии, симуляция с методами
                <strong>rules</strong> и <strong>regression</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="welcome-feature-card">
                <div class="wf-icon">🧪</div>
                <h3>Тестовый режим (experimental)</h3>
                <p>Экспериментальный сценарий для проверки нелинейных эффектов: отдельный
                генератор данных, рекомендации и симуляция на модели
                <strong>LightGBM</strong> с диагностикой качества данных.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="welcome-section-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-section-title">Быстрый запуск</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="welcome-timeline-wrap">
            <div class="welcome-timeline">
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">1</span>
                    <div class="welcome-timeline-body">
                        <strong>Режим</strong>
                        <span>В боковой панели выберите <em>проверенный</em> или <em>тестовый</em> режим
                        в зависимости от задачи.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">2</span>
                    <div class="welcome-timeline-body">
                        <strong>Данные</strong>
                        <span>Загрузите CSV с историей или нажмите «Сгенерировать историю»
                        для быстрого старта.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">3</span>
                    <div class="welcome-timeline-body">
                        <strong>Раздел</strong>
                        <span>Откройте «Обзор» для диагностики, «Рекомендации» для новой цены
                        или «Симуляцию» для оценки будущей прибыли.</span>
                    </div>
                </div>
                <div class="welcome-timeline-row">
                    <span class="welcome-timeline-num">4</span>
                    <div class="welcome-timeline-body">
                        <strong>Результат</strong>
                        <span>Сравните эффект по прибыли, при необходимости уточните период и
                        сохраните детализацию прогноза.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="welcome-section-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-section-title">Пример визуализаций</p>', unsafe_allow_html=True)
    _render_welcome_demo_charts()

    with st.expander("📎 Формат CSV и быстрый старт", expanded=False):
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-step">
                    <strong>Способ 1 — свой файл:</strong> колонки
                    <code class="theme-code">date, store_id, store, store_profile, brand_id, brand,
                    product_id, product, our_price, competitor_1_price, competitor_2_price,
                    competitor_price, is_oos, sales, revenue, cogs, profit</code>
                </div>
                <div class="welcome-step">
                    <strong>Способ 2 — без файла:</strong> в боковой панели задайте число дней и нажмите
                    <strong>«Сгенерировать историю»</strong> — будет создан учебный набор продаж для выбранного режима.
                </div>
                <div class="welcome-step">
                    <strong>Важно:</strong> при смене режима с <em>проверенного</em> на <em>тестовый</em>
                    (и наоборот) историю нужно сгенерировать заново, чтобы модели работали на согласованных данных.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="welcome-cta">
            <strong>Готовы начать?</strong><br>
            Загрузите данные слева или сгенерируйте историю — разделы «Обзор», «Рекомендации» и «Симуляция»
            откроются автоматически.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(df: pd.DataFrame, selected_product: str) -> None:
    st.title(f"📊 Анализ продаж: {selected_product}")
    prod_df_raw, _ = get_product_df_with_period(df, selected_product)
    if len(prod_df_raw) == 0:
        st.warning("Нет данных по выбранному товару.")
        st.stop()
    if selected_product == "Все товары":
        st.caption(
            "Метрики вверху — по всем строкам SKU. На **точечных и линейных** графиках данные "
            "сведены **по календарным дням**: цены — среднее по SKU, продажи — сумма по SKU."
        )

    available_dates = set(prod_df_raw["date"].dt.date)
    with st.expander("📅 Период анализа (календарь по датам из таблицы)", expanded=True):
        render_overview_date_calendar(available_dates)
        rs = st.session_state.ov_range_start
        re = st.session_state.ov_range_end
        if re is not None:
            st.success(f"Выбран период: **{rs}** — **{re}**")
        elif st.session_state.ov_pending_second:
            st.caption(f"Показаны данные с **{rs}** до последней даты в таблице (ожидается выбор конца периода).")

    prod_df = apply_overview_date_filter(prod_df_raw)
    if len(prod_df) == 0:
        st.warning("За выбранный период нет строк. Расширьте диапазон или сбросьте период.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Средняя цена", f"{prod_df['our_price'].mean():.2f} ₽")
    c2.metric("Общие продажи", f"{prod_df['sales'].sum():,.0f} шт")
    c3.metric("Прибыль", f"{prod_df['profit'].sum():,.0f} ₽")

    if selected_product == "Все товары" and "store" in prod_df.columns:
        st.subheader("Рейтинг магазинов по прибыли")
        agg_map = {"profit": "sum", "revenue": "sum", "sales": "sum"}
        if "is_oos" in prod_df.columns:
            agg_map["is_oos"] = "sum"
        top_stores = prod_df.groupby("store", as_index=False).agg(agg_map).sort_values("profit", ascending=False)
        if "is_oos" in top_stores.columns:
            top_stores = top_stores.rename(columns={"is_oos": "oos_days"})
        top_stores["profit"] = top_stores["profit"].round(2)
        top_stores["revenue"] = top_stores["revenue"].round(2)
        st.dataframe(_ru_table(top_stores), width='stretch', hide_index=True)

    st.subheader("Зависимость продаж от нашей цены")
    kind1 = st.selectbox("Тип графика", CHART_LABELS, key="ov_chart_price_sales")
    fig1 = plot_price_vs_sales(prod_df, kind1, aggregate_daily=(selected_product == "Все товары"))
    st.plotly_chart(fig1, width='stretch')

    st.subheader("Динамика нашей цены и цены конкурента")
    kind2 = st.selectbox("Тип графика", CHART_LABELS_TIME, key="ov_chart_time")
    fig2 = plot_prices_over_time(prod_df, kind2, aggregate_daily=(selected_product == "Все товары"))
    st.plotly_chart(fig2, width='stretch')


def render_recommendations_tab(df: pd.DataFrame, selected_product: str, app_mode: str = "baseline") -> None:
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
        st.caption(f"Период расчета рекомендаций: с **{rs}** до последней даты в таблице.")
    mode_experimental = app_mode == "experimental"
    enable_lgbm = False
    if not mode_experimental:
        st.caption("Проверенный режим: используются только эвристика и линейная регрессия.")

    if selected_product == "Все товары":
        work_df = (
            prod_df.groupby("date", as_index=False)
            .agg(
                {
                    "our_price": "mean",
                    "competitor_price": "mean",
                    "sales": "sum",
                    "profit": "sum",
                    "cogs": "mean",
                }
            )
            .sort_values("date")
        )
        st.caption(
            "Для «всех товаров» показатели за день: **продажи и прибыль — сумма по SKU**, "
            "**цены — среднее по SKU**. Регрессия по цене и симуляция с методом regression "
            "считаются **отдельно по каждому SKU** (как в симуляции); ниже — сводка и агрегированный график."
        )
    else:
        work_df = prod_df.sort_values("date")

    last_row = work_df.iloc[-1]

    if mode_experimental:
        st.caption("Тестовый режим: показывается только нелинейная модель (LightGBM).")
        if selected_product == "Все товары":
            res = get_recommendations_all_products(prod_df, work_df, include_lightgbm=True)
            if not res:
                st.warning("В данных нет ни одного товара из каталога.")
                st.stop()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Текущее состояние (день)")
                st.metric("Средняя наша цена", f"{last_row['our_price']:.2f} ₽")
                st.metric("Средняя цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
                st.metric("Суммарная прибыль", f"{last_row['profit']:.2f} ₽")
            with c2:
                st.subheader("LightGBM")
                st.metric(
                    "Средняя цена (LGBM)",
                    f"{res['mean_opt_nl']:.2f} ₽",
                    f"{res['growth_nl_pct']:+.1f}% к сумме прибыли",
                )
                if not res.get("all_nl_rel", False):
                    st.warning("У части SKU прогноз может быть неточным (см. колонку «Диагностика»).")
            st.subheader("LightGBM по SKU")
            st.dataframe(pd.DataFrame(res['nl_rows']), width='stretch', hide_index=True)
            st.info(
                f"👉 **Итог по портфелю (LightGBM)**: средняя рекомендованная цена {res['mean_opt_nl']:.2f} ₽, "
                f"прогноз изменения прибыли {res['growth_nl_pct']:+.1f}%."
            )
            return

        res = get_recommendations_single_product(prod_df, work_df, selected_product, include_lightgbm=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Текущее состояние")
            st.metric("Наша цена", f"{res['last_row']['our_price']:.2f} ₽")
            st.metric("Цена конкурента", f"{res['last_row']['competitor_price']:.2f} ₽")
            st.metric("Прибыль (день)", f"{res['last_row']['profit']:.2f} ₽")
        with c2:
            st.subheader("LightGBM")
            st.metric("Новая цена", f"{res['opt_price_nl']:.2f} ₽", f"{res['fc_nl']['growth_pct']:+.1f}% прибыли")
            if res.get("nl_warnings"):
                st.warning("Возможная неточность: " + " ".join(res["nl_warnings"]))
            else:
                st.caption("Качество данных достаточно для нелинейной модели.")
        compare_rows = pd.DataFrame([
            {"Источник": "Текущая", "Цена, ₽": round(float(res['last_row']['our_price']), 2)},
            {"Источник": "LightGBM", "Цена, ₽": round(float(res["opt_price_nl"]), 2)},
        ])
        st.dataframe(compare_rows, width="stretch", hide_index=True)
        st.info(
            f"👉 **Итог по {selected_product} (LightGBM)**: старая цена {res['last_row']['our_price']:.2f} ₽ → "
            f"новая цена {res['opt_price_nl']:.2f} ₽ | прогноз прибыли: {res['fc_nl']['growth_pct']:+.1f}%"
        )
        return

    if selected_product == "Все товары":
        res = get_recommendations_all_products(prod_df, work_df, include_lightgbm=enable_lgbm)
        if not res:
            st.warning("В данных нет ни одного товара из каталога.")
            st.stop()

        if enable_lgbm:
            c1, c2, c3, c4 = st.columns(4)
        else:
            c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Текущее состояние (день)")
            st.metric("Средняя наша цена", f"{last_row['our_price']:.2f} ₽")
            st.metric("Средняя цена конкурента", f"{last_row['competitor_price']:.2f} ₽")
            st.metric("Суммарная прибыль", f"{last_row['profit']:.2f} ₽")
        with c2:
            st.subheader("Эвристика")
            st.metric(
                "Средняя цена (правила)",
                f"{res['mean_rec_rules']:.2f} ₽",
                f"{res['growth_rules_pct']:+.1f}% к сумме прибыли",
            )
            st.caption("Правило считается **по каждому SKU**; прогноз — из PRODUCTS, затем сумма по товарам.")
        with c3:
            st.subheader("Регрессия")
            st.metric(
                "Средняя цена (регр.)",
                f"{res['mean_opt_reg']:.2f} ₽",
                f"{res['growth_reg_pct']:+.1f}% к сумме прибыли",
            )
            st.caption(
                "Оптимум **по каждому SKU** на своей истории; "
                "% — относительно суммы фактической прибыли за день."
            )
            if not res['all_rel']:
                st.warning(
                    "У части товаров наклон регрессии не отрицательный — для них оптимальная цена условна "
                    "(см. колонку «Надёжн.» в таблице)."
                )
        if enable_lgbm:
            with c4:
                st.subheader("LightGBM")
                st.metric(
                    "Средняя цена (LGBM)",
                    f"{res['mean_opt_nl']:.2f} ₽",
                    f"{res['growth_nl_pct']:+.1f}% к сумме прибыли",
                )
                st.caption("Нелинейная модель на истории SKU с диагностикой качества данных.")
                if not res.get("all_nl_rel", False):
                    st.warning("У части SKU прогноз может быть неточным (см. колонку «Диагностика»).")

        st.divider()
        if enable_lgbm:
            t1, t2, t3 = st.tabs(["Эвристика по SKU", "Регрессия по SKU", "LightGBM по SKU"])
            with t1:
                st.dataframe(pd.DataFrame(res['rule_rows']), width='stretch', hide_index=True)
            with t2:
                st.dataframe(pd.DataFrame(res['reg_rows']), width='stretch', hide_index=True)
            with t3:
                st.dataframe(pd.DataFrame(res['nl_rows']), width='stretch', hide_index=True)
        else:
            t1, t2 = st.tabs(["Эвристика по SKU", "Регрессия по SKU"])
            with t1:
                st.dataframe(pd.DataFrame(res['rule_rows']), width='stretch', hide_index=True)
            with t2:
                st.dataframe(pd.DataFrame(res['reg_rows']), width='stretch', hide_index=True)

        if enable_lgbm:
            nl_warn_count = 0
            if res.get("nl_rows"):
                nl_warn_count = int(sum(1 for row in res["nl_rows"] if row.get("Диагностика") and row.get("Диагностика") != "OK"))
            _render_model_disagreement_hint(
                lin_price=float(res.get("mean_opt_reg", 0.0)),
                lgbm_price=float(res.get("mean_opt_nl", 0.0)),
                lin_growth_pct=float(res.get("growth_reg_pct", 0.0)),
                lgbm_growth_pct=float(res.get("growth_nl_pct", 0.0)),
                n_rows=int(len(prod_df)),
                n_days=int(prod_df["date"].nunique()) if "date" in prod_df.columns else 0,
                unique_prices=int(prod_df["our_price"].nunique()) if "our_price" in prod_df.columns else 0,
                price_std=float(prod_df["our_price"].std(ddof=0)) if "our_price" in prod_df.columns and len(prod_df) > 1 else 0.0,
                nl_warnings=[f"предупреждений LightGBM: {nl_warn_count}"] if nl_warn_count > 0 else [],
                context_label="портфель",
            )

        st.divider()
        st.subheader("Агрегированная прибыль vs средняя цена портфеля (по дням)")
        st.caption(
            "Кривая строится по дневным точкам: средняя цена — ось X, суммарные продажи — регрессия; "
            "вершина — условный оптимум **для портфельного ряда**, не обязательно совпадает со средним "
            "пер-товарных оптимумов из таблицы."
        )
        anchor = res['opt_agg'] if res['opt_agg'] > 0 else float(last_row["our_price"])
        p_min = max(1.0, anchor * 0.5)
        p_max = max(p_min * 1.01, anchor * 1.5)
        prices = np.linspace(p_min, p_max, 100)
        c_val = float(work_df['cogs'].mean()) if 'cogs' in work_df.columns else 0.0
        profits = (prices - c_val) * (res['a_agg'] - res['b_agg'] * prices)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=profits, mode='lines', name="Прогноз прибыли (агрегат)", line=dict(color='gray', width=2), opacity=0.5))
        
        max_p = float(np.max(profits)) if len(profits) > 0 else 100
        y_range = [0, max_p * 1.1]

        def add_vtrace(fig, x_val, name, color, dash='solid'):
            fig.add_trace(go.Scatter(
                x=[x_val, x_val], y=y_range,
                mode='lines', name=name,
                line=dict(color=color, width=2, dash=dash),
                showlegend=True
            ))

        add_vtrace(fig, last_row["our_price"], f"Средняя сейчас ({last_row['our_price']:.2f})", "#d62728", 'dash')
        add_vtrace(fig, res["mean_rec_rules"], f"Эвристика ({res['mean_rec_rules']:.2f})", "#1f77b4", 'dot')
        add_vtrace(fig, res["mean_opt_reg"], f"Линейная по SKU ({res['mean_opt_reg']:.2f})", "#2ca02c")
        
        if enable_lgbm:
            add_vtrace(fig, res["mean_opt_nl"], f"LightGBM по SKU ({res['mean_opt_nl']:.2f})", "#9467bd", 'dashdot')
            
        if res['rel_agg'] and res['b_agg'] > 0:
            add_vtrace(fig, res['opt_agg'], f"Оптимум агрег. ({res['opt_agg']:.2f})", "#111111", 'longdash')

        fig.update_layout(
            xaxis_title="Средняя цена портфеля (₽)",
            yaxis_title="Прогнозируемая прибыль (модель по дням)",
            yaxis_range=y_range,
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, width='stretch')

        compare_portfolio_rows = [
            {"Источник": "Средняя текущая", "Цена, ₽": round(float(last_row["our_price"]), 2)},
            {"Источник": "Эвристика (средняя по SKU)", "Цена, ₽": round(float(res["mean_rec_rules"]), 2)},
            {"Источник": "Линейная (средняя по SKU)", "Цена, ₽": round(float(res["mean_opt_reg"]), 2)},
        ]
        if enable_lgbm:
            compare_portfolio_rows.append({"Источник": "LightGBM (средняя по SKU)", "Цена, ₽": round(float(res["mean_opt_nl"]), 2)})
        compare_portfolio_rows.append({"Источник": "Оптимум агрегированного ряда", "Цена, ₽": round(float(res["opt_agg"]), 2)})
        compare_portfolio = pd.DataFrame(compare_portfolio_rows)
        st.caption("Сводные цены по методам для режима «Все товары».")
        st.dataframe(compare_portfolio, width="stretch", hide_index=True)

        st.info(
            f"👉 **Итог по портфелю**: суммарная прибыль последнего дня {res['total_profit_actual']:.2f} ₽ → "
            f"суммарный прогноз по регрессии {res['total_pred_reg']:.2f} ₽ (**{res['growth_reg_pct']:+.1f}%**); "
            f"средняя рекомендованная цена по SKU {res['mean_opt_reg']:.2f} ₽."
        )
        return

    res = get_recommendations_single_product(prod_df, work_df, selected_product, include_lightgbm=enable_lgbm)

    if enable_lgbm:
        c1, c2, c3, c4 = st.columns(4)
    else:
        c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Текущее состояние")
        st.metric("Наша цена", f"{res['last_row']['our_price']:.2f} ₽")
        st.metric("Цена конкурента", f"{res['last_row']['competitor_price']:.2f} ₽")
        st.metric("Прибыль (день)", f"{res['last_row']['profit']:.2f} ₽")
    with c2:
        st.subheader("Эвристика")
        st.metric("Новая цена", f"{res['rec_price_rules']:.2f} ₽", f"{res['fc_rules']['growth_pct']:+.1f}% прибыли")
        st.caption(f"Правило: {res['rule_name']}")
    with c3:
        st.subheader("Регрессия")
        st.metric("Новая цена", f"{res['opt_price_reg']:.2f} ₽", f"{res['fc_reg']['growth_pct']:+.1f}% прибыли")
        st.caption(f"Формула: Profit = (P - C) * ({res['a']:.1f} - {res['b']:.2f}*P)")
        if not res['is_reg_reliable']:
            st.warning(
                "Наклон регрессии не отрицательный: оценка эластичности ненадежна, "
                "поэтому оптимальная цена может быть неточной."
            )
    if enable_lgbm:
        with c4:
            st.subheader("LightGBM")
            st.metric("Новая цена", f"{res['opt_price_nl']:.2f} ₽", f"{res['fc_nl']['growth_pct']:+.1f}% прибыли")
            if res.get("nl_warnings"):
                st.warning("Возможная неточность: " + " ".join(res["nl_warnings"]))
            else:
                st.caption("Качество данных достаточно для нелинейной модели.")

    if enable_lgbm:
        _render_model_disagreement_hint(
            lin_price=float(res["opt_price_reg"]),
            lgbm_price=float(res["opt_price_nl"]),
            lin_growth_pct=float(res["fc_reg"]["growth_pct"]),
            lgbm_growth_pct=float(res["fc_nl"]["growth_pct"]),
            n_rows=int(len(work_df)),
            n_days=int(work_df["date"].nunique()) if "date" in work_df.columns else 0,
            unique_prices=int(work_df["our_price"].nunique()) if "our_price" in work_df.columns else 0,
            price_std=float(work_df["our_price"].std(ddof=0)) if "our_price" in work_df.columns and len(work_df) > 1 else 0.0,
            nl_warnings=res.get("nl_warnings", []),
            context_label=selected_product,
        )

    st.divider()
    st.subheader("Сравнение рекомендованных цен")
    anchor = res['opt_price_reg'] if res['opt_price_reg'] > 0 else float(res['last_row']["our_price"])
    p_min = max(1.0, anchor * 0.5)
    p_max = max(p_min * 1.01, anchor * 1.5)
    prices = np.linspace(p_min, p_max, 100)

    comp_cogs = float(res['last_row'].get('cogs', PRODUCTS.get(selected_product, {}).get('cogs', 0.0)))
    profits = (prices - comp_cogs) * (res['a'] - res['b'] * prices)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=profits, mode='lines', name="Кривая прибыли (линейная модель)", line=dict(color='#8a8a8a', width=1.8), opacity=0.65))

    max_p = float(np.max(profits)) if len(profits) > 0 else 100
    y_range = [0, max_p * 1.1]

    def add_vtrace(fig, x_val, name, color, dash='solid'):
        fig.add_trace(go.Scatter(
            x=[x_val, x_val], y=y_range,
            mode='lines', name=name,
            line=dict(color=color, width=2, dash=dash),
            showlegend=True
        ))

    cur_price = res['last_row']['our_price']
    add_vtrace(fig, cur_price, f"Текущая ({cur_price:.2f})", "#d62728", 'dash')
    add_vtrace(fig, res['rec_price_rules'], f"Эвристика ({res['rec_price_rules']:.2f})", "#1f77b4", 'dot')
    add_vtrace(fig, res['opt_price_reg'], f"Оптимум линейной ({res['opt_price_reg']:.2f})", "#111111", 'longdash')

    if enable_lgbm:
        add_vtrace(fig, res['opt_price_nl'], f"LightGBM ({res['opt_price_nl']:.2f})", "#9467bd", 'dashdot')

    fig.update_layout(
        xaxis_title="Цена (₽)",
        yaxis_title="Прогнозируемая прибыль",
        yaxis_range=y_range,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, width='stretch')

    compare_rows_data = [
        {"Источник": "Текущая", "Цена, ₽": round(float(cur_price), 2)},
        {"Источник": "Эвристика", "Цена, ₽": round(float(res["rec_price_rules"]), 2)},
        {"Источник": "Линейная", "Цена, ₽": round(float(res["opt_price_reg"]), 2)},
    ]
    if enable_lgbm:
        compare_rows_data.append({"Источник": "LightGBM", "Цена, ₽": round(float(res["opt_price_nl"]), 2)})
    compare_rows = pd.DataFrame(compare_rows_data)
    st.caption(
        "Если линии визуально сливаются, это значит, что модели дали близкие цены. "
        "Точные значения — в таблице ниже."
    )
    st.dataframe(compare_rows, width="stretch", hide_index=True)

    st.info(
        f"👉 **Итог по {selected_product}**: старая цена {res['last_row']['our_price']:.2f} ₽ → "
        f"новая цена {res['opt_price_reg']:.2f} ₽ | прогноз прибыли: {res['fc_reg']['growth_pct']:+.1f}%"
    )


def render_simulation_tab(df: pd.DataFrame, selected_product: str, app_mode: str = "baseline") -> None:
    st.title("🔮 Симуляция будущего")
    st.caption(
        f"Товар для прогноза: **{selected_product}** — выбирается только в боковой панели «Выберите товар»."
    )
    rs = st.session_state.get("ov_range_start")
    re = st.session_state.get("ov_range_end")
    if rs:
        period_text = f"**{rs}** — **{re if re else '...'}**"
        st.info(
            f"💡 Модели будут обучаться на историческом периоде: {period_text}. "
            "Вы можете изменить его в календаре на вкладке '📊 Обзор'."
        )

    _, prod_df_period = get_product_df_with_period(df, selected_product)
    available_days = int(prod_df_period["date"].nunique()) if not prod_df_period.empty else 0
    price_unique = (
        int(prod_df_period["our_price"].nunique())
        if "our_price" in prod_df_period.columns and not prod_df_period.empty
        else 0
    )
    price_std = (
        float(prod_df_period["our_price"].std(ddof=0))
        if "our_price" in prod_df_period.columns and len(prod_df_period) > 1
        else 0.0
    )

    col1, col2 = st.columns(2)
    n_steps = col1.slider("Горизонт симуляции (дней)", 7, 30, 14)
    if app_mode == "experimental":
        method_options = ["lightgbm"]
    else:
        method_options = ["regression", "rules"]
    method = col2.selectbox("Метод принятия решений", method_options)
    retrain_every_days = 7
    train_window_days = 90
    max_daily_price_change_pct = 2.0
    if method in ("regression", "lightgbm"):
        st.caption(
            "Для выбранной ML-модели включено переобучение по скользящему окну и ограничение дневного шага цены."
        )
        if method == "lightgbm" and available_days < 60:
            st.warning(
                f"История короткая ({available_days} дн.): прогноз LightGBM может быть неточным. "
                "Рекомендуется расширить период и повысить вариативность цен."
            )
        if method == "lightgbm" and (price_unique < 8 or price_std < 1.0):
            st.warning(
                "⚠️ Прогноз может быть недостаточно точным: выявлена низкая вариативность данных "
                f"(уникальных цен: {price_unique}, std цены: {price_std:.2f}). "
                "Для LightGBM это снижает устойчивость и качество прогноза."
            )
        cfg1, cfg2, cfg3 = st.columns(3)
        retrain_every_days = cfg1.slider("Переобучать каждые N дней", 1, 30, 7)
        if available_days >= 21:
            default_window = min(90, available_days)
            train_window_days = cfg2.slider(
                "Окно обучения (дней)",
                21,
                available_days,
                default_window,
                step=7,
            )
        elif available_days >= 1:
            train_window_days = cfg2.slider(
                "Окно обучения (дней)",
                1,
                available_days,
                available_days,
                step=1,
            )
        else:
            train_window_days = 1
            cfg2.number_input(
                "Окно обучения (дней)",
                min_value=1,
                max_value=1,
                value=1,
                disabled=True,
                help="Нет доступных исторических данных для настройки окна.",
            )
        max_daily_price_change_pct = cfg3.slider("Лимит изменения цены в день (%)", 0.5, 10.0, 2.0, step=0.5)

    if method == "rules":
        from ui.rules_manager import render_rules_manager_inline
        render_rules_manager_inline()

    if st.button("Запустить симуляцию", type="primary"):
        if len(prod_df_period) < 2:
            st.error("⛔ Слишком короткий период для обучения. Выберите диапазон пошире в календаре.")
            st.stop()

        with st.spinner("Рынок просчитывается..."):
            simulated_df = simulate(
                prod_df_period,
                n_steps,
                method,
                target_product=selected_product,
                retrain_every_days=retrain_every_days,
                train_window_days=train_window_days,
                max_daily_price_change_pct=max_daily_price_change_pct,
            )

        if selected_product == "Все товары":
            hist_rev = prod_df_period.groupby("date")["profit"].sum()
            sim_rev = simulated_df.groupby("date")["profit"].sum()
            title_suffix = "по всем товарам"
        else:
            hist_rev = prod_df_period[prod_df_period["product"] == selected_product].groupby("date")[
                "profit"
            ].sum()
            sim_rev = simulated_df[simulated_df["product"] == selected_product].groupby("date")[
                "profit"
            ].sum()
            title_suffix = f"по товару '{selected_product}'"

        max_period_date = prod_df_period["date"].max()
        future_sim = sim_rev[sim_rev.index > max_period_date]
        predict_rows = simulated_df[simulated_df["date"] > max_period_date].copy()
        save_predict_file(predict_rows)

        avg_hist = float(hist_rev.mean()) if len(hist_rev) else float("nan")
        avg_sim = float(future_sim.mean()) if not future_sim.empty else float("nan")
        delta = (
            (avg_sim - avg_hist) / avg_hist * 100
            if avg_hist and not np.isnan(avg_sim) and avg_hist != 0
            else 0.0
        )
        first_day_rev = float(future_sim.iloc[0]) if not future_sim.empty else float("nan")
        last_day_rev = float(future_sim.iloc[-1]) if not future_sim.empty else float("nan")

        st.session_state["sim_last"] = {
            "sim_scope": selected_product,
            "n_steps": n_steps,
            "method": method,
            "hist_rev": hist_rev,
            "future_sim": future_sim,
            "max_period_date": max_period_date,
            "title_suffix": title_suffix,
            "avg_hist": avg_hist,
            "avg_sim": avg_sim,
            "delta": delta,
            "first_day_rev": first_day_rev,
            "last_day_rev": last_day_rev,
            "future_empty": future_sim.empty,
            "retrain_every_days": retrain_every_days,
            "train_window_days": train_window_days,
            "max_daily_price_change_pct": max_daily_price_change_pct,
        }
        st.session_state["sim_show_success"] = True
        st.rerun()

    sl = st.session_state.get("sim_last")
    params_match = (
        sl is not None
        and sl["sim_scope"] == selected_product
        and sl["n_steps"] == n_steps
        and sl["method"] == method
        and sl.get("retrain_every_days", 7) == retrain_every_days
        and sl.get("train_window_days", 90) == train_window_days
        and abs(float(sl.get("max_daily_price_change_pct", 2.0)) - float(max_daily_price_change_pct)) < 1e-9
    )
    if sl and not params_match:
        st.caption(
            "Параметры симуляции изменились относительно последнего запуска — "
            "нажмите «Запустить симуляцию», чтобы обновить график и таблицу."
        )
    if not params_match:
        return

    if st.session_state.pop("sim_show_success", False):
        st.success("✅ Симуляция завершена!")
    render_stored_simulation(sl)

    st.subheader("📋 Детализация прогноза (из predict_sales.csv)")
    pred_df = load_predict_data()
    if pred_df is None or pred_df.empty:
        st.info("Файл predict_sales.csv пока пуст. Запустите симуляцию, чтобы увидеть прогнозные строки.")
        return

    if selected_product != "Все товары":
        pred_df = pred_df[pred_df["product"] == selected_product].copy()
    if pred_df.empty:
        st.info("В predict_sales.csv нет строк для выбранной области симуляции.")
        return

    scope_key = f"{selected_product}|{sl['n_steps']}|{sl['method']}"
    with st.expander("📅 Период отображения прогноза (календарь)", expanded=True):
        render_predict_period_calendar(set(pred_df["date"].dt.date), scope_key=scope_key)
        prs = st.session_state.get("pd_range_start")
        pre = st.session_state.get("pd_range_end")
        if pre is not None:
            st.success(f"Выбран период таблицы: **{prs}** — **{pre}**")
        elif st.session_state.get("pd_pending_second"):
            st.caption(
                f"Показаны строки с **{prs}** до последней даты прогноза "
                "(ожидается второй клик для конца периода)."
            )

    pred_filtered = apply_predict_period_filter(pred_df)
    st.dataframe(_ru_table(pred_filtered), width='stretch')
