import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from generator.generate_data import main as run_data_generation
from ui.calendars import get_product_df_with_period
from ui.data_manager import clear_predict_file, df_fingerprint, load_data
from ui.views import (
    render_overview_tab,
    render_recommendations_tab,
    render_simulation_tab,
    render_welcome_screen,
)


def configure_page() -> None:
    st.set_page_config(page_title="Dynamic Pricing MVP", page_icon="💰", layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] { background-color: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2); padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text-color) !important; }

        .welcome-card {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.35);
            margin-bottom: 1.5rem;
        }
        .welcome-card * {
            color: var(--text-color) !important;
        }

        .welcome-step {
            padding: 12px;
            margin: 12px 0;
            background-color: var(--background-color);
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            color: var(--text-color);
        }
        .welcome-step * {
            color: var(--text-color) !important;
        }

        code.theme-code {
            background-color: rgba(128, 128, 128, 0.2);
            color: var(--text-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.35);
            padding: 2px 5px;
            border-radius: 4px;
        }

        .welcome-hero {
            background: linear-gradient(135deg, #153a7a 0%, #1f4ea8 40%, #2b6fcf 100%);
            padding: 2rem 1.75rem 2.25rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1.75rem;
            box-shadow: 0 12px 40px rgba(15, 40, 90, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .welcome-hero h1 {
            margin: 0 0 0.5rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff !important;
        }
        .welcome-hero .welcome-hero-sub {
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.55;
            color: rgba(255, 255, 255, 0.92) !important;
            max-width: 640px;
            margin-left: auto;
            margin-right: auto;
        }
        .welcome-hero-badges {
            margin-top: 1.25rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
        }
        .welcome-pill {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.18);
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }

        .welcome-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--primary-color);
            color: var(--text-color);
        }

        .welcome-feature-card {
            background: linear-gradient(180deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 14px;
            padding: 1.25rem 1.1rem;
            min-height: 168px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        .welcome-feature-card:hover {
            border-color: rgba(43, 111, 207, 0.45);
            box-shadow: 0 8px 28px rgba(31, 78, 168, 0.12);
        }
        .welcome-feature-card .wf-icon {
            font-size: 1.75rem;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .welcome-feature-card h3 {
            margin: 0 0 0.45rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-color) !important;
        }
        .welcome-feature-card p {
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-color) !important;
            opacity: 0.92;
        }

        .welcome-timeline-wrap {
            background-color: var(--secondary-background-color);
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            padding: 1.35rem 1.25rem 0.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        .welcome-timeline {
            max-width: 760px;
            margin: 0 auto;
        }
        .welcome-timeline-row {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1.15rem;
        }
        .welcome-timeline-num {
            flex-shrink: 0;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 50%;
            background: linear-gradient(135deg, #1f4ea8, #2b6fcf);
            color: #ffffff !important;
            font-weight: 700;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 10px rgba(31, 78, 168, 0.35);
        }
        .welcome-timeline-body {
            flex: 1;
            padding-top: 0.15rem;
        }
        .welcome-timeline-body strong {
            color: var(--text-color) !important;
            font-size: 0.98rem;
        }
        .welcome-timeline-body span {
            display: block;
            margin-top: 0.25rem;
            font-size: 0.88rem;
            line-height: 1.45;
            color: var(--text-color) !important;
            opacity: 0.88;
        }

        .welcome-demo-wrap {
            background: var(--secondary-background-color);
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            padding: 1rem 1rem 0.75rem;
            margin-bottom: 0.25rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        .welcome-demo-wrap .welcome-demo-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            color: var(--text-color) !important;
        }
        .welcome-demo-wrap .welcome-demo-desc {
            font-size: 0.82rem;
            margin: 0 0 0.5rem 0;
            line-height: 1.4;
            color: var(--text-color) !important;
            opacity: 0.85;
        }

        .welcome-cta {
            margin-top: 1.75rem;
            padding: 1.1rem 1.25rem;
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(31, 78, 168, 0.12), rgba(43, 111, 207, 0.08));
            border: 1px dashed rgba(43, 111, 207, 0.45);
            text-align: center;
            font-size: 0.95rem;
            color: var(--text-color) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar():
    st.sidebar.title("⚙️ Управление")
    uploaded_file = st.sidebar.file_uploader("Загрузить свой CSV (sales_history)", type=["csv"])
    use_uploaded_data = False
    if uploaded_file is not None:
        source_mode = st.sidebar.radio(
            "Источник данных для построения",
            ["Сгенерированные данные", "Свой CSV"],
            index=1,
            help="Выберите, на каком наборе строить графики и расчеты.",
        )
        use_uploaded_data = source_mode == "Свой CSV"

    st.sidebar.markdown("---")
    st.sidebar.write("Нет своих данных?")
    n_generate_days = st.sidebar.number_input(
        "Дней для генерации истории",
        min_value=7,
        max_value=365,
        value=100,
        step=1,
        help="История строится от сегодняшней даты назад на выбранное число дней "
        "(в CSV попадут N дней × число товаров). Полностью перезаписывает sales_history.csv.",
    )
    if st.sidebar.button("✨ Сгенерировать историю", help="Полностью перезапишет файл sales_history.csv"):
        with st.spinner("Генерация данных..."):
            run_data_generation(int(n_generate_days))
            st.cache_data.clear()
            clear_predict_file()
            st.sidebar.success("История сгенерирована!")
            st.rerun()

    st.sidebar.markdown("---")
    return uploaded_file, use_uploaded_data


def main() -> None:
    configure_page()
    uploaded_file, use_uploaded_data = build_sidebar()
    df = load_data(uploaded_file, use_uploaded=use_uploaded_data)

    if df is None:
        render_welcome_screen()
        st.stop()

    st.sidebar.divider()
    store_list = sorted(list(df["store"].unique())) if "store" in df.columns else []
    selected_store = st.sidebar.selectbox("Магазин", ["Все магазины"] + store_list) if store_list else "Все магазины"
    df_scope = df.copy()
    if selected_store != "Все магазины":
        df_scope = df_scope[df_scope["store"] == selected_store].copy()

    brand_list = sorted(list(df_scope["brand"].unique())) if "brand" in df_scope.columns else []
    selected_brand = st.sidebar.selectbox("Бренд", ["Все бренды"] + brand_list) if brand_list else "Все бренды"
    if selected_brand != "Все бренды":
        df_scope = df_scope[df_scope["brand"] == selected_brand].copy()

    if df_scope.empty:
        st.warning("По выбранным фильтрам Магазин/Бренд нет данных. Измените фильтры в сайдбаре.")
        st.stop()

    product_list = sorted(list(df_scope["product"].unique()))
    sidebar_products = ["Все товары"] + product_list
    selected_product = st.sidebar.selectbox(
        "Выберите товар",
        sidebar_products,
        help="«Все товары» — суммарная аналитика и агрегированные рекомендации по выбранному периоду.",
    )
    nav = st.sidebar.radio("Раздел", ["📊 Обзор", "💡 Рекомендации", "🔮 Симуляция"])

    get_product_df_with_period(df_scope, selected_product)
    forecast_ctx = (
        df_fingerprint(df_scope),
        use_uploaded_data,
        selected_store,
        selected_brand,
        selected_product,
        str(st.session_state.get("ov_range_start")) if st.session_state.get("ov_range_start") else None,
        str(st.session_state.get("ov_range_end")) if st.session_state.get("ov_range_end") else None,
        bool(st.session_state.get("ov_pending_second")),
    )
    if st.session_state.get("_forecast_context") != forecast_ctx:
        st.session_state._forecast_context = forecast_ctx
        clear_predict_file()
        st.session_state.pop("sim_last", None)

    if nav == "📊 Обзор":
        render_overview_tab(df_scope, selected_product)
    elif nav == "💡 Рекомендации":
        render_recommendations_tab(df_scope, selected_product)
    elif nav == "🔮 Симуляция":
        render_simulation_tab(df_scope, selected_product)
