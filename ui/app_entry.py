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
        .welcome-card { background-color: var(--secondary-background-color); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--primary-color); margin-bottom: 1.5rem; }
        .welcome-header { background: linear-gradient(120deg, var(--primary-color) 0%, #8fd3f4 100%); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem; color: white; }
        .welcome-step { padding: 12px; margin: 12px 0; background-color: var(--background-color); border-radius: 8px; border-left: 4px solid var(--primary-color); color: var(--text-color); }
        code.theme-code { background-color: rgba(128, 128, 128, 0.2); color: var(--primary-color); padding: 2px 5px; border-radius: 4px; }
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
    product_list = sorted(list(df["product"].unique()))
    selected_product = st.sidebar.selectbox("Выберите товар", product_list)
    nav = st.sidebar.radio("Раздел", ["📊 Обзор", "💡 Рекомендации", "🔮 Симуляция"])

    get_product_df_with_period(df, selected_product)
    forecast_ctx = (
        df_fingerprint(df),
        use_uploaded_data,
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
        render_overview_tab(df, selected_product)
    elif nav == "💡 Рекомендации":
        render_recommendations_tab(df, selected_product)
    elif nav == "🔮 Симуляция":
        render_simulation_tab(df)
