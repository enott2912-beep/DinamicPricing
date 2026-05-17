import json
import time

import streamlit as st
from model.rules_engine import get_rule_engine


def _save_rules_safe(engine, rules: list[dict]) -> bool:
    try:
        engine.save_rules(rules)
        return True
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return False


def render_rules_manager_inline():
    """
    Встроенный компонент управления правилами.
    Вызывается внутри вкладки настройки симуляции.
    """
    engine = get_rule_engine()
    st.markdown("---")
    st.subheader("Настройки Rule-Engine")
    
    t1, t2, t3 = st.tabs(["📋 Действующие правила", "➕ Добавить правило", "🕒 История правил"])
    
    with t1:
        st.write("Правила применяются сверху вниз (какое правило истинно первым, то и побеждает).")
        rules = engine.rules
        if not rules:
            st.info("Сейчас нет действующих правил.")
        else:
            for i, rule in enumerate(rules):
                name = rule.get('name', 'Без имени')
                with st.expander(f"{i+1}. {name}", expanded=False):
                    st.write(f"**Описание:** {rule.get('description', '')}")
                    st.info(f"**Условие:** `{rule.get('condition', '')}`")
                    st.success(f"**Действие:** `{rule.get('action', '')}`")
                    
                    if st.button("🗑 Удалить правило", key=f"del_rule_{i}"):
                        rules.pop(i)
                        if _save_rules_safe(engine, rules):
                            st.success("Правило удалено!")
                            time.sleep(0.5)
                            st.rerun()

    with t2:
        st.write("Заполните форму для добавления нового правила (оно добавится в конец списка).")
        st.caption(
            "Доступные переменные: `price`, `comp_1`, `comp_2`, `comp_price`, "
            "`sales`, `avg_sales_7d`, `cogs`, `margin`."
        )
        with st.form("add_rule_form"):
            r_name = st.text_input("Название (англ. без пробелов)", "new_rule")
            r_desc = st.text_input("Понятное описание", "Описание правила...")
            r_cond = st.text_input("Условие (например: margin < 0.1 and sales < 100)", "margin < 0.1")
            r_act = st.text_input("Действие (например: price * 1.05)", "price * 1.05")
            
            submitted = st.form_submit_button("Добавить правило")
            if submitted:
                if r_name and r_cond and r_act:
                    new_rule = {
                        "name": r_name,
                        "description": r_desc,
                        "condition": r_cond,
                        "action": r_act
                    }
                    rules_copy = list(engine.rules)
                    rules_copy.append(new_rule)
                    if _save_rules_safe(engine, rules_copy):
                        st.success("Правило успешно добавлено!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.error("Поля 'Название', 'Условие' и 'Действие' обязательны.")

    with t3:
        st.write("История версий конфигурационного файла:")
        history_dir = engine.rules_path.parent / "rules_history"
        
        has_history = False
        if history_dir.exists():
            files = list(history_dir.glob("*.json"))
            if files:
                has_history = True
                for f in sorted(files, reverse=True):
                    with st.expander(f.name):
                        try:
                            content = f.read_text(encoding="utf-8")
                            data = json.loads(content)
                            st.json(data)
                            if st.button("Восстановить эту версию", key=f"restore_{f.name}"):
                                if _save_rules_safe(engine, data):
                                    st.success(f"Правила восстановлены из {f.name}!")
                                    time.sleep(0.5)
                                    st.rerun()
                        except Exception as e:
                            st.write(f"Ошибка чтения файла: {e}")
        
        if not has_history:
            st.info("История правил пуста.")
