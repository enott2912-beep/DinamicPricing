"""Валидация правил ценообразования."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Разрешенные переменные в правилах
ALLOWED_VARIABLES = {
    "price", "comp_1", "comp_2", "comp_price",
    "sales", "avg_sales_7d", "cogs", "margin"
}

# Разрешенные функции
ALLOWED_FUNCTIONS = {"max", "min", "abs", "round"}

# Разрешенные операторы (проверяется через AST)
ALLOWED_OPERATORS = {"+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!=", "and", "or", "not"}


def validate_rule(rule: dict) -> tuple[bool, str]:
    """
    Валидирует правило перед сохранением.

    Args:
        rule: Словарь с полями name, condition, action

    Returns:
        (is_valid, error_message)
    """
    # Проверка обязательных полей
    if not isinstance(rule, dict):
        return False, "Правило должно быть словарем"

    if not rule.get("name"):
        return False, "Отсутствует название правила"

    if not rule.get("condition"):
        return False, "Отсутствует условие правила"

    if not rule.get("action"):
        return False, "Отсутствует действие правила"

    # Проверка длины
    if len(rule["name"]) > 100:
        return False, "Название правила слишком длинное (макс. 100 символов)"

    if len(rule["condition"]) > 500:
        return False, "Условие правила слишком длинное (макс. 500 символов)"

    if len(rule["action"]) > 200:
        return False, "Действие правила слишком длинное (макс. 200 символов)"

    # Тестовый контекст для проверки
    test_context = {
        "price": 100.0,
        "comp_1": 105.0,
        "comp_2": 95.0,
        "comp_price": 95.0,
        "sales": 250.0,
        "avg_sales_7d": 240.0,
        "cogs": 60.0,
        "margin": 0.4,
    }

    safe_functions = {
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
    }

    # Проверка условия
    try:
        from simpleeval import simple_eval, NameNotDefined, InvalidExpression

        result = simple_eval(
            rule["condition"],
            names=test_context,
            functions=safe_functions
        )
        if not isinstance(result, (bool, int, float)):
            return False, f"Условие должно возвращать boolean, получено {type(result).__name__}"
    except NameNotDefined as e:
        return False, f"Неизвестная переменная в условии: {e}"
    except InvalidExpression as e:
        return False, f"Недопустимое выражение в условии: {e}"
    except Exception as e:
        return False, f"Ошибка в условии: {e}"

    # Проверка действия
    try:
        result = simple_eval(
            rule["action"],
            names=test_context,
            functions=safe_functions
        )
        if not isinstance(result, (int, float)):
            return False, f"Действие должно возвращать число, получено {type(result).__name__}"

        # Проверка разумности результата
        if result < 0:
            return False, "Действие возвращает отрицательную цену"

        if result > 10000:
            return False, "Действие возвращает нереалистично высокую цену (>10000)"

    except NameNotDefined as e:
        return False, f"Неизвестная переменная в действии: {e}"
    except InvalidExpression as e:
        return False, f"Недопустимое выражение в действии: {e}"
    except Exception as e:
        return False, f"Ошибка в действии: {e}"

    return True, ""


def validate_rules_list(rules: list[dict]) -> tuple[bool, list[str]]:
    """
    Валидирует список правил.

    Args:
        rules: Список правил

    Returns:
        (all_valid, list_of_errors)
    """
    if not isinstance(rules, list):
        return False, ["Правила должны быть списком"]

    if len(rules) > 50:
        return False, ["Слишком много правил (макс. 50)"]

    errors = []
    for i, rule in enumerate(rules):
        is_valid, error = validate_rule(rule)
        if not is_valid:
            errors.append(f"Правило #{i+1} ({rule.get('name', 'unnamed')}): {error}")

    return len(errors) == 0, errors
