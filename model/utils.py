"""
Вспомогательные функции для аналитики и расчетов.
"""
import numpy as np
import pandas as pd
from collections import Counter

# True = товар отсутствует (out of stock)
_IS_OOS_TRUE = frozenset({
    "1", "true", "t", "yes", "y", "да", "д", "on",
    "oos", "out of stock", "out-of-stock", "outofstock",
    "нет в наличии", "отсутствует", "отсутствие", "недоступен",
})
_IS_OOS_FALSE = frozenset({
    "", "0", "false", "f", "no", "n", "нет", "off",
    "in stock", "instock", "available", "в наличии", "есть",
})


def parse_is_oos_value(value) -> bool:
    """Разбор одного значения is_oos (не использовать .astype(bool) для строк)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0

    text = str(value).strip().lower()
    if text in _IS_OOS_FALSE:
        return False
    if text in _IS_OOS_TRUE:
        return True
    try:
        return float(text.replace(",", ".")) != 0.0
    except ValueError as exc:
        raise ValueError(f"Не удалось разобрать is_oos: {value!r}") from exc


def parse_is_oos_series(series: pd.Series) -> pd.Series:
    """Приводит колонку is_oos к bool: True — нет в наличии."""
    if series.dtype == bool:
        return series
    return series.map(parse_is_oos_value).astype(bool)


def safe_growth_pct(predicted: float, actual: float) -> float:
    """
    Безопасный расчет процента роста с обработкой edge cases.

    Args:
        predicted: Прогнозное значение
        actual: Фактическое значение

    Returns:
        Процент роста. Для роста из нуля возвращает 100.0.
    """
    if actual > 0:
        return (predicted - actual) / actual * 100
    elif predicted > 0:
        return 100.0
    else:
        return 0.0


def safe_mean_or_none(values: list[float]) -> float | None:
    """
    Возвращает среднее значение или None для пустого списка.

    Args:
        values: Список числовых значений

    Returns:
        Среднее значение или None
    """
    return float(np.mean(values)) if values else None


def format_rule_names(names: list[str]) -> str:
    """
    Форматирует список правил для UI с процентным распределением.

    Args:
        names: Список названий правил

    Returns:
        Строка с названием правила или распределением

    Examples:
        >>> format_rule_names(['hold', 'hold', 'hold'])
        'hold'
        >>> format_rule_names(['hold', 'hold', 'undercut'])
        'hold (66%), undercut (33%)'
    """
    if not names:
        return "no_data"
    unique_rules = set(names)
    if len(unique_rules) == 1:
        return names[0]
    counts = Counter(names)
    total = len(names)
    parts = [f"{rule} ({counts[rule]*100//total}%)" for rule in sorted(counts.keys())]
    return ", ".join(parts)


def safe_all_check(bool_list: list[bool]) -> bool:
    """
    Безопасная проверка all() с обработкой пустого списка.

    Args:
        bool_list: Список булевых значений

    Returns:
        True если список непустой и все элементы True, иначе False
    """
    return bool(bool_list) and all(bool_list)
