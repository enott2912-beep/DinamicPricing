"""
Вспомогательные функции для аналитики и расчетов.
"""
import numpy as np
from collections import Counter


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
