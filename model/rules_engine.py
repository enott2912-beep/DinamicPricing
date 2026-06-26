import json
import datetime
import logging
from pathlib import Path
from simpleeval import simple_eval, NameNotDefined, InvalidExpression

logger = logging.getLogger(__name__)

# Абсолютный путь к data/rules.json для надежности
BASE_DIR = Path(__file__).parent.parent
RULES_FILE = BASE_DIR / "data" / "rules.json"
HISTORY_DIR = BASE_DIR / "data" / "rules_history"
_ENGINES_BY_SESSION: dict[str, "RuleEngine"] = {}


def _resolve_session_id_from_streamlit() -> str | None:
    try:
        import streamlit as st

        return st.session_state.get("session_id")
    except Exception:
        return None

class RuleEngine:
    """
    Динамический движок бизнес-правил, работающий с конфигурацией в JSON.
    Обеспечивает парсинг, выполнение "Что-если" (evaluate) и версионирование правил.
    """

    def __init__(self, rules_path: Path = RULES_FILE, session_id: str | None = None):
        self.rules_path = rules_path
        self.session_id = session_id
        self.rules = self.load_rules()

    def load_rules(self) -> list[dict]:
        if self.session_id:
            from ui.session_store import load_rules as load_rules_for_session

            try:
                session_rules = load_rules_for_session(self.session_id)
            except Exception as exc:
                logger.exception("Не удалось загрузить правила сессии %s", self.session_id)
                raise RuntimeError(
                    "Не удалось загрузить правила сессии из базы данных. Повторите попытку позже."
                ) from exc
            if session_rules is not None:
                return session_rules
            # В сессии ещё нет своих правил — подставляем глобальный шаблон из файла.
            return self._load_rules_from_file()

        return self._load_rules_from_file()

    def _load_rules_from_file(self) -> list[dict]:
        if not self.rules_path.exists():
            return []
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Ошибка загрузки правил из %s: %s", self.rules_path, e)
            return []

    def _backup_session_rules_to_history(self, old_rules: list[dict]) -> None:
        if not old_rules:
            return
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sid = (self.session_id or "nosession")[:8]
        backup_path = HISTORY_DIR / f"rules_session_{sid}_{timestamp}.json"
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(old_rules, f, ensure_ascii=False, indent=2)

    def _save_rules_to_session(self, new_rules: list[dict]) -> None:
        from ui.session_store import load_rules as load_rules_for_session
        from ui.session_store import save_rules as save_rules_for_session

        try:
            old_rules = load_rules_for_session(self.session_id) or []
            self._backup_session_rules_to_history(old_rules)
            save_rules_for_session(self.session_id, new_rules)
        except Exception as exc:
            logger.exception("Не удалось сохранить правила сессии %s", self.session_id)
            raise RuntimeError(
                "Не удалось сохранить правила сессии в базу данных. Глобальный файл правил не изменён."
            ) from exc
        self.rules = new_rules

    def _save_rules_to_file(self, new_rules: list[dict]) -> None:
        if self.rules_path.exists():
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = HISTORY_DIR / f"rules_{timestamp}.json"
            try:
                import shutil

                shutil.copy2(self.rules_path, backup_path)
            except Exception as e:
                logger.warning("Ошибка создания бэкапа правил: %s", e)

        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rules_path, "w", encoding="utf-8") as f:
            json.dump(new_rules, f, ensure_ascii=False, indent=2)
        self.rules = new_rules

    def save_rules(self, new_rules: list[dict]):
        """Сохраняет правила и создает backup в rules_history."""
        from model.rules_validator import validate_rules_list

        is_valid, errors = validate_rules_list(new_rules)
        if not is_valid:
            error_msg = "Ошибки валидации правил:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.session_id:
            self._save_rules_to_session(new_rules)
            return

        self._save_rules_to_file(new_rules)

    def evaluate(self, context: dict) -> tuple[float, str]:
        """
        Проходит по массиву правил сверху вниз. Если правило выполняется, возвращает (new_price, rule_name).
        В context ожидаются:
        price, comp_1, comp_2, comp_price, sales, avg_sales_7d, cogs.
        Автоматически добавляет 'margin' = (price - cogs) / price (если price > 0).
        Также разрешаем встроенные функции max, min, abs, round.
        """
        if not self.rules:
            return context.get("price", 0.0), "no_rules"

        # Безопасный namespace для математики
        safe_locals = context.copy()

        # Расчет базовых производных переменных
        price = safe_locals.get("price", 0.0)
        cogs = safe_locals.get("cogs", 0.0)
        if price > 0:
            safe_locals["margin"] = (price - cogs) / price
        else:
            safe_locals["margin"] = 0.0

        # Разрешенные функции
        safe_functions = {
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
        }

        for rule in self.rules:
            if not rule.get("condition") or not rule.get("action"):
                continue

            try:
                # Безопасная оценка условия через simpleeval
                is_true = simple_eval(
                    rule["condition"],
                    names=safe_locals,
                    functions=safe_functions
                )

                if is_true:
                    # Безопасное вычисление нового значения
                    new_val = simple_eval(
                        rule["action"],
                        names=safe_locals,
                        functions=safe_functions
                    )
                    return round(float(new_val), 2), rule.get("name", "unnamed_rule")

            except (NameNotDefined, InvalidExpression, ValueError, TypeError) as e:
                logger.warning(f"Ошибка в правиле '{rule.get('name', 'unnamed')}': {e}")
                continue
            except Exception as e:
                logger.error(f"Неожиданная ошибка в правиле '{rule.get('name', 'unnamed')}': {e}")
                continue

        return round(float(context.get("price", 0.0)), 2), "hold"


def get_rule_engine() -> RuleEngine:
    session_id = _resolve_session_id_from_streamlit()
    if session_id:
        if session_id not in _ENGINES_BY_SESSION:
            _ENGINES_BY_SESSION[session_id] = RuleEngine(session_id=session_id)
        return _ENGINES_BY_SESSION[session_id]
    # Fallback для вне-UI запусков.
    return RuleEngine()
