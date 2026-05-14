"""Конфигурация логирования для проекта."""
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None):
    """
    Настраивает логирование для всего проекта.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
    """
    # Формат логов
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Базовая конфигурация
    handlers = [logging.StreamHandler(sys.stdout)]

    # Добавляем файловый handler если указан путь
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # Переопределяет существующую конфигурацию
    )

    # Настройка уровней для сторонних библиотек
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено: уровень={level}, файл={log_file}")
