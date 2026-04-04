# Алгоритм динамического ценообразования для ритейла (MVP)

Проект для генерации, анализа и применения алгоритмов динамического ценообразования для ритейла.

## Структура проекта

```
Reteil/
├── config.py                  # Общие константы (PRODUCTS, SEED, N_DAYS)
├── requirements.txt           # Зависимости проекта
├── recommender.py             # Rule-based рекомендатор (автономный скрипт)
├── generate_nb.py             # Генератор Jupyter-ноутбука (вспомогательный)
├── generator/
│   └── generate_data.py       # Генератор синтетических данных продаж
├── data/
│   ├── sales_history.csv      # Сгенерированный датасет (100 дней × 5 товаров)
│   ├── recommendations.csv    # Рекомендации от recommender.py
│   ├── recommendations_rules.csv      # Рекомендации (эвристика) из ноутбука
│   ├── recommendations_regression.csv # Рекомендации (регрессия) из ноутбука
│   └── plots/                 # Графики из генератора
├── notebook/
│   └── pricing_mvp.ipynb      # Основной MVP-ноутбук (EDA + модели + симуляция)
├── model/                     # (зарезервировано для будущих моделей)
└── ui/                        # (зарезервировано для интерфейса)
```

## Быстрый старт

### 1. Установить зависимости
```bash
pip install -r requirements.txt
```

### 2. Сгенерировать данные
```bash
python generator/generate_data.py
```

### 3. Открыть ноутбук в Jupyter
```bash
jupyter notebook notebook/pricing_mvp.ipynb
```

### 4. Экспорт в HTML (веб-страница с отчётом)
```bash
jupyter nbconvert --to html notebook/pricing_mvp.ipynb
```
После этого откройте файл `notebook/pricing_mvp.ipynb.html` в браузере.

## Товары

| Товар    | Базовая цена | Эластичность | Базовые продажи |
|----------|:------------:|:------------:|:---------------:|
| Молоко   | 80 ₽         | 2.0          | 300             |
| Хлеб     | 50 ₽         | 1.5          | 250             |
| Сок      | 120 ₽        | 3.0          | 150             |
| Кофе     | 450 ₽        | 1.2          | 80              |
| Шоколад  | 100 ₽        | 2.5          | 200             |

## Стек

Python 3.11 · Pandas · NumPy · scikit-learn · matplotlib · Jupyter Notebook
