# Алгоритм динамического ценообразования для ритейла (MVP)

## Структура проекта

```
Reteil/
├── config.py                  # Константы (PRODUCTS, SEED, N_DAYS)
├── requirements.txt           # Зависимости
├── recommender.py             # Rule-based рекомендатор
├── generator/
│   └── generate_data.py       # Генератор синтетических данных
├── ui/
│   └── app.py                 # Streamlit-интерфейс
├── notebook/
│   ├── pricing_mvp.ipynb      # MVP-ноутбук (EDA + модели + симуляция)
│   └── pricing_mvp.html       # HTML-отчёт для просмотра в браузере
└── data/                      # Генерируемые данные (в .gitignore)
    ├── sales_history.csv
    ├── recommendations.csv
    └── plots/
```

## Быстрый старт

```bash
pip install -r requirements.txt
python generator/generate_data.py
streamlit run ui/app.py
```

## Товары

| Товар    | Базовая цена | Эластичность | Базовые продажи |
|----------|:------------:|:------------:|:---------------:|
| Молоко   | 80 ₽         | 2.0          | 300             |
| Хлеб     | 50 ₽         | 1.5          | 250             |
| Сок      | 120 ₽        | 3.0          | 150             |
| Кофе     | 450 ₽        | 1.2          | 80              |
| Шоколад  | 100 ₽        | 2.5          | 200             |

## Стек

Python 3.11 · Pandas · NumPy · scikit-learn · matplotlib · Jupyter · Streamlit
