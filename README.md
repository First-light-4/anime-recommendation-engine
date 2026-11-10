# Anime Recommendation Engine

Проект персональных рекомендаций аниме на основе предпочтений и истории
взаимодействий пользователей, включая новых пользователей вне обучающего датасета.

## Текущий статус

ARE-001: создана базовая структура Python-проекта, настроены сборка и smoke test.
Для ARE-002.1 добавлен исследовательский notebook интеграции с YummyAnime API.
ARE-003 содержит исследование обучающих данных; ARE-004 добавляет локальный pipeline
отбора полных историй, проверки схемы, очистки и сохранения Parquet.
Обучение моделей, REST API, интерфейс и инфраструктурные сервисы пока не реализованы.

## Планируемая архитектура

Data Sources → Ingestion → Storage → Preprocessing → Training → Evaluation →
Model Registry → Inference → REST API → Frontend.

Планируется начать с popularity baseline и implicit ALS с fold-in для новых
пользователей, затем добавить ранжирование CatBoost. Предполагаемый стек включает
PostgreSQL, Parquet, Polars/Pandas, SciPy, MLflow, FastAPI, Streamlit и Docker
Compose. На текущем этапе подключены только библиотеки для исследования и обработки
данных; инфраструктурные сервисы не запускаются. Исследование источников и MAL mapping
описано в отчёте ARE-003.

## Структура

- `src/anime_recommender/` — импортируемый Python-пакет.
  Подпакеты: `ingestion`, `preprocessing`, `training`, `evaluation`, `inference`, `api`.
- `tests/` — smoke tests и синтетические проверки исследования и обработки данных.
- `configs/` — конфигурация подготовки обучающих таблиц.
- `data/` — локальные данные, содержимое исключено из Git.
- `notebooks/` — исследование YummyAnime API.
- `scripts/` — исследовательские скрипты.

## Локальная установка и проверка

Требуется Python 3.11+; первоначальная проверка выполнена на Python 3.12.
Команды для PowerShell из корня репозитория:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

Runtime-зависимостей нет. Для разработки используется `pytest>=8,<9`,
для сборки — `setuptools>=68`. Версии пока не зафиксированы lock-файлом.
Для установки без инструментов разработки используйте `pip install -e .`.

Не добавляйте в Git секреты, датасеты и локальные артефакты моделей.
`.gitignore` исключает `.env`, виртуальные окружения, данные и MLflow artifacts.

## Исследование YummyAnime API

Notebook: `notebooks/01_yummyanime_api_exploration.ipynb`.

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
.venv\Scripts\python.exe -m jupyterlab
```

Выберите kernel локального `.venv`. Публичный application token задаётся в
`YUMMYANIME_APPLICATION_TOKEN` через корневой `.env` или окружение.
Не используйте приватный токен. Значения окружения имеют приоритет над `.env`.
Notebook выполняет до четырёх GET-запросов для одного профиля, строит DataFrame,
проверяет качество данных и сверяет избранное и распределение оценок с отдельными методами API.
Счётчики сайта заполняются вручную. Списки являются снимком состояния, а не историей событий.

Не сохраняйте реальные данные в outputs: перед сохранением notebook очистите
все outputs и верните ручные счётчики сайта в `None`. Jupyter может автоматически
сохранять результаты во время работы. DataFrame не экспортируется в файлы.

## Исследование обучающего датасета (ARE-003)

Отчёт: [docs/ARE-003-training-dataset-research.md](docs/ARE-003-training-dataset-research.md).
Ограниченное воспроизводимое EDA без обучения и сохранения исходных данных:

```powershell
.venv\Scripts\python.exe scripts/research_training_dataset.py
```

Читается каталог и пять небольших HTTP Range-фрагментов рейтингов закреплённой
ревизии Hugging Face. Для сопоставления с MAL используется `mal_url`, а не внутренний
`animeID`. Необязательная проверка собственного профиля описана в отчёте.

## Подготовка обучающих таблиц (ARE-004)

Инструкция и ограничения: [docs/ARE-004-training-data-pipeline.md](docs/ARE-004-training-data-pipeline.md).

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
# Только после разрешения на один полный проход исходного CSV (~2,25 GB трафика):
.venv\Scripts\python.exe -m anime_recommender.ingestion.pipeline --allow-full-scan
# Повторно, используя локальный кэш, без сети:
.venv\Scripts\python.exe -m anime_recommender.ingestion.pipeline
```

Конфигурация: `configs/training_data.json`. По умолчанию выбираются полные истории
1000 пользователей по seed. Полный CSV не сохраняется. Parquet, кэш и quality report
хранятся в игнорируемом `data/`. Без кэша обычный запуск отказывается скачивать источник.
Рейтинги сохраняются без implicit-преобразования; конфликтующие пары исключаются.
