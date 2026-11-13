# Anime Recommendation Engine

Проект персональных рекомендаций аниме на основе предпочтений и истории
взаимодействий пользователей, включая новых пользователей вне обучающего датасета.

## Текущий статус

ARE-001: создана базовая структура Python-проекта, настроены сборка и smoke test.
Для ARE-002.1 добавлен исследовательский notebook интеграции с YummyAnime API.
ARE-003 содержит исследование обучающих данных; ARE-004 добавляет локальный pipeline
отбора полных историй, проверки схемы, очистки и сохранения Parquet.
ARE-006: реализованы Popularity, CPU ALS, fold-in новых пользователей, сохранение моделей
и диагностика на validation. REST API, интерфейс и инфраструктурные сервисы пока не реализованы.

## Планируемая архитектура

Data Sources → Ingestion → Storage → Preprocessing → Training → Evaluation →
Model Registry → Inference → REST API → Frontend.

Планируется начать с popularity baseline и implicit ALS с fold-in для новых
пользователей, затем добавить ранжирование CatBoost. Предполагаемый стек включает
PostgreSQL, Parquet, Polars/Pandas, SciPy, MLflow, FastAPI, Streamlit и Docker
Compose. На текущем этапе подключены библиотеки для исследования, обработки данных
и локального обучения моделей; инфраструктурные сервисы не запускаются. Источники и MAL mapping
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
ARE-004.1: HTTP-клиент, нормализация и проверки вынесены в `src/anime_recommender/`.
Модули и примеры: [docs/ARE-004.1-modular-exploration.md](docs/ARE-004.1-modular-exploration.md).

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

Исследование готовых таблиц: `notebooks/02_training_data_exploration.ipynb`.
Для обоих notebook установите `.[dev,data,research]` и выберите kernel `.venv`.
Второй notebook читает только локальные Parquet, проверяет hashes отчёта ARE-004,
показывает статистику и покрытие MAL ID. `check_profile=False` отключает его два
API-запроса и оставляет полностью локальное исследование. Датасет повторно не скачивается.

## Подготовка ML-датасета (ARE-005 / ARE-005.1)

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml]"
.venv\Scripts\python.exe -m anime_recommender.training.dataset
```

Конфигурация: `configs/ml_dataset.json`. Основной порог — `rating >= 5`, доступны
5/6/7/8. Сначала делятся пользователи, затем исходные очищенные взаимодействия
validation/test разделяются на context/target, и только после этого выделяются
positives. Все evaluation-пользователи сохраняются, включая пользователей без
positive targets или warm context. Минимумы активности 5/5 применяются только к train.
Item vocabulary строится исключительно по train. Cold items сохраняются в таблицах
и статистике, но исключаются из отдельных CSR для fold-in context.
Parquet, CSR NPZ, mappings и отчёт находятся в игнорируемом `data/processed/ml/`.
ALS не обучается; API и исходный датасет не запрашиваются.

Исследование: `notebooks/03_ml_dataset_preparation.ipynb`, для него нужен также extra
`research`. Текущий протокол и статистика: [docs/ARE-005.1-evaluation-protocol.md](docs/ARE-005.1-evaluation-protocol.md).
Требования будущего online inference: [ADR-001](docs/ADR-001-yummyanime-positive-feedback.md).

## Обучение рекомендательных моделей (ARE-006)

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,models,research]"
.venv\Scripts\python.exe -m anime_recommender.training.pipeline
# Загрузка без повторного обучения:
.venv\Scripts\python.exe -m anime_recommender.training.pipeline --load-only
```

Конфигурация: `configs/recommendation_models.json`. Используются локальные артефакты
ARE-005.1; test не читается. ALS: 64 фактора, regularization=0.1, 20 итераций, seed=42,
CPU/BLAS по одному потоку. Обе модели исключают весь известный context, включая низкие
и нулевые оценки. Fold-in использует warm positives и не изменяет item factors.
Без warm positives ALS явно недоступен; автоматический fallback в диагностику не включён.
ALS scores не являются вероятностями.

Модели и mappings: `artifacts/models/are006/`; диагностика: `artifacts/validation/are006/`.
Оба каталога игнорируются Git. Native NPZ/Parquet/JSON проверяются по SHA-256 и fingerprint
mapping. Notebook `04_recommendation_models.ipynb` по умолчанию загружает готовые модели;
для первого обучения установите `retrain=True`, затем верните `False`. Outputs перед сохранением очистить.
Отчёт и описание модулей: [ARE-006](docs/ARE-006-recommendation-models.md).

В начале Notebook 04 задаются параметры ALS, `top_k`, validation user_id и необязательный
nickname. При смене пользователя или `top_k` достаточно повторить разделы рекомендаций;
переобучение не требуется. Context содержит исходные оценки и warm/cold-признаки;
скрытые targets используются только для последующей диагностики и колонки `is_hit`.
Добавлены графики метрик, scores, популярности выдачи и попаданий.

Эксперимент YummyAnime выключен при `nickname=None`. После включения выполняются до трёх
запросов существующим клиентом (профиль, списки, избранное). Бинарные positives: оценка
от 5, просмотренное без оценки или избранное; planned/dropped исключаются из positives.
Сопоставление выполняется через MAL ID, fold-in — по warm positives; все сопоставленные
известные аниме исключаются из выдачи. При пустом warm context используется Popularity.
Подготовка сигналов находится в `preprocessing/profile_signals.py`, исследовательские
вызовы — в `recommendation_exploration.py`. История API хранится только в памяти.
