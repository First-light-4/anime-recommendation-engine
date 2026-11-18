# Anime Recommendation Engine

Проект персональных рекомендаций аниме на основе предпочтений и истории
взаимодействий пользователей, включая новых пользователей вне обучающего датасета.

## Текущий статус

ARE-008A: подготовлены реальные выборки 5k/10k за один согласованный потоковый проход
при фиксированных validation/test. Train после фильтрации: 4621/9455 пользователей,
359967/745354 взаимодействий, vocabulary 5279/6719 аниме. Notebook 07 показывает
все три реальные выборки. План подготовки данных без сети:
`.venv\Scripts\python.exe -m anime_recommender.training.scaling`.
Архитектура, команды, стоимость и ограничения resume:
[ARE-008A](docs/ARE-008A-training-data-scaling.md).
Результаты загрузки и проверок: [контролируемый проход](docs/ARE-008A-controlled-scaling-run.md).

ARE-008B: обучены отдельные ALS/Popularity 5k/10k; 1k переиспользована по fingerprint.
На фиксированной validation ALS 10k: Full Recall@10=0.30515, Full NDCG@10=0.42528.
Основное сравнение без franchise filtering; test не используется. Notebook 05
переключает сохранённые модели через `model_version` без повторной загрузки профиля
и обучения. Default остаётся `1k` до согласования, рекомендована `10k`.
[Метрики, переключение и ограничения ARE-008B](docs/ARE-008B-model-comparison.md).
ARE-008B.1: paired bootstrap по сохранённым результатам выполнен. Уверенное
превосходство 10k над 5k не установлено; рекомендация 10k основана на trade-off
каталога и point estimates, default остаётся 1k.
[Bootstrap, popularity exposure и финальный review](docs/ARE-008B.1-final-model-selection.md).

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

В начале Notebook 04 задаются параметры ALS, `top_k` и validation user_id.
При смене пользователя или `top_k` достаточно повторить разделы рекомендаций;
переобучение не требуется. Context содержит исходные оценки и warm/cold-признаки;
скрытые targets используются только для последующей диагностики и колонки `is_hit`.
Добавлены графики метрик, scores, популярности выдачи и попаданий.

Ручная проверка YummyAnime вынесена в `notebooks/05_yummyanime_recommendations.ipynb`.
Notebook 04 не обращается к API. Notebook 05 использует сохранённую модель и каталог,
не читает train/validation/test-взаимодействия и никогда не обучает модель.
Никнейм берётся из `YUMMYANIME_TEST_NICKNAME`, иначе запрашивается через `input()`;
Enter без ввода выбирает `First_Light`. Выполняются до трёх запросов существующим
клиентом (профиль, списки, избранное). При смене только `top_k` достаточно повторить
ячейку Top-N: уже полученная история используется без новых API-запросов.
По умолчанию Notebook 05 показывает 15 рекомендаций. Таблица содержит Rank,
English Title, Russian Title, ALS Score и MAL ID. Для русских названий допускается
один ограниченный запрос `/anime` по MAL ID текущей выдачи (до 20 ID); результат
кэшируется только в памяти. Достоверно не найденные названия отображаются как `—`.
При Popularity fallback ALS Score также равен `—`: счётчик популярности не является
ALS score. Выбранная модель остаётся в сводке профиля.
Для просмотра истории установите `show_history=True`. Бинарные positives: оценка
от 5, просмотренное без оценки или избранное; planned/dropped исключаются из positives.
Сопоставление выполняется через MAL ID, fold-in — по warm positives; все сопоставленные
известные аниме исключаются из выдачи. При пустом warm context используется Popularity.
Подготовка сигналов находится в `preprocessing/profile_signals.py`, исследовательские
вызовы — в `recommendation_exploration.py`. История API хранится только в памяти.
Описание разделения и проверок: [ARE-006.1](docs/ARE-006.1-separate-profile-notebook.md).

## Evaluation framework (ARE-007A)

```powershell
.venv\Scripts\python.exe -m anime_recommender.evaluation.ranking_evaluator
```

Конфигурация `configs/evaluation.json`: только validation, K=5/10/15/20.
Сохранённые модели оцениваются без обучения и обращения к API. Реализованы Precision,
Recall, NDCG, MAP, HitRate, покрытие каталога, Full/Warm-режимы и cohort analysis.
ALS-only и ALS с Popularity fallback публикуются отдельно. Для пустых targets метрики
undefined; macro исключает их и показывает eligible counts. MAP нормируется через
`min(K, число targets)`, Precision — через K даже при короткой выдаче.

Отчёты JSON/Parquet сохраняются в игнорируемом `artifacts/evaluation/are007a/`
с fingerprints, версиями, конфигурациями, UTC timestamp и проверкой целостности.
Notebook `06_model_evaluation.ipynb` показывает готовые таблицы и графики из `src/`.
Test не читается; threshold и splits не меняются. Определения, реальные результаты
и ограничения: [ARE-007A](docs/ARE-007A-evaluation-framework.md).

## Franchise-aware post-processing (ARE-007C)

Notebook 05 использует локальный структурированный `viewing_order` YummyAnime для
Attack on Titan, Demon Slayer и Cyberpunk: Edgerunners. Настройки:
`configs/franchise_filter.json`, по умолчанию max=1 на известную
франшизу. Точные известные MAL ID исключаются при любом статусе; только completed
подтверждает просмотр для правил порядка. Спин-оффы, фильмы и альтернативные ветки
не превращаются автоматически в линейную обязательную последовательность.

Кандидаты запрашиваются с запасом и при необходимости расширяются до train vocabulary.
Scores и порядок допустимых рекомендаций сохраняются; запрещённые части не возвращаются
ради заполнения Top-N. ALS не переобучается, Popularity fallback сохраняется. Никаких
сетевых запросов связей в post-processing нет. Неизвестные связи не определяются по
названиям и не приводят к необоснованным исключениям.

Порядок основной линии строится по `data.index`, роли — по `data.text`; дополнения
и компиляции не становятся обязательными сезонами. После самой поздней completed-части
допускается только следующий этап. Отсутствующий в модели или несопоставленный следующий
этап не пропускается ради более позднего сезона. Для подтверждённых альтернатив
достаточно одной версии; рекап не считается автоматически просмотром всего сериала.

Покрытие снимка — 32 MAL-сопоставленных элемента, 22 из 2091 train items; полную
franchise-diversity не заявляем. Обновление источника выполняется отдельно и ограниченно,
online inference читает только `configs/yummyanime_franchise_relations.json`.
Исследование API и сценарии: [viewing_order](docs/ARE-007C-yummy-viewing-order.md).
Отдельная validation-диагностика, не изменяющая результаты ARE-007A:

```powershell
.venv\Scripts\python.exe -m anime_recommender.evaluation.franchise_evaluation
```

Исходный отчёт: [ARE-007C](docs/ARE-007C-franchise-filtering.md). Результаты хранятся только
локально в игнорируемом `artifacts/evaluation/are007c/`. В ratings-only validation нет
completed-статусов: оценки не считаются доказательством просмотра. Это ограничивает
интерпретацию сравнения с реальным online-профилем.

## Automatic metadata discovery (ARE-007C.2)

```powershell
# По умолчанию: весь train vocabulary как план, без HTTP и записи файлов.
.venv\Scripts\python.exe -m anime_recommender.ingestion.viewing_order_metadata
# Только небольшой диагностический subset; полный запуск требует отдельного согласования.
.venv\Scripts\python.exe -m anime_recommender.ingestion.viewing_order_metadata --execute --max-requests 5 --mal-ids 37450,38329
```

Настройки: `configs/viewing_order_metadata.json`. Реализованы budget на все попытки,
rate limit, временные retries/backoff и Retry-After, checkpoint/resume, TTL и negative
cache, объединение пересечений по ID с проверкой неоднозначностей. Родственные части
вне ALS vocabulary сохраняются. Ручного списка франшиз в алгоритме discovery нет.

Версии mapping/parts/source records/coverage и checksums находятся только в
`data/processed/franchises/`, исключённом из Git. Notebook 05 предпочитает проверенный
`latest.json`, затем статический partial fallback; при отсутствии обоих продолжает
точное seen filtering с предупреждением. Online discovery не выполняется.

Живая диагностика Rascal сейчас не завершена: оба ограниченных запуска получили
HTTP 502 на MAL lookup. Связи не подставлены вручную. Реализация и synthetic regression
проверены; подробности и ограничения — [ARE-007C.2](docs/ARE-007C.2-metadata-pipeline.md).
