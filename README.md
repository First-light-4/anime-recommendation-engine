# Anime Recommendation Engine

Проект персональных рекомендаций аниме на основе предпочтений и истории
взаимодействий пользователей, включая новых пользователей вне обучающего датасета.

## Текущий статус

ARE-001: создана базовая структура Python-проекта, настроены сборка и smoke test.
Для ARE-002.1 добавлен исследовательский notebook интеграции с YummyAnime API.
Алгоритмы рекомендаций, production ingestion, API, интерфейс и инфраструктурные сервисы
пока не реализованы. Подпакеты компонентов содержат только `__init__.py`.

## Планируемая архитектура

Data Sources → Ingestion → Storage → Preprocessing → Training → Evaluation →
Model Registry → Inference → REST API → Frontend.

Планируется начать с popularity baseline и implicit ALS с fold-in для новых
пользователей, затем добавить ранжирование CatBoost. Предполагаемый стек включает
PostgreSQL, Parquet, Polars/Pandas, SciPy, MLflow, FastAPI, Streamlit и Docker
Compose. Эти зависимости и сервисы пока не подключены. Источники данных,
их лицензии и совместимость идентификаторов предстоит проверить перед ingestion.

## Структура

- `src/anime_recommender/` — импортируемый Python-пакет.
  Подпакеты: `ingestion`, `preprocessing`, `training`, `evaluation`, `inference`, `api`.
- `tests/` — тесты, сейчас только проверка импорта пакета и его компонентов.
- `configs/` — будущие конфигурации.
- `data/` — локальные данные, содержимое исключено из Git.
- `notebooks/` — будущие исследовательские ноутбуки.
- `scripts/` — будущие служебные скрипты.

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
Notebook выполняет два GET-запроса для одного профиля, строит DataFrame и статистику.
Счётчики сайта заполняются вручную. Списки являются снимком состояния, а не историей событий.

Не сохраняйте реальные данные в outputs: перед сохранением notebook очистите
все outputs и верните ручные счётчики сайта в `None`. Jupyter может автоматически
сохранять результаты во время работы. DataFrame не экспортируется в файлы.
