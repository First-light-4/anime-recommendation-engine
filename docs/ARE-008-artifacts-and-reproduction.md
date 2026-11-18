# ARE-008 — Default 10k, local artifacts and reproduction

Финально согласован **ALS 10k**, Notebook 05: `model_version="10k"`, `top_k=15`.
1k/5k доступны через тот же переключатель. Никаких автоматических fit/download,
пересборки данных или переключения на другую версию при отсутствии bundle нет.

## Что есть в Git, а что получить отдельно

Git содержит код, configs, tests, Notebook без outputs и отчёты с агрегатами.
**Публичного URL для скачивания готовых model binaries сейчас нет.** Git clone
сам по себе не даёт готовую 10k-модель. Платный registry или новый hosting не подключены.

Можно скопировать свой уже подготовленный trusted local bundle из другой рабочей
копии, сохранив пути ниже, или явно воспроизвести его командами следующего раздела.
Не копировать `.env`, чужие профили, API cache или Jupyter outputs.

| Version | Каталог bundle |
| --- | --- |
| 1k | `artifacts/models/are006/` |
| 5k | `artifacts/models/are008b/5k/` |
| 10k (default) | `artifacts/models/are008b/10k/` |

Для каждой версии нужен **весь** каталог из шести файлов:
`als.npz`, `item_mapping.parquet`, `train_user_mapping.parquet`, `popularity.parquet`,
`training_config.json`, `metadata.json`. Копировать один ALS NPZ недостаточно:
fingerprints связывают модель с её item mapping, конфигурацией и train data.
10k имеет 6719 item factors и 9455 train-user factors, по 64 компоненты.

Дополнительно Notebook 05 требует:

- `data/processed/anime_catalog.parquet` и соответствующий
  `data/processed/data_quality_report.json` ARE-004, с проверяемым catalog SHA.
- `configs/franchise_filter.json` уже есть в Git. Для прежнего franchise coverage
  нужно перенести свой локальный `data/processed/franchises/` целиком. Если его нет,
  используется доступный локальный static fallback или пустой mapping с явным
  предупреждением. Online discovery не выполняется. Не переносить static mapping
  вместо prepared metadata без проверки его источника и версии.
- Для реального профиля — свой локальный `.env` по `.env.example`, только публичный
  `YUMMYANIME_APPLICATION_TOKEN`. Для загрузки модели токен не нужен.

Source train/validation/test datasets для Notebook 05 **не нужны**. Они нужны только
для offline воспроизведения обучения/оценки. Notebook загружает self-contained
model bundle и каталог, а profile cache существует лишь в памяти текущего kernel.
SHA всех bundle файлов, mapping fingerprint, версия и catalog alignment проверяются
при загрузке. Отсутствующий/несовместимый bundle вызывает ошибку с выбранной версией.

## Воспроизведение на чистом клоне

Ниже команды для **новой пустой рабочей копии**, не для перезаписи текущих моделей.
Каждый `--allow-full-scan` — явное разрешение оператора на чтение источника.
В этой публикации эти команды повторно **не запускались**.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,data,ml,models,research]"

# ARE-004: полные истории baseline 1000, pinned source + SHA verification.
.venv\Scripts\python.exe -m anime_recommender.ingestion.pipeline --allow-full-scan
# ARE-005.1: воспроизводимый user-disjoint split / raw holdout / positive feedback.
.venv\Scripts\python.exe -m anime_recommender.training.dataset
# ARE-006: только здесь явно обучается 1k baseline в новой рабочей копии.
.venv\Scripts\python.exe -m anime_recommender.training.pipeline

# ARE-008A: один ОБЩИЙ дополнительный проход для 5k и 10k.
.venv\Scripts\python.exe -m anime_recommender.training.scaling --execute --allow-full-scan
# ARE-008B: использовать 1k bundle, обучить только отсутствующие 5k/10k.
.venv\Scripts\python.exe -m anime_recommender.training.scaling_experiments --train-new
# ARE-008B.1: анализ уже сохранённых validation results, без fit.
.venv\Scripts\python.exe -m anime_recommender.evaluation.model_selection --resamples 10000 --seed 42

.venv\Scripts\python.exe -m jupyterlab notebooks/05_yummyanime_recommendations.ipynb
```

Чистому клону без исходных локальных cache нужны **два** полных прохода ratings.csv:
первый для исходного baseline, второй общий для 5k/10k, около 4.51 ГБ ratings payload
плюс каталоги. Существующей рабочей копии с baseline для ARE-008A потребовался
только **один** новый проход. Полный CSV не сохраняется. Download не запускается
из Notebook; interruption не разрешает автоматический повтор полного прохода.

Если verified cache и datasets уже перенесены, флаг full scan не нужен:
`python -m anime_recommender.training.scaling --execute` переиспользует их.
Scaling references к shared raw evaluation файлам локальные абсолютные: при переносе
в другую директорию безопаснее воспроизвести preparation, чем вручную редактировать
сертифицирующие hashes/пути. Self-contained model bundles этого ограничения не имеют.

Configs фиксируют источник `mramazan/User-Animelist-Dataset`, revision
`6d0f78003440a90ca9a417389f13c47387f0c942`, source size/SHA, seed=42,
threshold=5, train k-core=5/5. Источник CC-BY-4.0; attribution и изменения описаны
в ARE-003/004. Не выбирать свежий revision и не менять пороги для воспроизведения.
ALS: 64 factors, regularization=.1, iterations=20, seed=42, float32, CPU/BLAS 1 thread,
alpha=1, CG enabled, cg_steps=3. Версии библиотек записаны в model metadata.
Dependency ranges не являются lockfile: для битовой идентичности нужны прежние
версии/платформа; одинаковый seed не обещает одинаковые float results на любом CPU.

## Validation и известные ограничения

10k: Full Recall@10 **0.305145**, Full NDCG@10 **0.425284**, vocabulary **6719**;
к 1k **+7.56% Recall / +9.56% NDCG**. Использована фиксированная validation из
100 пользователей; test не использовался при выборе модели. Main evaluation
ALS-only, без franchise filtering. Warm/cold пересчитывались для каждого vocabulary,
но исходные history/context/targets и cohort IDs не менялись.

10k vs 5k: paired-bootstrap 95% CI Recall [-0.020856; +0.028158],
NDCG [-0.016071; +0.019663]. **Статистически уверенное превосходство не установлено**;
это не доказательство равенства. CI pointwise, 10000 resamples, seed=42, без
multiple-comparison correction; train/seed/domain variability не моделировалась.
Повторный selection на той же validation ограничивает обобщаемость результатов.
Отсутствие timestamps ограничивает оценку будущего временного поведения.

**99%** ALS 10k рекомендаций на validation приходятся на top-10% train vocabulary
по popularity. Это сильная концентрация, улучшение long-tail diversity не заявляется.
Franchise mapping для 10k знает **27/6719 items, около 0.40%**, всего 3 группы;
неизвестные франшизы не достраиваются по названиям. Без локальных metadata coverage
будет ниже, вплоть до нуля, с предупреждением. Нельзя обещать предотвращение
неподходящих sequels во всём каталоге или перенос ALS-only метрик на filtered Top-15.

Подробные числа: [ARE-008B](ARE-008B-model-comparison.md),
[bootstrap/diversity ARE-008B.1](ARE-008B.1-final-model-selection.md).

## Финальная проверка публикации

**202 tests passed** (14.55 с). Notebook 05 выполнен с default 10k и синтетическим
профилем; проверены 1k→10k→5k→1k, Top-15, mapping fingerprints, known-item exclusion,
profile cache, franchise filtering и partial-coverage warning. Network, fit,
ingestion и test reads из Notebook заблокированы. Проверка заняла 2.66 с.
SHA 119 защищённых файлов моделей, исходных splits, metadata, исходных evaluation
reports и Notebook 01 совпали до/после. Модели не перезаписывались.
