# ARE-008A — Fixed evaluation, growing train

Этот документ описывает реализацию и первоначальную проверку **до загрузки**.
После отдельного разрешения выполнен один проход, реальные 5k/10k подготовлены:
[итоговые характеристики и проверки](ARE-008A-controlled-scaling-run.md).

Реализована инфраструктура подготовки, без обучения и без нового чтения источника.
Реальные sample_5000/sample_10000 пока **не получены**: нужен отдельный разрешённый проход.
Notebook 01–06, текущие модели, ARE-005.1 и franchise filtering не изменяются.

## Архитектура

- `ingestion/scaling.py`: проверка baseline/provenance, план, один общий cache максимального
  размера, вызовы существующих acquisition и cleaning, последовательная подготовка размеров.
- `training/scaling.py`: вложенный отбор, фиксированные holdouts, проверки, сохранение,
  загрузка, сводки и явно синтетическая демонстрация. Здесь нет обучения модели.
- `training/dataset.py`: необязательные `fixed_assignments`/`fixed_holdouts`;
  стандартный ARE-005.1 вызов сохраняет прежнее поведение. Переиспользуются
  positive_feedback, filter_train, build_train_matrix/build_context_matrix и leakage checks.
- `configs/scaling.json`: sizes 5000/10000 и отдельные пути.
- `notebooks/07_training_data_scaling.ipynb`: локальный baseline и синтетические
  10/20/30 пользователей; вывод агрегатов, без пользовательских историй и outputs.
- `tests/test_scaling.py`: полностью offline regression tests.

## Отбор и неизменность evaluation

ARE-004 выбирает минимальные SHA-256(`seed:userID`), поэтому bottom-1000 вложен
в bottom-5000, а тот — в bottom-10000 при неизменном источнике и seed=42.
Сначала сохраняются ВСЕ baseline IDs; новые ранжируются тем же SHA-256, с ID как
tie-break. Если baseline когда-либо получен другим протоколом, явное сохранение
baseline продолжает работать, но выборка уже не объявляется чистым bottom-k.

Один потоковый проход выбирает полные истории 10000 пользователей. Из него локально
получается 5000. Полнота историй обеспечивается существующим алгоритмом: порог
приоритета только уменьшается; пользователь из окончательной выборки сохраняется
с первого появления. Размер относится к raw users, включая 200 закреплённых eval users;
даже пользователь без строк после очистки остаётся в sampling manifest/assignments.

Не вызывается новый assign_users для расширений. Все старые train pool IDs остаются
train; новые попадают только в train. Сырые context/target читаются из baseline;
positive feedback применяется после fixed holdout. Сравниваются исходные строки
и рейтинги; изменения cold flags разрешены. Удержанные train users и пары предыдущего
размера должны входить в следующий. При тех же порогах k-core монотонен при добавлении
полных новых историй; нарушения вызывают DataValidationError.

Train vocabulary строится только из train. Матрицы evaluation содержат только warm
positive context, targets в fold-in не входят. Пользователи без context/positive targets
сохраняются. Test читается для фиксации идентичности и статистики покрытия; метрики
качества, обучение и подбор параметров здесь отсутствуют.

## Локальные артефакты

`data/processed/scaling/baseline_1000/` — ссылочный dataset_report, проверяющий SHA
исходного отчёта и всех исходных файлов. Исходный baseline не копируется/не меняется.

`sample_5000/`, `sample_10000/` — train tables/mappings/CSR, evaluation positives с
новыми cold flags, context CSR и user mappings, sampling_manifest.json (исходные IDs,
seed, стратегия, версия, source revision/size/SHA, config fingerprint), dataset_report
(качество/проверки/statistics/artifact hashes). Четыре raw evaluation context/target
файла доступны по абсолютным локальным ссылкам с SHA; не дублируются.
Эти ссылки привязаны к рабочей директории: для переноса нужен исходный baseline.

Сохранение расширения идёт во временный каталог; опубликованный каталог появляется
одним rename после completion report. При исключении временные файлы удаляются.
После аварийного завершения ОС может остаться непубликованный `.scaling-*` каталог;
он игнорируется загрузчиком. Запуски последовательные, concurrent writers не поддержаны.
Существующий эксперимент с тем же fingerprint проверяется и переиспользуется;
с другим — ошибка, без перезаписи.

## Запуск и интерфейс

Без сети, только проверки и план:

```powershell
.venv\Scripts\python.exe -m anime_recommender.training.scaling
```

Подготовка из уже проверенного cache, без права скачать отсутствующий источник:

```powershell
.venv\Scripts\python.exe -m anime_recommender.training.scaling --execute
```

**Только после отдельного разрешения**: один общий полный проход для ОБОИХ размеров:

```powershell
.venv\Scripts\python.exe -m anime_recommender.training.scaling --execute --allow-full-scan
```

Повторный `--execute` использует cache и готовые эксперименты. Не запускать отдельно
два ARE-004 acquire с размерами 5000 и 10000: это потребовало бы двух проходов.

```python
from anime_recommender.training.scaling import load_scaling_dataset, scaling_summary
data = load_scaling_dataset("data/processed/scaling/sample_5000")
matrix = data.matrix
items = data.tables["item_mapping"]
context = data.tables["validation_context"]
target = data.tables["validation_target"]
metadata = data.report
```

Интерфейс возвращает PreparedDataset, не модель. Старый load_training_inputs напрямую
не читает ссылочные scaling artifacts; ARE-008B должен использовать этот новый loader
или отдельный selective adapter (особенно чтобы не читать test при оценке).

## Источник, стоимость и resume

Immutable revision `6d0f78003440a90ca9a417389f13c47387f0c942`, ratings.csv:
2 254 824 347 байт, SHA-256 `2ee00f83262cd4cdb2d4a0f1661545307ccc956b01314df0d5e5232433339e68`.
Существующий verified cache содержит лишь 1000 полных историй (78421 raw rows,
75900 cleaned). Других историй локально нет; отсутствие user index не позволяет
гарантировать полноту без чтения всех 148170496 строк.

Нужен **один** новый проход; полный CSV не сохраняется. Чанк 250000 строк, в памяти
только текущие bottom-10000 истории и рабочие DataFrame. Это ограничение относительно
числа выбранных пользователей/их активности, а не жёсткий лимит байтов RAM.
При экстремально длинных историях память вырастет; переход к disk-backed sampling
не реализован и не маскируется обещанием постоянного потребления RAM.

Оценки, НЕ измерения полного нового запуска:

| Ресурс | 5k | 10k |
| --- | --- | --- |
| Clean rows при линейной экстраполяции текущих 75.9/user | ~380000 | ~759000 |
| RAM для отбора и подготовки | ориентир 1–3 ГБ для общего 10k прохода; зависит от активности/объектов pandas | тот же общий проход |
| Свободный диск для cache, двух наборов и временных файлов | рекомендуется 0.5–1 ГБ суммарно, без полного CSV | тот же бюджет |
| Новые train pool IDs до k-core | 4800 | 9800 |

Сеть 2.255 ГБ: теоретически ~3 мин при 100 Мбит/с или ~30 мин при 10 Мбит/с.
Разбор 148 млн строк и SHA selection на CPU может потребовать десятки минут
(плановый запас 10–60+ мин сверх сетевых задержек, не benchmark). Обучения нет.
Фактические train counts/vocabulary/warm coverage станут известны после загрузки;
не подменять их линейной экстраполяцией.

Checkpoint/resume: завершённый source cache проверяется по SHA и параметрам;
завершённые размеры переиспользуются, так что после сохранения 5k можно продолжить
10k локально. Mid-stream byte resume отсутствует: pandas buffer/CSV boundaries,
heap историй и состояние полного SHA должны восстанавливаться согласованно.
После разрыва частичная выборка не сертифицируется и потребуется новый проход
с отдельным разрешением; автоматических повторов сетевого чтения нет.

## Результат текущей проверки

Реальный baseline: train 772 × 2091, nnz=48080; 100 validation и 100 test users.
Notebook 07 выполнился на baseline/синтетике, 5 code cells. SHA всех 26 файлов
baseline ML dataset и ARE-006 модели до/после совпали. Новых реальных выборок нет.
Сохранённые outputs/execution counts пусты. Data/cache/manifests с IDs остаются
под игнорируемым `data/`; commit/push не выполняются.

Финальный полный запуск: `python -m pytest -q -p no:cacheprovider` — **187 passed**
(20.39 с), в том числе 10 новых scaling tests. Проверены также полный локальный
synthetic ingestion → ML baseline → scaling → повторный scaling, блокировка HTTP
без разрешения, несовместимые версии/параметры, изменение holdouts и повреждение SHA.
Список staged files пуст; сканирование файлов ARE-008A на значения локальных токенов
не выявило совпадений. Notebook 01 содержит прежнее пользовательское изменение и
не входит в ARE-008A.
