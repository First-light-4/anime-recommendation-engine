# ARE-008A — Результат контролируемого полного прохода

За один разрешённый проход получены реальные 5000/10000 пользователей.
Seed=42, threshold=5, binary confidence=1, train k-core=5/5 сохранены.
Новые модели не обучались; оценка качества на test не выполнялась.

## Данные

| Показатель | Baseline | 5k | 10k |
| --- | ---: | ---: | ---: |
| Selected raw users | 1000 | 5000 | 10000 |
| Train pool users до фильтрации | 800 | 4800 | 9800 |
| Train users после фильтрации | 772 | 4621 | 9455 |
| Clean interactions, все cohorts | 75900 | 413507 | 834165 |
| Positive interactions, все cohorts | 70031 | 384219 | 771689 |
| Train positives до фильтрации | 54414 | 368602 | 756072 |
| Train positives после фильтрации / CSR nnz | 48080 | 359967 | 745354 |
| Train vocabulary | 2091 | 5279 | 6719 |
| Train CSR | 772 × 2091 | 4621 × 5279 | 9455 × 6719 |
| Train sparsity | 97.0215% | 98.5244% | 98.8267% |
| Validation users | 100 | 100 | 100 |
| Test users | 100 | 100 | 100 |

Общие positive counts включают context/target evaluation, а train nnz — только
train после k-core. Не интерпретировать рост coverage как измеренное улучшение ALS.

## Context CSR

| Матрица | Baseline: shape / nnz | 5k: shape / nnz | 10k: shape / nnz |
| --- | --- | --- | --- |
| Validation context | 100 × 2091 / 5126 | 100 × 5279 / 5829 | 100 × 6719 / 5918 |
| Test context | 100 × 2091 / 5608 | 100 × 5279 / 6351 | 100 × 6719 / 6401 |

В CSR входят только warm positive context. Targets остаются таблицами,
никогда не передаются в построение train vocabulary или context CSR.

## Покрытие

В каждой ячейке — **warm / cold**, проценты от положительных взаимодействий:

| Cohort / часть | Baseline | 5k | 10k |
| --- | --- | --- | --- |
| Validation context, 5990 строк | 85.576 / 14.424% | 97.312 / 2.688% | 98.798 / 1.202% |
| Validation targets, 1538 строк | 85.501 / 14.499% | 97.984 / 2.016% | 99.025 / 0.975% |
| Test context, 6443 строки | 87.040 / 12.960% | 98.572 / 1.428% | 99.348 / 0.652% |
| Test targets, 1646 строк | 86.695 / 13.305% | 98.663 / 1.337% | 99.514 / 0.486% |

Cold validation target counts: 223 → 31 → 15; test: 219 → 22 → 8.
Идентичность строк и исходных рейтингов context/target не менялась.
Локально сохранённой истории YummyAnime-профиля не найдено. Его coverage не
оценивалось; запросы к YummyAnime и сохранение приватного профиля не выполнялись.
Новые train vocabularies ещё не являются vocabulary обученных ALS-моделей.

## Очистка и баланс

| Показатель | Baseline | 5k | 10k |
| --- | ---: | ---: | ---: |
| Raw sample rows | 78421 | 423841 | 859159 |
| Exact duplicate rows removed | 2491 | 9582 | 23938 |
| Conflicting pair rows removed | 30 | 752 | 1056 |
| Removed total | 2521 | 10334 | 24994 |
| Retained clean rows | 75900 | 413507 | 834165 |
| Нулевые рейтинги сохранены | 50 | 379 | 685 |
| Train positives removed by activity/k-core | 6334 | 8635 | 10718 |

Для всех размеров raw = clean + removed; train positives до фильтрации = train
после фильтрации + activity exclusions + k-core exclusions. Пропущенных/невалидных
рейтингов и unmapped anime в выбранных историях нет. Нулевые рейтинги не positive.

## Источник и ресурсы

- Источник только `mramazan/User-Animelist-Dataset`, immutable revision
  `6d0f78003440a90ca9a417389f13c47387f0c942`.
- Прочитано 148170496 строк, **один новый полный проход**.
- Ratings payload: 2254824347 байт; SHA-256
  `2ee00f83262cd4cdb2d4a0f1661545307ccc956b01314df0d5e5232433339e68` — совпал.
- Каталог: 10816239 байт; суммарный проверенный payload **2265640586 байт**.
  HTTP/TLS overhead не измерялся.
- Полный запуск с подготовкой обеих выборок: **569.66 с**, около 9 мин 30 с.
- Свободно перед запуском: 177.97 ГиБ.
- Cache: 12871328 байт; baseline reference: 7880 байт; 5k artifacts: 2513333 байт;
  10k artifacts: 4879520 байт. Суммарно 20272061 байт (~19.33 МиБ), без небольших
  служебных журналов и итоговых audit JSON. Полный CSV не сохранён.
- Достоверный пик RAM рабочего процесса **не измерен**: Windows venv launcher
  запускал отдельный Python worker, а монитор измерял launcher. Его значение
  4407296 байт не выдаётся за расход pipeline; telemetry исправлена на null с причиной.
  Промежуточный OS snapshot показал worker PeakWorkingSet=330260480 байт (~315 МиБ);
  это не максимум полного запуска. Дополнительный проход ради замера не выполнялся.

## Повторное использование и проверка целостности

Повторный `run_scaling(execute=True, allow_full_scan=False)` выполнен с mock,
запрещающим любые HTTP-запросы. Успешно, **0 запросов**, 26.65 с с проверкой
контрольных сумм. SHA и mtime всех cache/experiment файлов не изменились.

Проверки `evaluation/scaling_validation.py` подтверждают:

- baseline train ⊆ 5k train ⊆ 10k train, включая сохранение пар взаимодействий;
- baseline assignments, evaluation users и context/target фиксированы;
- train/validation/test users disjoint, новые IDs только в train;
- train-only vocabulary, корректные CSR coordinates и warm/cold flags;
- правильный баланс очистки, positive feedback и train фильтрации;
- неизменность 32 защищённых файлов ARE-004/005/006 и Notebook 01.

Notebook 07 выполняет только локальное исследование через функции `src/`.
Для проверки исполнения сеть и `ALS.fit` принудительно блокируются;
outputs/execution counts в сохраняемом Notebook пусты.
Выполнены все 6 code cells и дополнительные проверочные assertions за 11.26 с.
При закрытии Windows Jupyter kernel выведено служебное сообщение libzmq
`Connection reset by peer [10054]`; ячейки завершились без ошибок, процесс проверки
вернул код 0. Новые зависимости или изменения инфраструктуры не требовались.

До загрузки: **187 passed** (20.12 с). После новых проверок и завершения загрузки:
**189 passed** (20.83 с), включая 12 scaling tests. Тесты не используют сеть.
`git diff --check` проходит; staged files пусты. Notebook 01 сохранил прежнее
пользовательское изменение, в задаче он не редактировался.

Локальные доказательства находятся под игнорируемым `data/processed/scaling/`:
`preflight_audit.json`, `controlled_run.log`, `controlled_run_resources.json`,
`aggregate_audit.json`, `local_verification.json`, `notebook_verification.json`.
Sampling manifests содержат исходные dataset user IDs и остаются только локально.
Commit/push не выполнялись; ARE-008B не начата.
