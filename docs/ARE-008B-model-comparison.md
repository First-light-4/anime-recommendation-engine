# ARE-008B — Fixed-validation model comparison and Notebook 05 switching

Дополнительные paired bootstrap CI, нормированная popularity/exposure и уточнённая
интерпретация выбора default: [ARE-008B.1](ARE-008B.1-final-model-selection.md).

Обучены отдельные ALS/Popularity 5k и 10k. Существующие 1k ALS/Popularity
переиспользованы: training fingerprint, mapping и конфигурация совпали.
Baseline artifacts не перезаписывались. Источник заново не читался, сеть и test
file access принудительно заблокированы в реальном запуске. Hyperparameter tuning нет.

## Совместимость и протокол

`training/inputs.py` теперь поддерживает проверенные baseline references и shared
validation artifacts. Reader проверяет SHA только train/validation файлов; test
артефакты не открываются. Fingerprint baseline train остался прежним.
При загрузке models дополнительно проверяется согласованность metadata и
отдельного `training_config.json`.

Перед первым fit проверены все три размера: одинаковый preparation config,
threshold=5, train minima=5/5, nested train users/items, фиксированная validation,
совместимость сохранённой baseline-модели с train fingerprint и гиперпараметрами.
Общий fingerprint исходных validation history/context/target без warm/cold flags:
`d01f4dfe46d178ab9a6c721a24e262de76c96e2939a31178b9ea65dda0c4b63e`.
Experiment dataset fingerprints различаются намеренно: меняются train и cold flags.

Одинаковые ALS settings: 64 factors, regularization=0.1, 20 iterations,
random_state=42, num_threads=1, BLAS threads=1, alpha=1, float32, use_cg=True,
cg_steps=3. Feedback бинарный, confidence=1.

Оценка: существующий ARE-007A evaluator, только 100 фиксированных validation users,
K=10, ALS-only и Popularity, без fallback и **без franchise post-processing**.
Внутри каждого размера модели имеют одинаковые candidates: train vocabulary
минус ВСЕ observed context items, включая низкие/нулевые оценки. Между размерами
candidate sets растут вместе с vocabulary — это часть сравнения масштаба данных.
Targets доступны только после генерации для метрик/is_hit.

Full Recall/NDCG включают все positive targets, включая cold. Warm Recall учитывает
targets из vocabulary соответствующей модели; это условный показатель с меняющейся
областью покрытия, не фиксированная подвыборка targets между версиями. Macro
aggregation и undefined semantics остаются ARE-007A. Основной критерий выбора
задан до запуска: Full NDCG@10, Full Recall@10 для разрешения равенства.

## Результаты

| Показатель | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| Train users | 772 | 4621 | 9455 |
| Train item vocabulary | 2091 | 5279 | 6719 |
| Train positives | 48080 | 359967 | 745354 |
| ALS Full Recall@10 | 0.283710 | 0.301667 | **0.305145** |
| ALS Full NDCG@10 | 0.388166 | 0.423164 | **0.425284** |
| ALS Warm Recall@10 | 0.291948 | 0.302479 | **0.305863** |
| Popularity Full Recall@10 | 0.212262 | 0.210726 | 0.214250 |
| Popularity Full NDCG@10 | 0.219222 | 0.218759 | 0.220517 |
| ALS Catalog Coverage@10 | 11.1430% | 4.9820% | 4.2119% |
| Distinct ALS recommended items | 233 | 263 | 283 |
| Popularity Catalog Coverage@10 | 1.8651% | 0.7577% | 0.5953% |
| Cold validation target share | 14.4993% | 2.0156% | 0.9753% |
| ALS fit time, seconds | 0.2410* | 1.1694 | 2.2366 |
| Popularity fit time, seconds | 0.0085* | 0.0436 | 0.0864 |

`*` Время baseline взято из сохранённой metadata предыдущего обучения: baseline
в этой задаче не переобучалась. Fit time исключает чтение, оценку и сериализацию;
это локальные измерения, не benchmark производительности.

| Изменение к 1k | 5k | 10k |
| --- | ---: | ---: |
| Full Recall@10, absolute | +0.017957 | +0.021435 |
| Full Recall@10, relative | +6.3293% | +7.5553% |
| Full NDCG@10, absolute | +0.034998 | +0.037117 |
| Full NDCG@10, relative | +9.0162% | +9.5623% |

Рекомендована **10k**: лучший point estimate обоих Full metrics. Преимущество над
5k небольшое: Recall +0.003478, NDCG +0.002120. Это 100 validation users без
confidence intervals; статистическая значимость не заявляется. Catalog Coverage
как доля падает из-за роста знаменателя, хотя число показанных различных items растёт.
Ниже приведён один диагностический пример, не дополнительный критерий выбора.
Default в config и Notebook оставлен **1k** до явного согласования пользователя.

## Артефакты и команды

- 1k: существующий `artifacts/models/are006/`, без изменений.
- 5k: `artifacts/models/are008b/5k/`.
- 10k: `artifacts/models/are008b/10k/`.
- Каждая модель: ALS NPZ, popularity/item/user Parquet, training config и metadata
  с mapping/train/file fingerprints. Новые metadata имеют явный model_version.
- Reports: `artifacts/evaluation/are008b/{1k,5k,10k}/`, comparison JSON/Parquet,
  validation_example.parquet, execution_checks.json и notebook_checks.json.
  Всё остаётся локально под gitignore. Старые ARE-007A reports не изменяются.

```powershell
# Проверка/оценка уже сохранённых моделей, без обучения:
.venv\Scripts\python.exe -m anime_recommender.training.scaling_experiments
# Явное обучение только отсутствующих 5k/10k; 1k никогда не переобучается:
.venv\Scripts\python.exe -m anime_recommender.training.scaling_experiments --train-new
```

Несовместимый или неполный existing bundle вызывает ошибку; overwrite/retrain
существующей модели команда не выполняет. Порядок размеров фиксирован как 1k→5k→10k,
независимо от порядка JSON keys; это покрыто integration regression test.

## Notebook 05

```python
model_version = "10k"  # "1k", "5k", "10k"
top_k = 15
```

Изменить model_version и выполнить load/Top-N cells. Указывать другой nickname
не требуется: введённое значение сохраняется, environment override по-прежнему
имеет приоритет. `ProfileSession` живёт в globals текущего kernel и не пересоздаётся
при повторном выполнении imports. Для одного nickname source profile запрашивается
один раз (profile/lists/favorites); история хранится до model-specific MAL mapping.
Переключение ремапит эту же source history к выбранному vocabulary. Изменение K
использует уже готовые signals и cached workspace, пересчитывая рекомендации.

Все version loads проверяют bundle SHA, mapping fingerprint и metadata/config;
новые версии также сверяют model_version. Отсутствующая модель — понятная ошибка,
без переключения на другую. Существующий Popularity fallback внутри выбранной
версии сохранён исключительно для отсутствия warm positives.

Один bounded lookup русских названий возможен только при первом Top-N display
nickname в сессии. При смене модели/K дополнительных metadata requests нет;
новые items без проверенного локального/кэшированного перевода показывают `—`.
Явное `refresh_profile=True` отдельно разрешает новое получение профиля; по
умолчанию запросы, включая неудачные, автоматически не повторяются.

Franchise post-processing в Notebook 05 сохранён, ALS scores не меняются.
Единый локальный mapping не обновлялся, discovery отсутствует. Coverage по полному
vocabulary выбранной модели (отдельно выводится candidate coverage выдачи):

| Версия | Items с известной франшизой | Доля vocabulary | Известные группы |
| --- | ---: | ---: | ---: |
| 1k | 22 / 2091 | 1.0521% | 3 |
| 5k | 26 / 5279 | 0.4925% | 3 |
| 10k | 27 / 6719 | 0.4018% | 3 |

Большинство франшиз неизвестно; им не присваиваются связи по названиям. Сохраняется
предупреждение о partial metadata и прежних API errors; online discovery нет.

## Проверки и ограничения

- **196 tests passed** (14.26 с), включая 7 новых model-version/integration tests.
- Synthetic integration подтверждает ровно 2 новых fit, сохранение baseline,
  повторный запуск без новых fit, неизменность файлов моделей и одинаковые outputs.
- Notebook 05: реальные три модели и синтетический профиль, 2.37 с;
  проверены Top-15/Top-7, all-seen exclusion, переключение всех версий, сохранность
  исходной history, один вызов каждого profile endpoint и initial title helper.
- Notebook 07: реальные train/validation и saved reports/plots, 3.41 с.
- При исполнении обоих Notebook network, fit и test reads запрещены; outputs и
  execution counts не сохранены. Есть служебные Windows Jupyter warnings без ошибок ячеек.
- Реальные YummyAnime API-запросы в ARE-008B не выполнялись; live profile здесь
  не использован и не сохранён. Поведение HTTP покрыто mocks существующих модулей.
- Fingerprints и baseline/franchise/Notebook 01 checksums подтверждены runtime guard.
  Секреты, модели, датасеты и evaluation artifacts остаются вне Git.
- Notebook 01 не изменён этой задачей. Commit/push ARE-008A/B не выполнялись.

## ???? validation-????????????: 13026

?????????? raw context ? targets, ??? franchise filtering. `is_hit` ???????? ????? ????????????. Scores ?????? ??????? ?? ??????????? ????? ?????.

### 1k ? hits@10: 5

| Rank | Title | ALS score | is_hit |
| ---: | --- | ---: | --- |
| 1 | Clannad | 0.747589 | True |
| 2 | OreImo | 0.684091 | True |
| 3 | Elfen Lied | 0.609099 | False |
| 4 | Erased | 0.581201 | False |
| 5 | Bakemonogatari | 0.553329 | True |
| 6 | Fullmetal Alchemist: Brotherhood | 0.531247 | False |
| 7 | Code Geass: Lelouch of the Rebellion | 0.485660 | True |
| 8 | High School of the Dead | 0.485646 | False |
| 9 | Hyouka | 0.473591 | True |
| 10 | Charlotte | 0.468032 | False |

### 5k ? hits@10: 7

| Rank | Title | ALS score | is_hit |
| ---: | --- | ---: | --- |
| 1 | Hyouka | 0.739011 | True |
| 2 | The Pet Girl of Sakurasou | 0.674061 | True |
| 3 | Bakemonogatari | 0.625820 | True |
| 4 | The Devil is a Part-Timer! | 0.594856 | True |
| 5 | Angel Beats! | 0.591322 | False |
| 6 | OreImo | 0.590228 | True |
| 7 | Beyond the Boundary | 0.559146 | False |
| 8 | Terror in Resonance | 0.550876 | True |
| 9 | Clannad | 0.536300 | True |
| 10 | Anohana: The Flower We Saw That Day | 0.521241 | False |

### 10k ? hits@10: 7

| Rank | Title | ALS score | is_hit |
| ---: | --- | ---: | --- |
| 1 | The Pet Girl of Sakurasou | 0.749088 | True |
| 2 | Hyouka | 0.718412 | True |
| 3 | Bakemonogatari | 0.715202 | True |
| 4 | The Devil is a Part-Timer! | 0.698330 | True |
| 5 | OreImo | 0.692322 | True |
| 6 | From the New World | 0.570463 | True |
| 7 | Angel Beats! | 0.557260 | False |
| 8 | The World God Only Knows | 0.552373 | False |
| 9 | Terror in Resonance | 0.541183 | True |
| 10 | Neon Genesis Evangelion: The End of Evangelion | 0.529850 | False |
