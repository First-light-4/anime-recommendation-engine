# ARE-008B.1 — Final evaluation and model selection

Анализ выполнен по **сохранённым** ARE-008B per-user validation results и Top-10.
Рекомендации и весь эксперимент заново не вычислялись; модели не обучались,
train/validation/test, ALS settings и алгоритм рекомендаций не менялись.
Сеть, fit, ingestion и test reads заблокированы в runtime checks.

## Метод и достоверность сопоставления

Для 1k/5k/10k проверены SHA report tables, model/train fingerprints, одинаковый
protocol version, validation-only scope и выключенный franchise filtering.
Fixed-validation fingerprint совпадает:
`d01f4dfe46d178ab9a6c721a24e262de76c96e2939a31178b9ea65dda0c4b63e`.
Модельные content fingerprints пересчитаны из сохранённых bundles и сверены с
reports. Проверены одинаковые user IDs, positive target counts, отсутствие
duplicate users и совпадение macro metrics с mean per-user results.

Paired bootstrap: **100 users**, **10000 resamples**, **seed=42**, NumPy default_rng.
User IDs сортируются; один и тот же набор draws применяется ко всем шести
сравнениям. В каждом draw берётся mean(candidate_user_metric-reference_user_metric).
95% interval — percentile 2.5/97.5% распределения этой разницы. Relative delta —
разница macro means / reference macro mean, не среднее индивидуальных процентов.

Undefined metrics не заменяются нулями: обе модели должны иметь одинаковую маску,
а такие users исключаются с явным счётчиком. Здесь у всех 100 users Full metrics
определены; excluded=0. Improved/worsened/unchanged считаются по пользователям;
unchanged означает abs(delta)<=1e-12 только для числовой классификации. В bootstrap
идут исходные, не округлённые разности.

Это **pointwise**, не simultaneous confidence intervals: correction для шести
сравнений не применялась. Bootstrap условен на конкретных fitted models и этих
validation users; он не отражает вариацию train sample, training seed, другой
популяции или повторного model selection. IID user bootstrap — рабочее допущение.
Интервал, включающий ноль, не доказывает равенство или эквивалентность моделей.

## Интерпретация 5k и 10k и рекомендация

У 10k выше point estimates: Recall +0.003478 (+1.153%), NDCG +0.002120 (+0.501%)
к 5k. Но оба paired CI включают отрицательные и положительные эффекты; уверенный
порядок качества между 5k и 10k на этих 100 users не установлен. NDCG против 1k
имеет положительный pointwise 95% interval у обеих моделей, а Recall — нет.
Особенно нижний предел NDCG 10k-vs-1k близок к нулю; это не основание заявлять
устойчивую победу во всех сценариях или после multiple-comparison correction.

Vocabulary растёт 5279→6719 (+1440, +27.28%); cold validation targets сокращаются
31→15. На фиксированных 1000 рекомендованных позициях уникальных items 263→283.
Но Catalog Coverage относительно своего vocabulary снижается 4.982%→4.212%,
а top-10%-концентрация растёт 98%→99%. Это не доказательство улучшения long-tail
diversity. Нет измерений genre/semantic intra-list diversity или novelty.

Fit time растёт 1.169→2.237 с, около 1.91 раза, абсолютная разница 1.067 с.
Это прежние сохранённые timings, не новый benchmark; inference latency здесь не
измерялась. Нельзя утверждать практическую равнозначность serving costs.

**Рекомендую 10k как product default после согласования**: более широкий доступный
каталог, меньше cold targets, больше абсолютных unique recommendations и лучшие
точечные Full metrics при небольшом абсолютном fit time. Это решение по trade-off,
а не статистически доказанное превосходство над 5k. Если важнее минимальный размер
модели/затраты, 5k остаётся обоснованной альтернативой; данных для объявления
10k обязательным победителем недостаточно. Улучшение diversity не заявляется.

Конфигурация и Notebook 05 **остались на 1k**. Никакого автоматического изменения
default не выполнено. Для более уверенного ранжирования 5k/10k потребуется
отдельно согласованная более крупная независимая validation, не чтение test.

## Нормировка популярности

Используются unique positive train-user counts из сохранённой Popularity модели.
Средние взвешены по рекомендованным позициям: повторение item разным пользователям
учитывается. Всего 1000 позиций для каждой модели, одинаковый K=10.

- Raw popularity — средний unique train-user count.
- Нормированная popularity — тот же count / число retained train users модели.
- Percentile — позиция item в полном vocabulary своей модели, а не среди выдачи.
  Сортировка popularity descending / dataset ID ascending; percentile=
  100*(N-zero_based_rank)/N. 100 означает наиболее популярный item.
- Top-10% — ровно ceil(0.1*N) items; ties на границе разбиваются по ID, не по
  предпочтению, названию или validation. Counts: 210/528/672; на границе 6/9/6 ties.
  Cutoff popularity: 48/154/253 users. Это детерминированная rank-конвенция.

С ростом train средняя доля train users у рекомендованных items падает, но их
percentile относительно расширенного vocabulary растёт. Метрики имеют разные
референтные множества: ни сырые counts, ни percentile в одиночку не доказывают
улучшения разнообразия. Catalog Coverage использует train vocabulary, не весь
исходный каталог из 20237 аниме.

## Notebook 05 и замороженные артефакты

Выполнен настоящий Notebook 05 с **реальными saved models** и synthetic profile,
затем проверены переходы **1k → 10k → 5k → 1k**. Сеть, fit и source acquisition
принудительно запрещены; test files не читаются. Проверены:

- Vocabulary 2091 / 6719 / 5279 / 2091, metadata model_version и mapping fingerprints.
- По 15 рекомендаций; отсутствие всех known/seen items в каждой выдаче.
- Один mock-вызов каждого profile/lists/favorites на всю последовательность,
  неизменность исходной history, без refetch при switching.
- Franchise filtering включён, предупреждение о partial metadata присутствует;
  online franchise discovery не выполняется, mapping не обновлён.
- Default model_version="1k", top_k=15. Outputs и execution counts не сохранены.
- SHA **66 защищённых файлов** (models, исходные evaluation reports, franchise
  metadata, Notebook 01/05 и registry) остались неизменными.

Реальный YummyAnime-профиль не загружался и не сохранялся. Вновь проверялся только
синтетический сценарий без сети, а не доступность внешнего API.

## Воспроизводимость и файлы

```powershell
.venv\Scripts\python.exe -m anime_recommender.evaluation.model_selection --resamples 10000 --seed 42
```

Команда читает сохранённые reports/model bundles; генерации рекомендаций, обучения,
чтения train interaction datasets или API calls в ней нет. Результаты только в
игнорируемом `artifacts/evaluation/are008b/final_selection/`: bootstrap.parquet,
diversity.parquet, popularity_distribution.parquet, report.json, switch_checks.json.
JSON содержит sources/fingerprints, формулировку протокола и checksums таблиц.
Исходные ARE-008B reports не перезаписываются.

Изменены в ARE-008B.1: новый `evaluation/model_selection.py`, новые
`tests/test_model_selection.py`, порядок переходов в `tests/test_model_versions.py`,
этот отчёт, README и ссылка на финальный audit в предыдущем отчёте ARE-008B.
Notebook 05/07, registry, модели и datasets в ARE-008B.1 не изменялись.
Commit/push ARE-008A/B не выполнялись.

Финальный полный запуск: **201 tests passed** (13.96 с). Из них 5 новых tests
проверяют paired alignment/детерминизм, undefined/zero baseline, взвешивание
exposure, нормировку, ties, source certificates и согласованность macro metrics.
Существующий model-switch integration test теперь проверяет требуемый порядок
1k→10k→5k→1k. Тесты работают offline, на synthetic fixtures.

## Paired bootstrap: ????????? ??????????

? ? CI ??????? ? ?????????? ???????? ???????; relative ? ? win/loss/tie ? ????????. ??? ??????: n=100, undefined=0.

| Comparison | Metric | ? absolute | ? relative | 95% CI | Improved / worsened / unchanged |
| --- | --- | ---: | ---: | --- | --- |
| 5k vs 1k | Full recall@10 | +0.017957 | +6.329% | [-0.020189; +0.054477] | 35% / 13% / 52% |
| 5k vs 1k | Full ndcg@10 | +0.034998 | +9.016% | [+0.002441; +0.067541] | 50% / 26% / 24% |
| 10k vs 1k | Full recall@10 | +0.021435 | +7.555% | [-0.019511; +0.061135] | 39% / 15% / 46% |
| 10k vs 1k | Full ndcg@10 | +0.037117 | +9.562% | [+0.000697; +0.073897] | 48% / 26% / 26% |
| 10k vs 5k | Full recall@10 | +0.003478 | +1.153% | [-0.020856; +0.028158] | 21% / 15% / 64% |
| 10k vs 5k | Full ndcg@10 | +0.002120 | +0.501% | [-0.016071; +0.019663] | 40% / 32% / 28% |

## Exposure ? coverage

| ?????????? | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| Catalog Coverage@10, % | 11.143 | 4.982 | 4.212 |
| Unique Recommended Items@10 | 233 | 263 | 283 |
| Mean unique train-user count | 160.564 | 862.041 | 1625.724 |
| Mean train-user share, % | 20.798 | 18.655 | 17.194 |
| Mean vocabulary popularity percentile | 96.673 | 98.177 | 98.303 |
| Recommendations in top-10% items, % | 93.1 | 98.0 | 99.0 |

## ????????????? ???????????? ?? popularity percentile

100 ? ????? ?????????? item. ? ?????? count / share, ????? 1000 ??????? ?? ??????. Percentile bins ??????????????: ???????? (90,100].

| Percentile bin | 1k | 5k | 10k |
| --- | ---: | ---: | ---: |
| 0-10 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 10-20 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 20-30 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 30-40 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 40-50 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 50-60 | 0 / 0.0% | 0 / 0.0% | 0 / 0.0% |
| 60-70 | 1 / 0.1% | 0 / 0.0% | 0 / 0.0% |
| 70-80 | 11 / 1.1% | 2 / 0.2% | 1 / 0.1% |
| 80-90 | 57 / 5.7% | 18 / 1.8% | 9 / 0.9% |
| 90-100 | 931 / 93.1% | 980 / 98.0% | 990 / 99.0% |

Notebook 05 runtime check: 2.45 ?, ?????????????????? 1k?10k?5k?1k. ??????????? outputs ???.

## ?????????? ARE-008

????????? default ? 10k. Source code, ????????????, ????????? Notebook ? ??????
??????????? ??? ???????, ?????? ? cache.
[????????? artifacts ? ??????????? ?????????????????](ARE-008-artifacts-and-reproduction.md).
