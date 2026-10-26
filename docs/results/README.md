# Эталонные результаты сетки экспериментов

Здесь лежат копии `leaderboard.md` сетки экспериментов, на которые ссылается корневой README.
Каталог `reports/` в git не попадает (это черновики прогонов), поэтому опорные цифры хранятся отдельно.

| Файл | Что внутри | Как получен |
|---|---|---|
| [grid_offline.md](grid_offline.md) | офлайн-лесенка: BM25 (окна → разделы → заголовки → составные токены), LSA, гибриды | `kbrag grid --only bm25_fixed bm25_recursive_noheader bm25_recursive bm25_recursive_split_mixed lsa_dense hybrid_bm25_lsa hybrid_bm25_lsa_weighted --out reports/grid_offline` |
| [grid_neural.md](grid_neural.md) | вся сетка без тяжёлых моделей: офлайн-шаги + e5, гибрид с e5, реранкер, размер фрагмента | `kbrag grid --out reports/grid_baseline` (нужен `make install-neural`) |

Условия прогона: эталон из 237 вопросов (test: 131 отвечаемый + 35 неотвечаемых), корпус из 48 документов
(`corpus_sha256` be5fe72bcb47f3a6, `gold_sha256` f2c54f0f3e8a96ef — те же хэши стоят в `provenance` каждого
эксперимента в `results.json`), модели e5 и реранкера — ревизии из `configs/default.yaml`, CPU, Windows 11,
Python 3.12, код коммита 1ac846b (`environment.git_sha` в `results.json`).

Как читать:

- **Столбцов задержки здесь нет**: p50/p95 зависят от машины и её загрузки (один прогон без прогрева), поэтому
  для эталона они бесполезны. Задержку смотрите в своём прогоне (`reports/grid/leaderboard.md`).
- Это **описательная абляция на test**. Боевые конфиги (`configs/offline.yaml`, `configs/default.yaml`) зафиксированы
  заранее, а не выбраны по этим таблицам; единственный подобранный параметр (`lsa_dim`) выбран на dev.
- Один отвечаемый test-вопрос ≈ 0.008 Recall@5. Строки, у которых в разделе «Отличие от лидера» стоит «≈»,
  по этим данным неотличимы.
- Легенда метрик — в конце каждого файла.
