# Боевые конфигурации: сквозные прогоны

Здесь — оценка конфигураций в том виде, в каком их запускает сервис: `configs/offline.yaml`,
`configs/default.yaml` (с классификатором отказа, как сейчас в сервисе, и с прежним порогом) и
`configs/default.yaml` с LLM вместо extractive-генератора.

Чем это отличается от опорной сетки ([grid_offline.md](grid_offline.md), [grid_neural.md](grid_neural.md)):

- **Отказ — сервисный.** Строки сетки наследуют `abstention.score_source: auto`, поэтому нейро-строка
  `hybrid_bm25_e5_rerank` ставит порог по score реранкера. Сервис с `configs/default.yaml` с коммита `5f5080c`
  отказывается по классификатору отказа (`abstention.method: classifier`, модель
  `reports/abstention_default.json`), а до него брал уверенность из косинуса e5 (`score_source: dense`). Поэтому
  цифры отказов и ответов здесь свои.
- **Есть задержка.** Она измерена на конкретном железе (условия ниже), поэтому годится только как порядок величины.
- **Есть LLM-режим.** В сетке все строки используют extractive-генератор.

## Метрики на test

Test: 166 вопросов, из них 131 отвечаемый и 35 неотвечаемых.

| Конфигурация | Recall@5 [95% ДИ] | MRR | FactRecall (отвеч.) | FactRecall | CitationPrecision | Архивные ссылки | Abstention F1 | FAR | Порог | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline: BM25 + LSA, extractive | 0.837 [0.774–0.898] | 0.755 | 0.802 | 0.594 | 0.627 | 0.010 | 0.449 | 0.260 | dense: 0.603 | 14.7 / 17.1 мс |
| default: BM25 + e5 + реранкер, extractive, классификатор отказа | 0.903 [0.854–0.948] | 0.843 | 0.699 | 0.667 | 0.548 | 0.008 | 0.677 | 0.046 | classifier:catboost: 0.619 | 2.67 / 3.48 с |
| default, extractive, порог (до 5f5080c) | 0.903 [0.854–0.948] | 0.843 | 0.806 | 0.523 | 0.653 | 0.012 | 0.541 | 0.351 | dense: 0.871 | 2.57 / 3.27 с |
| default + Qwen2.5-1.5B-Instruct, 4 бита, порог (до 5f5080c) | 0.903 [0.854–0.948] | 0.843 | 0.698 | 0.453 | 0.324 | 0.024 | 0.541 | 0.351 | dense: 0.871 | 15.5 / 30.8 с |

У строки с классификатором шкала порога другая: это P(нет ответа), и правило обратное — **отказ, если p ≥ 0.619**.
Порог взят из файла модели, где его подобрали на dev при `kbrag train-abstention`.

Отказы подробнее:

| Конфигурация | AUROC [95% ДИ] | Abstention precision | Abstention recall |
|---|---:|---:|---:|
| offline (уверенность dense) | 0.714 [0.627; 0.797] | 0.370 | 0.571 |
| default, классификатор (уверенность = 1 − P(нет ответа)) | 0.871 [0.784; 0.949] | 0.778 | 0.600 |
| default, порог (обе строки с порогом; уверенность dense) | 0.796 [0.714; 0.875] | 0.395 | 0.857 |

## Как читать

- **Классификатор отказа против порога (нейро).** Поиск у строк один и тот же, разница только в решении «отвечать
  или нет». Классификатор зря отказывает 6 раз из 131 вместо 46 и ловит 21 из 35 вопросов без ответа вместо 30:
  рабочая точка сдвинута к «отвечать». Сквозной FactRecall растёт с 0.523 до 0.667. FactRecall (отвеч.) и
  CitationPrecision ниже из-за отбора, а не из-за генерации: на 85 отвечаемых вопросах, где ответили оба метода,
  обе метрики одинаковы (FactRecall 0.806, CitationPrecision 0.653), а классификатор ответил ещё на 40, из них 25
  перефразов, где поиск слабее (на этих 40 FactRecall 0.471, CitationPrecision 0.325). Эти числа посчитаны по
  `per_question` прогонов `reports/abstention_cmp_neural/default_threshold` и `reports/abstention_cmp_neural_grid`
  (сниппет — [ONBOARDING.md](../../ONBOARDING.md), раздел 9.8).
  Переключение сделано по правилу, записанному до обучения (коммит `8029af0` в `docs/DESIGN.md`), и по dev; test —
  только отчёт. Парные разности с интервалами и офлайн-эксперимент, где классификатор порог не обогнал, — в
  [abstention.md](abstention.md).
- **Две нижние строки различаются только генератором.** Поиск и порог у них общие, поэтому они ответили на одни и
  те же 85 отвечаемых test-вопросов. Парная разница LLM минус extractive на этих вопросах (95% интервал, парный
  bootstrap): FactRecall −0.108 [−0.204; −0.008], CitationPrecision −0.329 [−0.431; −0.229]. Модель на 1.5 млрд
  параметров уступает extractive-бейзлайну и по фактам, и особенно по ссылкам: в 49 из 85 ответов нет ни одной
  ссылки `[n]`, в 19 скопирована служебная шапка фрагмента, 10 ответов упёрлись в `max_tokens`. Для LLM-режима нужна
  модель 7B и больше (vLLM, профиль `llm` в `docker-compose.yml`); такой замер ещё не сделан. С классификатором
  отказа LLM-режим не измерялся.
- **FactRecall (отвеч.) между offline и default сравнивайте осторожно.** Он считается только по отвеченным вопросам,
  а пороги у конфигов разные, поэтому разные и наборы отвеченных вопросов. На 80 вопросах, где ответили оба
  (offline и default с порогом), FactRecall 0.806 (offline) против 0.812 (default): генератор один и тот же.
- **Отказы по порогу — слабое место.** Больше половины отказов ошибочны (abstention precision 0.37 и 0.39), а FAR
  0.26 и 0.35 значит, что каждый четвёртый (offline) или третий (default с порогом) вопрос с ответом в базе получает
  отказ. Классификатор в нейроконфиге это исправил (precision 0.78, FAR 0.046), но 14 из 35 вопросов без ответа
  всё равно получают «ответ». Офлайн-конфиг остался на пороге.
- **Метрики ответа на test слегка оптимистичны.** Test использовался для разбора ошибок: правила extractive-генератора
  про пункты списка и архивные редакции и правка корпуса для x-009 появились после разбора test-вопросов
  (подробно — [ONBOARDING.md](../../ONBOARDING.md), раздел 7.4).

## Условия прогонов

Общее для всех: эталонный набор из 237 вопросов (`gold_sha256` f2c54f0f3e8a96ef), корпус из 48 документов
(`corpus_sha256` be5fe72bcb47f3a6), Windows 11, Python 3.12, поиск на CPU AMD Ryzen 5 2600X (torch CPU), те же
снапшоты e5 и реранкера, что закреплены в `configs/default.yaml`. Каталог `reports/` в git не попадает, поэтому
таблица выше перенесена сюда вручную из `leaderboard.md` каждого отчёта.

| | offline | default, классификатор | default, extractive, порог | default + Qwen2.5-1.5B, порог |
|---|---|---|---|---|
| Отчёт | `reports/_check/offline_check/` | `reports/abstention_cmp_neural_grid/` (шаг сетки `hybrid_bm25_e5_rerank_clf`) | `reports/neural/` | `reports/llm/neural_llm_qwen1.5b/` |
| Код по `results.json` | `git_sha` 92f88f9, `git_dirty: true` | `git_sha` 8029af0, `git_dirty: false` | `git_sha` 1e7bcc3, `git_dirty: true` | `git_sha` 35d77d8, `git_dirty: true` |
| Что было не закоммичено | правило про архивные редакции, позже закоммиченное как 1e7bcc3: все метрики совпадают со строкой `hybrid_bm25_lsa` опорной сетки на чистом 98b7adc | ничего из отслеживаемых файлов; модель `reports/abstention_default.json` (+ `.cbm`) закоммичена следом, в 00a0579 | переход `default.yaml` на dense-уверенность и новый порог, позже закоммиченные как 98b7adc: в сообщении коммита те же FAR 0.351 и Abstention F1 0.541 | незакоммиченные правки отслеживаемых файлов (`git_dirty` не говорит, каких) |
| Генератор | extractive | extractive | extractive | `KBRAG_GENERATION_BACKEND=openai`; [scripts/serve_local_llm.py](../../scripts/serve_local_llm.py) `--quantize 4bit` (bitsandbytes, nf4) на GPU NVIDIA GTX 1650 Super; `KBRAG_LLM_TIMEOUT_S=180`, `KBRAG_LLM_MAX_TOKENS=256`; ошибок LLM — 0 |

Шаг сетки `hybrid_bm25_e5_rerank_clf` совпадает с `configs/default.yaml` по отпечатку поиска (это проверяет
`tests/test_configs.py`), а отказ в нём решает та же модель, что в сервисе, поэтому источник уверенности сетки
(`auto`) на его отказы не влияет.

Строку «default, extractive, порог» подтверждает повторный прогон на чистом коммите 8029af0
(`reports/abstention_cmp_neural/default_threshold`, `git_dirty: false`; на этом коммите `configs/default.yaml` ещё
отказывался по порогу): Recall@5, MRR, все метрики ответа, Abstention F1, FAR и AUROC dense те же, p50 2.65 с. На
чистом коммите, таким образом, оба нейро-прогона с extractive-генератором; офлайн- и LLM-прогоны — нет, и их
стоит перезапустить на одном чистом коммите.

## Как повторить

Без `--save-calibration`, чтобы не перезаписать закоммиченный порог отказа:

```bash
kbrag eval --config configs/offline.yaml --name offline
kbrag eval --config configs/default.yaml --name neural            # нужен make install-neural; отказ — классификатор
kbrag grid --only hybrid_bm25_e5_rerank_clf --out reports/abstention_cmp_neural_grid   # то же шагом сетки

# Строки «порог»: тот же конфиг с отказом по порогу. Повторный прогон default_threshold сделан на коммите 8029af0,
# где default.yaml ещё стоял на пороге; сейчас то же даёт переменная окружения поверх конфига.
KBRAG_ABSTENTION_METHOD=threshold kbrag eval --config configs/default.yaml --name default_threshold --out reports/abstention_cmp_neural

# LLM: сервер в отдельном окружении .venv-llm с CUDA-сборкой torch (ONBOARDING.md, раздел 6.4)
python scripts/serve_local_llm.py --model Qwen/Qwen2.5-1.5B-Instruct --quantize 4bit --port 8001
KBRAG_GENERATION_BACKEND=openai KBRAG_LLM_BASE_URL=http://127.0.0.1:8001/v1 \
KBRAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct KBRAG_LLM_TIMEOUT_S=180 KBRAG_LLM_MAX_TOKENS=256 \
  kbrag eval --config configs/default.yaml --name neural_llm_qwen1.5b --out reports/llm
```

Задержка зависит от машины и её загрузки (один проход без прогрева): сравнивайте порядок величин.
