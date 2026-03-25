# LLMAI v3.0 — Документация

Платформа пакетной обработки контента через Novita AI и WD Tagger.

---

## 1. Обзор

**LLMAI v3.0** — веб-приложение для массовой обработки текстового и визуального контента:
- Массовый рерайт текстов с множителем x1–x10
- Перевод на 8 языков
- Постобработка с системой стоп-слов
- AI Process 3.0: анализ изображений → маппинг тегов → SEO-генерация

### Стек

| Слой | Технология |
|------|-----------|
| Frontend | Next.js 16 + TypeScript + shadcn/ui + Tailwind CSS |
| Backend | Next.js API Routes + BullMQ |
| БД | PostgreSQL 16 (Prisma ORM) |
| Кеш/Очередь | Redis 7 |
| AI (LLM) | Novita AI API (OpenAI-совместимый) |
| AI (Vision) | WD Tagger (HuggingFace Gradio API) |
| Деплой | Docker Compose + GitHub Actions → Timeweb |

---

## 2. Быстрый старт

```bash
# 1. Клонировать
git clone https://github.com/AlexGolubevpm/llmai.git
cd llmai/llmai-v3

# 2. Настроить
cp .env.example .env
# Отредактировать .env: вписать NOVITA_API_KEY и DB_PASSWORD

# 3. Запустить
docker-compose -f docker-compose.prod.yml up -d --build

# Приложение на http://localhost:3000
```

---

## 3. Архитектура

```
llmai-v3/
├── prisma/schema.prisma          # БД модели
├── src/
│   ├── app/                      # Next.js App Router
│   │   ├── page.tsx              # Dashboard
│   │   ├── rewrite/page.tsx      # Массовый рерайт
│   │   ├── translate/page.tsx    # Перевод
│   │   ├── postprocess/page.tsx  # Постобработка
│   │   ├── ai-process/page.tsx   # AI Process 3.0
│   │   ├── stopwords/page.tsx    # Стоп-слова
│   │   ├── tags/page.tsx         # Теги & Категории
│   │   ├── settings/page.tsx     # Пресеты
│   │   └── api/                  # REST API
│   ├── lib/                      # Клиенты и утилиты
│   │   ├── novita-client.ts      # Novita AI (realtime)
│   │   ├── novita-batch-client.ts # Novita AI (batch)
│   │   ├── wd-tagger-client.ts   # WD Tagger
│   │   ├── text-processing.ts    # Очистка текста
│   │   ├── queue.ts              # BullMQ очереди
│   │   ├── db.ts                 # Prisma
│   │   └── redis.ts              # Redis
│   ├── workers/                  # Фоновые воркеры
│   │   ├── rewrite.worker.ts
│   │   ├── translate.worker.ts
│   │   ├── postprocess.worker.ts
│   │   └── ai-process.worker.ts
│   ├── components/               # UI компоненты
│   └── types/index.ts            # TypeScript типы
├── docker-compose.prod.yml       # Продакшн (4 контейнера)
├── Dockerfile                    # Next.js app
├── Dockerfile.worker             # BullMQ worker
└── .github/workflows/deploy.yml  # CI/CD
```

### Поток данных

```
Пользователь загружает CSV
        ↓
POST /api/files/upload → сохраняет в ./uploads/
        ↓
POST /api/jobs → создаёт Job в БД + добавляет в BullMQ очередь
        ↓
Worker берёт задачу → обрабатывает строки → публикует прогресс в Redis
        ↓
GET /api/jobs/:id/stream (SSE) → фронтенд показывает прогресс в реальном времени
        ↓
Worker завершает → сохраняет результат в ./results/
        ↓
GET /api/files/:id → скачивание результата
```

---

## 4. Страницы UI

### Dashboard (`/`)
Главная страница с обзором всех задач.
- **4 карточки**: Всего задач, Активные, Завершённые, С ошибками
- **Таблица задач**: тип, статус (цветной Badge), прогресс, дата, действия
- **Действия**: скачать результат, отменить задачу
- **Автообновление**: polling каждые 5 секунд

### Массовый рерайт (`/rewrite`)
Пакетная перезапись текста через LLM с множителем.
- **Загрузка**: drag-and-drop CSV/TXT
- **Настройки**: колонка, промпт, модель, пресет параметров
- **Множитель**: x1–x10 (каждый проход берёт результат предыдущего)
- **Потоки**: 1–20 параллельных запросов
- **Стоп-слова**: toggle для автоматического применения после каждого прохода
- **Прогресс**: SSE с проходом, строкой, %, ETA, скоростью, ошибками

### Перевод (`/translate`)
Пакетный перевод через LLM.
- **Языки**: English, Chinese, Japanese, Hindi, Spanish, Russian, German, French
- **Настройки**: колонка, модель, пресет, потоки

### Постобработка (`/postprocess`)
Локальная очистка текста без LLM.
- **Pipeline**: домены → комментарии → стоп-слова → иероглифы → эмодзи → символы → пробелы → truncate
- **Вредные паттерны**: textarea, по одному на строку
- Стоп-слова из БД применяются автоматически

### AI Process 3.0 (`/ai-process`)
3-шаговый pipeline для генерации SEO-контента.
- **Шаг 1** — WD Tagger: анализ тумбы → теги + рейтинг (parallel, batch по 10)
- **Шаг 2** — LLM: маппинг тегов на разрешённые из БД (5 тегов + 1-3 категории)
- **Шаг 3** — LLM: генерация SEO title (90 символов) + description (160 символов)
- **Входной CSV**: `thumbnail_url`, `title`, `tags`, `categories`, `video_url`
- **Выходной CSV**: + `ai_raw_tags`, `rating`, `mapped_tags`, `mapped_categories`, `seo_title`, `seo_description`

### Стоп-слова (`/stopwords`)
CRUD-интерфейс для управления стоп-словами.
- **Добавление**: слово + замена (пустая = удалить)
- **Toggle**: включить/выключить отдельные слова
- **Удаление**: кнопка с подтверждением

### Теги & Категории (`/tags`)
Управление разрешёнными тегами и категориями для AI Process.
- **Tabs**: Теги / Категории
- **Теги**: имя + категория (action, body_type, setting, ...)
- **Категории**: имя + slug (автогенерация)

### Настройки (`/settings`)
Управление пресетами параметров LLM.
- **Список пресетов**: таблица с клик-для-загрузки
- **Редактор**: 9 параметров (system prompt, max_tokens, temperature, top_p, min_p, top_k, presence_penalty, frequency_penalty, repetition_penalty)

---

## 5. API Reference

### `POST /api/files/upload`
Загрузка CSV/TXT файла.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `file` | File (form-data) | CSV или TXT, макс 100MB |

**Ответ:**
```json
{
  "fileUrl": "./uploads/uuid.csv",
  "filename": "original_name.csv",
  "size": 12345,
  "lineCount": 500
}
```

---

### `GET /api/files/:id`
Скачивание результата задачи (id = jobId).

**Ответ:** CSV файл с заголовком `Content-Disposition: attachment`.

---

### `GET /api/jobs`
Список задач с фильтрацией.

| Параметр | Тип | По умолчанию |
|----------|-----|-------------|
| `type` | REWRITE/TRANSLATE/POSTPROCESS/AI_PROCESS | — |
| `status` | PENDING/RUNNING/COMPLETED/FAILED/CANCELLED | — |
| `limit` | number | 50 |
| `offset` | number | 0 |

**Ответ:** `{ jobs: Job[], total: number }`

---

### `POST /api/jobs`
Создание задачи. Автоматически добавляет в очередь BullMQ.

**Тело запроса:**
```json
{
  "type": "REWRITE",
  "inputFileUrl": "./uploads/uuid.csv",
  "config": {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "systemPrompt": "You are a helpful assistant.",
    "userPrompt": "Rewrite this text:",
    "multiplier": 3,
    "maxTokens": 512,
    "temperature": 0.7,
    "titleCol": "title",
    "maxWorkers": 5,
    "chunkSize": 10,
    "applyStopWords": true
  }
}
```

**Ответ:** `{ job: Job }` (status 201)

---

### `GET /api/jobs/:id`
Детали одной задачи.

**Ответ:** `{ job: Job }`

---

### `DELETE /api/jobs/:id`
Отмена задачи (PENDING/RUNNING → CANCELLED).

**Ответ:** `{ success: true }`

---

### `GET /api/jobs/:id/stream`
SSE-поток прогресса задачи в реальном времени.

**Формат события:**
```json
{
  "jobId": "clx...",
  "status": "RUNNING",
  "processedRows": 150,
  "totalRows": 500,
  "failedRows": 2,
  "currentPass": 2,
  "totalPasses": 3,
  "eta": 120,
  "speed": 2.5
}
```

Поток закрывается автоматически при COMPLETED/FAILED/CANCELLED.

---

### `GET /api/models`
Список доступных моделей Novita AI. Кешируется в Redis на 1 час.

**Ответ:** `{ models: [{ id: string, object: string }] }`

---

### `GET /api/stopwords` | `POST` | `PATCH` | `DELETE`

| Метод | Описание | Тело/Параметры |
|-------|----------|---------------|
| GET | Список всех | — |
| POST | Создать (одно или массив) | `{ word, replacement? }` или `[...]` |
| PATCH | Обновить | `{ id, isActive?, word?, replacement? }` |
| DELETE | Удалить | `?id=xxx` |

---

### `GET /api/presets` | `POST` | `DELETE`

| Метод | Описание |
|-------|----------|
| GET | Список всех пресетов |
| POST | Создать/обновить (upsert по имени) |
| DELETE | Удалить `?id=xxx` |

---

### `GET /api/tags` | `POST` | `DELETE`

| Метод | Описание |
|-------|----------|
| GET | `?type=tags` или `?type=categories` |
| POST | `{ type, items: [{ name, category/slug }] }` |
| DELETE | `?type=tags&id=xxx` |

---

## 6. Workers

### Rewrite Worker
Массовый рерайт с множителем и двумя режимами.

**Режим Batch API** (>= 50 строк):
1. Собирает все строки в JSONL: `{"custom_id":"row-0","body":{...}}`
2. Загружает в Novita (`POST /v1/files`)
3. Создаёт batch (`POST /v1/batches`)
4. Поллит статус каждые 5 сек (макс 2 часа)
5. Скачивает результаты (`GET /v1/files/:id/content`)
6. При ошибке batch — fallback на realtime

**Режим Realtime** (< 50 строк):
1. Чанки по `chunkSize` строк
2. Параллельные запросы (до `maxWorkers`)
3. Per-row retry: 3 попытки с exponential backoff

**Общее для обоих:**
- Множитель: каждый проход берёт результат предыдущего
- Failed rows: 2 дополнительных цикла realtime retry
- Стоп-слова применяются после каждого прохода
- Выходные колонки: `rewrite_1`, `rewrite_2`, ..., `rewrite_N`

### Translate Worker
- Чанки + параллельность (как realtime rewrite)
- Промпт: "Translate from {source} to {target}"
- Выходная колонка: `translated_title`

### Postprocess Worker
- Локальная обработка (без API вызовов) — самый быстрый
- Pipeline из 10 шагов очистки
- Обновляет прогресс каждые 100 строк
- Выходная колонка: `cleaned`

### AI Process Worker
**3 шага, каждый с параллельностью:**

| Шаг | API | Concurrency | Retry |
|-----|-----|-------------|-------|
| 1. WD Tagger | HuggingFace Gradio | 3 parallel, 500ms interval | 4 попытки + wake-up |
| 2. Tag Mapping | Novita Chat | maxWorkers | 3 попытки |
| 3. SEO Generation | Novita Chat | maxWorkers | 3 попытки |

---

## 7. Библиотеки

### novita-client.ts
Realtime клиент для Novita AI API.
- Base URL: `https://api.novita.ai/openai`
- Retry: 5 попыток, exponential backoff (1s → 16s) при 429/5xx
- Кеш моделей: Redis, TTL 1 час
- Функции: `listModels()`, `chatCompletion()`

### novita-batch-client.ts
Batch API клиент для больших объёмов.
- Формат JSONL: `{"custom_id":"row-N","body":{...}}`
- Endpoints: `/v1/files`, `/v1/batches`, `/v1/batches/:id`, `/v1/files/:id/content`
- Лимиты: 50,000 запросов/batch, 100MB, 48 часов
- Функции: `buildBatchJsonl()`, `uploadBatchFile()`, `createBatch()`, `pollBatchUntilDone()`, `downloadBatchResults()`, `submitBatchAndWait()`

### wd-tagger-client.ts
Клиент для WD Tagger на HuggingFace.
- API: Gradio `POST /api/predict`
- Модели: ViT (default), ViT-Large, EVA02-Large, ConvNeXt, SwinV2
- Rate limit: 3 concurrent, 500ms min interval
- Retry: 4 попытки, wake-up при cold start (503)
- Timeout: 60 сек
- Функции: `analyzeThumbnail()`, `analyzeThumbnailsBatch()`

### text-processing.ts
Pipeline очистки текста (порт из Python `app.py`).
- `cleanText(text, stopWords, harmfulPatterns)` — полный pipeline
- `removeEmojis()`, `removeHieroglyphs()`, `removeDomains()`
- `applyStopWordReplacements()`, `stripCommentaryPhrases()`
- `fixMissingSpaces()`, `truncateTitle(text, maxLen=100)`
- `postprocessLLMResponse()` — очистка ответов LLM

---

## 8. БД Схема (Prisma)

### Job
| Поле | Тип | Описание |
|------|-----|----------|
| id | String (cuid) | PK |
| type | JobType | REWRITE / TRANSLATE / POSTPROCESS / AI_PROCESS |
| status | JobStatus | PENDING / RUNNING / COMPLETED / FAILED / CANCELLED |
| config | Json | Параметры задачи |
| inputFileUrl | String | Путь к входному файлу |
| outputFileUrl | String? | Путь к результату |
| totalRows | Int | Всего строк |
| processedRows | Int | Обработано |
| failedRows | Int | С ошибками |
| currentPass | Int | Текущий проход |
| totalPasses | Int | Всего проходов |
| errorLog | Json? | [{row, error, retries}] |
| startedAt | DateTime? | |
| completedAt | DateTime? | |

### Preset
| Поле | Тип |
|------|-----|
| name | String (unique) |
| systemPrompt | String |
| maxTokens | Int |
| temperature, topP, minP | Float |
| topK | Int |
| presencePenalty, frequencyPenalty, repetitionPenalty | Float |
| isDefault | Boolean |

### StopWord
| Поле | Тип | Описание |
|------|-----|----------|
| word | String (unique) | Слово для поиска |
| replacement | String? | null = удалить |
| isActive | Boolean | Включено/выключено |

### AllowedTag
| Поле | Тип |
|------|-----|
| name | String (unique) |
| category | String |

### AllowedCategory
| Поле | Тип |
|------|-----|
| name | String (unique) |
| slug | String (unique) |

---

## 9. Конфигурация

### Переменные окружения (.env)

| Переменная | Обязательна | По умолчанию | Описание |
|-----------|-------------|-------------|----------|
| DATABASE_URL | Да | — | PostgreSQL connection string |
| DB_PASSWORD | Да | — | Пароль БД (используется в docker-compose) |
| REDIS_URL | Нет | redis://localhost:6379 | Redis connection |
| NOVITA_API_KEY | Да | — | API ключ Novita AI |
| NOVITA_BASE_URL | Нет | https://api.novita.ai/openai | Base URL API |
| NOVITA_RATE_LIMIT | Нет | 60 | Лимит запросов/мин |
| WD_TAGGER_URL | Нет | https://deepghs-wd-tagger-heatmap-more-models.hf.space | URL WD Tagger |
| UPLOAD_DIR | Нет | ./uploads | Директория загрузок |
| RESULT_DIR | Нет | ./results | Директория результатов |
| NODE_ENV | Нет | — | production для Docker |

### Docker Compose (продакшн)

| Контейнер | Образ | Порт | Описание |
|-----------|-------|------|----------|
| llmai-app | Dockerfile | 3000 | Next.js веб-сервер |
| llmai-worker | Dockerfile.worker | — | BullMQ воркеры (tsx) |
| llmai-postgres | postgres:16-alpine | 5432 (localhost only) | БД |
| llmai-redis | redis:7-alpine | 6379 (localhost only) | Кеш + очередь |

---

## 10. Деплой

### CI/CD Pipeline (GitHub Actions)

```
git push → GitHub Actions:
  1. npm ci + prisma generate
  2. tsc --noEmit (type check)
  3. npm run build
  4. SSH на сервер:
     → git pull
     → docker-compose build --no-cache
     → docker-compose up -d
     → health check (curl localhost:3000)
```

**Триггер:** push в `main` при изменениях в `llmai-v3/`

**GitHub Secrets:**

| Secret | Значение |
|--------|----------|
| TIMEWEB_HOST | IP сервера |
| TIMEWEB_USER | root |
| TIMEWEB_PORT | 22 |
| TIMEWEB_SSH_KEY | Приватный SSH ключ (ed25519) |

### Ручной деплой

```bash
ssh root@<server-ip>
cd /opt/llmai/llmai-v3
./scripts/deploy.sh main
```

### Первая установка на чистый сервер

```bash
ssh root@<server-ip>
curl -fsSL https://raw.githubusercontent.com/AlexGolubevpm/llmai/main/llmai-v3/scripts/full-setup.sh | bash
# Отредактировать .env: nano /opt/llmai/llmai-v3/.env
```

---

## 11. UX/UI Анализ

### Компоненты

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| Sidebar | `components/sidebar.tsx` | Навигация (8 пунктов), подсветка активного |
| FileUpload | `components/file-upload.tsx` | Drag-and-drop + клик, показ имени файла |
| JobProgress | `components/job-progress.tsx` | SSE-подписка, прогресс-бар, ETA, скорость |
| ModelSelector | `components/model-selector.tsx` | Dropdown моделей с refresh |
| PresetSelector | `components/preset-selector.tsx` | Dropdown пресетов + 8 слайдеров |
| Providers | `components/providers.tsx` | React Query + ThemeProvider (тёмная тема) |

### shadcn/ui компоненты (20 шт)
Button, Card, Badge, Dialog, Sheet, Tabs, Select, Slider, Progress, Table, Input, Textarea, Label, Separator, DropdownMenu, Sonner (toast), ScrollArea, Switch, Tooltip, Avatar

### Что работает хорошо
- Тёмная тема по умолчанию (next-themes)
- Real-time прогресс через SSE с ETA
- Drag-and-drop загрузка файлов
- Адаптивная сетка (grid-cols-1 → grid-cols-2 на lg)
- Toast уведомления через sonner
- Цветовая кодировка статусов задач

### Проблемы и рекомендации

#### Высокий приоритет

| Проблема | Файл | Рекомендация |
|----------|------|-------------|
| Нет мобильного меню | `sidebar.tsx` | Добавить Sheet с hamburger-кнопкой на `md:hidden` |
| Ошибки проглатываются | `page.tsx` (все страницы) | Заменить `catch {}` на `toast.error()` |
| Нет ARIA на icon-кнопках | `page.tsx` (Dashboard) | Добавить `aria-label` на Download, Trash, Refresh |
| Нет деталей FAILED задач | `page.tsx` (Dashboard) | Показывать errorLog при клике на FAILED задачу |

#### Средний приоритет

| Проблема | Файл | Рекомендация |
|----------|------|-------------|
| Нет loading на CRUD | `stopwords/`, `tags/`, `settings/` | Добавить disabled-состояние при fetch |
| SSE без reconnect | `job-progress.tsx` | Добавить reconnect с backoff при onerror |
| alert() вместо toast | `file-upload.tsx` | Заменить на `toast.error()` |
| Нет скелетонов | Все страницы | Добавить Skeleton компоненты при первой загрузке |

#### Низкий приоритет

| Проблема | Рекомендация |
|----------|-------------|
| Нет валидации CSV | Проверять наличие ожидаемых колонок после загрузки |
| Нет мемоизации | Обернуть тяжёлые компоненты в React.memo |
| Нет виртуализации таблиц | Добавить react-virtual для 1000+ строк |
