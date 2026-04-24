# LLMAI v3.0 — Документация

## Обзор

LLMAI v3.0 — платформа массовой обработки контента через AI. Рерайт текстов, генерация SEO, анализ изображений, категоризация видео, PBN-тексты.

---

## Стек

| Слой | Технология | Версия |
|------|-----------|--------|
| Frontend | Next.js (App Router) + TypeScript | 16.2 |
| UI | shadcn/ui + Tailwind CSS | latest |
| Анимации | Framer Motion | 12.x |
| Таблицы | TanStack Table | 8.x |
| Backend | Next.js API Routes | — |
| Очередь задач | BullMQ | 5.71 |
| БД | PostgreSQL (Prisma ORM) | 16 |
| Кеш/Брокер | Redis | 7 |
| AI API | OpenRouter (OpenAI-совместимый) | v1 |
| Деплой | Docker Compose | — |
| CI/CD | GitHub Actions → SSH → Timeweb | — |

---

## OpenRouter API

### Подключение

```
Base URL: https://openrouter.ai/api/v1
Auth: Authorization: Bearer <OPENROUTER_API_KEY>
Headers: HTTP-Referer, X-Title (для рейтинга на OpenRouter)
```

### Endpoints используемые в приложении

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/chat/completions` | POST | Генерация текста (все воркеры) |
| `/models` | GET | Список доступных моделей (кеш 1 час в Redis) |

### Rate Limiting

- Минимальный интервал между запросами: 300ms (~200 req/min)
- Retry с exponential backoff при 429 (до 5 попыток)
- Поддержка `Retry-After` заголовка
- `provider: { allow_fallbacks: true }` для гео-ограничений

### Параметры генерации

```json
{
  "model": "openai/gpt-4o-mini",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": 40,
  "min_p": 0.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0
}
```

### Vision API (AI Process)

Картинки скачиваются на сервере → base64 → отправляются в `image_url` поле:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j..."}},
      {"type": "text", "text": "Analyze this image..."}
    ]
  }]
}
```

---

## Разделы приложения

### Dashboard (`/`)

Обзор всех задач. Таблица с типом, моделью, статусом, прогрессом, датой. Клик → детали задачи (`/jobs/[id]`). Автообновление каждые 5 сек. Скачивание результатов, отмена задач.

### Рерайт (`/rewrite`)

Массовый рерайт текстов с множителем x1-x10.

- **Вход:** CSV/TXT файл или текст в textarea
- **Настройки:** модель, промпт, множитель, потоки, стоп-слова
- **Множитель:** каждый проход берёт результат предыдущего
- **Выход:** CSV с колонками `rewrite_1`, `rewrite_2`, ... `rewrite_N`
- **Retry:** 3 попытки + 2 доп. цикла для failed строк

### Перевод (`/translate`)

Пакетный перевод через LLM.

- **Языки:** EN, CN, JP, HI, ES, RU, DE, FR
- **Вход:** CSV/TXT с колонкой title
- **Выход:** CSV + колонка `translated_title`

### Постобработка (`/postprocess`)

Локальная очистка текста без LLM (самый быстрый инструмент).

Pipeline из 8 шагов: домены → комментарии → стоп-слова → иероглифы → эмодзи → символы → пробелы → truncate. Выход: колонка `cleaned`.

### AI Process (`/ai-process`)

Vision-модель анализирует тумбу видео.

- **Модель по умолчанию:** `google/gemini-2.5-flash-preview-05-20`
- **1 API вызов на строку** → JSON: tags, scene, type, title, description
- **Бандлы:** теги/категории из бандла вставляются в промпт
- **Base64:** сервер скачивает картинку → конвертирует → отправляет
- **Выход:** ai_tags, scene_description, content_type, seo_title, seo_description

### Категоризация (`/categorize`)

Присвоение тегов/категорий из бандла к каждому видео.

- **Вход:** фид (id, title, tags, categories) + выбранный бандл
- **AI:** анализирует title + существующие теги → подбирает N категорий и M тегов из списка бандла
- **Настройки:** кол-во категорий (1-10), кол-во тегов (1-20)
- **Выход:** + колонки `new_categories`, `new_tags`

### Описания фида (`/feed-descriptions`)

Генерация SEO description для видео на основе названия, тегов и категорий.

- **Вход:** TXT файл формата `#ID|Название|Категории|Тэги`
- **Промпт:** `{title}`, `{categories}`, `{tags}` — подставляются для каждой строки
- **Выход:** CSV + колонка `description`

### PBN тексты (`/pbn`)

Генерация уникальных SEO текстов для PBN сетей.

- **Вход:** 1-5 сайтов (домены) + количество (1-500)
- **Промпт:** `{sites}`, `{number}`, `{total}`
- **Температура 0.9** для максимальной уникальности
- **Выход:** CSV (id, text, sites)

### SEO Categories (`/seo-generator`)

SEO title + description для страниц тегов/категорий.

- **Вход:** список тегов/категорий (по одному на строку)
- **Промпт:** `{name}` — подставляется название
- **Выход:** CSV (tag_category, seo_title, seo_description)

### Бандлы (`/bundles`)

Наборы тегов и категорий для разных ниш (Транс, Гей, JAV, Хентай).

- Название, описание, теги (через запятую), категории (через запятую)
- Кастомный промпт (опционально)
- isDefault — автовыбор в AI Process и Категоризации

### Стоп-слова (`/stopwords`)

Управление заменами при обработке текста.

- Одиночное и массовое добавление (paste/file)
- CSV парсер: term + synonyms + переводы
- Toggle вкл/выкл, поиск
- Применяются автоматически в рерайте, AI Process, постобработке

### Настройки (`/settings`)

Пресеты параметров LLM. Master-detail layout.

- Модель, system prompt, 8 слайдеров (temperature, top_p, top_k и т.д.)
- Применяются в рерайте и переводе через PresetSelector

---

## Архитектура задач

Все инструменты работают через BullMQ:

```
UI → POST /api/jobs → создание Job в БД + добавление в очередь
     ↓
BullMQ Worker берёт задачу → обрабатывает строки → публикует прогресс в Redis
     ↓
SSE (GET /api/jobs/:id/stream) → фронтенд показывает прогресс
     ↓
Worker завершает → сохраняет CSV → обновляет Job в БД
     ↓
Dashboard → скачивание результата (GET /api/files/:id)
```

### 8 очередей / воркеров

| Очередь | Job Type | Воркер |
|---------|----------|--------|
| rewrite | REWRITE | rewrite.worker.ts |
| translate | TRANSLATE | translate.worker.ts |
| postprocess | POSTPROCESS | postprocess.worker.ts |
| ai-process | AI_PROCESS | ai-process.worker.ts |
| seo-categories | SEO_CATEGORIES | seo-categories.worker.ts |
| pbn | PBN | pbn.worker.ts |
| categorize | CATEGORIZE | categorize.worker.ts |
| feed-descriptions | FEED_DESCRIPTIONS | feed-descriptions.worker.ts |

### Общие паттерны воркеров

- Чанковая обработка с concurrency limit (`maxWorkers`)
- Retry 3 попытки с exponential backoff
- Cancellation check каждые N строк
- Прогресс через Redis pub/sub → SSE
- Error log: `{ row, step, error, retries }`
- Выходной CSV сохраняется на диск

---

## БД Схема

### Job
| Поле | Тип |
|------|-----|
| id | String (cuid) |
| type | JobType (enum) |
| status | JobStatus (enum) |
| config | Json |
| inputFileUrl | String |
| outputFileUrl | String? |
| totalRows / processedRows / failedRows | Int |
| currentPass / totalPasses | Int |
| errorLog | Json? |
| startedAt / completedAt / createdAt | DateTime |

### Preset
model, systemPrompt, maxTokens, temperature, topP, minP, topK, presencePenalty, frequencyPenalty, repetitionPenalty, isDefault

### StopWord
word (unique), replacement?, isActive

### Bundle
name (unique), description?, tags, categories, prompt?, isDefault

### AllowedTag / AllowedCategory
Общие списки тегов и категорий (legacy, основное в бандлах)

---

## Docker Compose (продакшн)

| Контейнер | Образ | Порт |
|-----------|-------|------|
| llmai-app | Dockerfile (Next.js standalone) | 3000 |
| llmai-worker | Dockerfile.worker (tsx) | — |
| llmai-postgres | postgres:16-alpine | 5432 (localhost) |
| llmai-redis | redis:7-alpine | 6379 (localhost) |

---

## Деплой

### CI/CD (GitHub Actions)

```
git push main → GitHub Actions:
  1. npm ci + prisma generate + tsc + build
  2. SSH на сервер → git pull → docker-compose build → restart
```

### Ручной деплой

```bash
cd /opt/llmai && git pull origin main
cd llmai-v3 && /usr/local/bin/docker-compose -f docker-compose.prod.yml up -d --build
```

### Переменные окружения (.env)

| Переменная | Описание |
|-----------|----------|
| DATABASE_URL | PostgreSQL connection string |
| DB_PASSWORD | Пароль БД |
| REDIS_URL | Redis connection |
| OPENROUTER_API_KEY | API ключ OpenRouter |
| OPENROUTER_BASE_URL | `https://openrouter.ai/api/v1` |
| UPLOAD_DIR | Директория загрузок (`./uploads`) |
| RESULT_DIR | Директория результатов (`./results`) |
| NEXT_PUBLIC_APP_URL | URL приложения |

---

## Форматы файлов

### Pipe-delimited TXT (авто-парсинг)

Автоматически определяет порядок колонок по содержимому:
- ID: чистые цифры
- thumbnail_url: URL с .jpg/.png/screenshot
- video_url: URL с .mp4/.webm
- title: несколько слов
- tags: через запятую
- categories: остальное

### CSV

Стандартный CSV с заголовками. Поддерживает кавычки и запятые в полях.
