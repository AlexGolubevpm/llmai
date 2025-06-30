import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pandas as pd
import time
import concurrent.futures
import re
import math
import csv

#######################################
# 1) НАСТРОЙКИ ПРИЛОЖЕНИЯ
#######################################

# Базовый URL API Novita
API_BASE_URL = "https://api.novita.ai/v3/openai"
LIST_MODELS_ENDPOINT = f"{API_BASE_URL}/models"
CHAT_COMPLETIONS_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Ключ по умолчанию (НЕБЕЗОПАСНО в реальном проде)
DEFAULT_API_KEY = "sk_MyidbhnT9jXzw-YDymhijjY8NF15O0Qy7C36etNTAxE"

# Максимальное количество повторных попыток при 429 (Rate Limit)
MAX_RETRIES = 3

st.set_page_config(page_title="🧠 Novita AI Batch Processor", layout="wide")

#######################################
# 2) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#######################################

def custom_postprocess_text(text: str) -> str:
    """
    Функция постобработки текста:
    1. Удаляет фрагменты, начинающиеся с "Note:".
    2. Удаляет нежелательные слова "fucking", "explicit" и "intense", если они появляются в начале предложения.
    3. Заменяет цензурированное "F***" на "fuck".
    4. Удаляет китайские символы.
    5. Удаляет эмодзи.
    6. Убирает все двойные кавычки и лишние пробелы.
    """
    text = re.sub(r'\s*Note:.*', '', text, flags=re.IGNORECASE)
    pattern_sentence = re.compile(r'(^|(?<=[.!?]\s))\s*(?:fucking|explicit|intense)[\s,:\-]+', flags=re.IGNORECASE)
    text = pattern_sentence.sub(r'\1', text)
    text = re.sub(r'\bF\*+\b', 'fuck', text, flags=re.IGNORECASE)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.replace('"', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_model_list(api_key: str):
    """Загружаем список доступных моделей через API Novita AI"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(LIST_MODELS_ENDPOINT, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return models
        else:
            st.error(f"Не удалось получить список моделей. Код: {resp.status_code}. Текст: {resp.text}")
            return []
    except Exception as e:
        st.error(f"Ошибка при получении списка моделей: {e}")
        return []

def chat_completion_request(
    api_key: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float
):
    """Синхронный chat-комплишн с retries при 429."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "min_p": min_p
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    attempts = 0
    while attempts < MAX_RETRIES:
        attempts += 1
        try:
            resp = requests.post(CHAT_COMPLETIONS_ENDPOINT, headers=headers, data=json.dumps(payload))
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"].get("content", "")
            elif resp.status_code == 429:
                time.sleep(2)
                continue
            else:
                return f"Ошибка: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Исключение: {e}"
    return "Ошибка: Превышено число попыток при 429 RATE_LIMIT."

def process_single_row(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    row_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float
):
    """Обёртка для параллельного вызова генерации."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n{row_text}"}
    ]
    raw_response = chat_completion_request(
        api_key,
        messages,
        model,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        presence_penalty,
        frequency_penalty,
        repetition_penalty
    )
    final_response = custom_postprocess_text(raw_response)
    return final_response

def process_file(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    title_col: str,
    response_format: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    chunk_size: int = 10,
    max_workers: int = 5
):
    """
    Параллельная обработка файла построчно с использованием чанков.
    Отображается прогресс-бар и примерное оставшееся время.
    """
    results = []
    total_rows = len(df)
    progress_bar = st.progress(0)
    time_placeholder = st.empty()  # для отображения оставшегося времени
    overall_start = time.time()
    rows_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_indices = list(df.index[start_idx:end_idx])
        chunk_results = [None] * len(chunk_indices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_i = {}
            for i, row_idx in enumerate(chunk_indices):
                row_text = str(df.loc[row_idx, title_col])
                future = executor.submit(
                    process_single_row,
                    api_key,
                    model,
                    system_prompt,
                    user_prompt,
                    row_text,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    presence_penalty,
                    frequency_penalty,
                    repetition_penalty
                )
                future_to_i[future] = i
            for future in concurrent.futures.as_completed(future_to_i):
                i = future_to_i[future]
                chunk_results[i] = future.result()

        results.extend(chunk_results)
        rows_processed += len(chunk_indices)
        progress_bar.progress(min(rows_processed / total_rows, 1.0))

        # Расчет оставшегося времени
        chunk_time = time.time() - chunk_start
        if len(chunk_indices) > 0:
            time_per_row = chunk_time / len(chunk_indices)
            rows_left = total_rows - rows_processed
            est_time_sec = rows_left * time_per_row
            if est_time_sec < 60:
                time_text = f"~{est_time_sec:.1f} сек."
            else:
                time_text = f"~{est_time_sec/60:.1f} мин."
            time_placeholder.info(f"Примерное оставшееся время: {time_text}")

    overall_time = time.time() - overall_start
    time_placeholder.success(f"Обработка завершена за {overall_time:.1f} сек.")
    df_out = df.copy()
    df_out["rewrite"] = results
    return df_out

# --- Функции для перевода (аналогичны обработке текста, с прогресс-баром) ---

def translate_completion_request(
    api_key: str,
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float
):
    raw_response = chat_completion_request(
        api_key,
        messages,
        model,
        max_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        presence_penalty,
        frequency_penalty,
        repetition_penalty
    )
    final_response = custom_postprocess_text(raw_response)
    return final_response

def process_translation_single_row(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    row_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n{row_text}"}
    ]
    translated_text = translate_completion_request(
        api_key=api_key,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty
    )
    return translated_text

def process_translation_file(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    title_col: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    chunk_size: int = 10,
    max_workers: int = 5
):
    """
    Параллельная обработка файла для перевода с использованием чанков.
    Отображается прогресс-бар и примерное оставшееся время.
    """
    results = []
    total_rows = len(df)
    progress_bar = st.progress(0)
    time_placeholder = st.empty()
    overall_start = time.time()
    rows_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_indices = list(df.index[start_idx:end_idx])
        chunk_results = [None] * len(chunk_indices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_i = {}
            for i, row_idx in enumerate(chunk_indices):
                row_text = str(df.loc[row_idx, title_col])
                future = executor.submit(
                    process_translation_single_row,
                    api_key,
                    model,
                    system_prompt,
                    user_prompt,
                    row_text,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    presence_penalty,
                    frequency_penalty,
                    repetition_penalty
                )
                future_to_i[future] = i
            for future in concurrent.futures.as_completed(future_to_i):
                i = future_to_i[future]
                chunk_results[i] = future.result()

        results.extend(chunk_results)
        rows_processed += len(chunk_indices)
        progress_bar.progress(min(rows_processed / total_rows, 1.0))

        chunk_time = time.time() - chunk_start
        if len(chunk_indices) > 0:
            time_per_row = chunk_time / len(chunk_indices)
            rows_left = total_rows - rows_processed
            est_time_sec = rows_left * time_per_row
            if est_time_sec < 60:
                time_text = f"~{est_time_sec:.1f} сек."
            else:
                time_text = f"~{est_time_sec/60:.1f} мин."
            time_placeholder.info(f"Примерное оставшееся время: {time_text}")

    overall_time = time.time() - overall_start
    time_placeholder.success(f"Перевод завершен за {overall_time:.1f} сек.")
    df_out = df.copy()
    df_out["translated_title"] = results
    return df_out

# --- Функция для постобработки файла с удалением вредных паттернов ---
def clean_text(text: str, harmful_patterns: list) -> str:
    """
    Для каждого паттерна из списка удаляет его вхождения из текста.
    """
    for pattern in harmful_patterns:
        text = re.sub(re.escape(pattern), "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_postprocessing_file(df: pd.DataFrame, text_col: str, harmful_patterns: list):
    """Обрабатывает DataFrame, применяя очистку текста по заданным паттернам."""
    cleaned_texts = df[text_col].astype(str).apply(lambda txt: clean_text(txt, harmful_patterns))
    df_out = df.copy()
    df_out["cleaned"] = cleaned_texts
    return df_out

#######################################
# 3) ПРЕСЕТЫ МОДЕЛЕЙ
#######################################

PRESETS = {
    "Default": {
        "system_prompt": "Act like you are a helpful assistant.",
        "max_tokens": 512,
        "temperature": 0.70,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 40,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0
    },
    "NSFW": {
        "system_prompt": "You are an advanced NSFW content rewriter and evaluator. Generate one vivid and explicit title based on the input, ensuring it stays within 90 characters. The title should align with NSFW standards, SEO relevance, and native fluency.",
        "max_tokens": 32000,
        "temperature": 0.70,
        "top_p": 1.0,
        "min_p": 0.0,
        "top_k": 40,
        "presence_penalty": 0.20,
        "frequency_penalty": 0.40,
        "repetition_penalty": 1.22
    },
    "Adult_Content_Generator": {
        "system_prompt": "You are a professional content creator specializing in adult NSFW content. Generate creative and engaging content based on the input provided.",
        "max_tokens": 2500,
        "temperature": 0.85,
        "top_p": 0.95,
        "min_p": 0.0,
        "top_k": 50,
        "presence_penalty": 0.30,
        "frequency_penalty": 0.50,
        "repetition_penalty": 1.50
    },
    "Erotic_Story_Teller": {
        "system_prompt": "You are an expert in writing erotic stories. Generate a captivating and tasteful story based on the input provided.",
        "max_tokens": 5000,
        "temperature": 0.75,
        "top_p": 0.90,
        "min_p": 0.0,
        "top_k": 60,
        "presence_penalty": 0.25,
        "frequency_penalty": 0.35,
        "repetition_penalty": 1.30
    }
}

#######################################
# 4) ИНТЕРФЕЙС
#######################################

st.title("🧠 Novita AI Batch Processor")

# Поле ввода API Key
st.sidebar.header("🔑 Настройки API")
api_key = st.sidebar.text_input("API Key", value=DEFAULT_API_KEY, type="password")

# Создаем 7 вкладок: Обработка текста, Перевод текста, Разделение файла, Постобработка,
# Выбор столбцов, Удаление кавычек, Автодополнение тегов
tabs = st.tabs([
    "🔄 Обработка текста",
    "🌐 Перевод текста",
    "📂 Разделение файла",
    "🧹 Постобработка",
    "🗂️ Выбор столбцов",
    "❌ Удаление кавычек",
    "🏷️ Теги"
])

########################################
# Вкладка 1: Обработка текста
########################################
with tabs[0]:
    st.header("🔄 Обработка текста")
    # ... (ваш существующий код вкладки 0 без изменений) ...

########################################
# Вкладка 2: Перевод текста
########################################
with tabs[1]:
    st.header("🌐 Перевод текста")
    # ... (ваш существующий код вкладки 1 без изменений) ...

########################################
# Вкладка 3: Разделение файла
########################################
with tabs[2]:
    st.header("📂 Разделение файла")
    # ... (ваш существующий код вкладки 2 без изменений) ...

########################################
# Вкладка 4: Постобработка
########################################
with tabs[3]:
    st.header("🧹 Постобработка")
    # ... (ваш существующий код вкладки 3 без изменений) ...

########################################
# Вкладка 5: Выбор столбцов из CSV
########################################
with tabs[4]:
    st.header("🗂️ Выбор столбцов из CSV")
    # ... (ваш существующий код вкладки 4 без изменений) ...

########################################
# Вкладка 6: Удаление кавычек из выбранных столбцов
########################################
with tabs[5]:
    st.header("❌ Удаление кавычек")
    # ... (ваш существующий код вкладки 5 без изменений) ...

########################################
# Вкладка 7: Автодополнение и фильтрация тегов
########################################
with tabs[6]:
    st.header("🏷️ Автодополнение и фильтрация тегов")
    st.write("1) Введите список разрешённых тегов (Allowed Tags).")
    tags_input = st.text_area(
        "Allowed Tags (через запятую или по строкам):",
        height=150,
        placeholder="например: wellness, health, beauty, fitness, anti-aging..."
    )
    user_tags = [t.strip() for t in re.split(r'[\n,]+', tags_input) if t.strip()]

    st.write("2) Загрузите CSV с колонкой существующих тегов.")
    uploaded = st.file_uploader("CSV-файл", type="csv", key="tags_csv")
    if uploaded and user_tags:
        df_tags = pd.read_csv(uploaded)
        cols = df_tags.columns.tolist()
        tag_col = st.selectbox("Колонка с существующими тегами", cols, key="tag_col")
        max_workers_tags = st.slider("Параллельных потоков", 1, 10, 5, key="max_workers_tags")

        if st.button("▶️ Обработать теги", key="process_tags"):
            def process_tags_row(row):
                existing = [t.strip() for t in re.split(r'[;,]', str(row[tag_col])) if t.strip()]
                context = row[tag_col]  # используем ту же колонку как контекст
                allowed = ", ".join(user_tags)

                system_msg = "You are an expert tag selector."
                user_msg = f"""
Allowed tags: {allowed}.
Existing tags: {', '.join(existing)}.
Context: {context}

Task: From the Allowed tags list, pick exactly 5 tags that best describe the Context.
- Always include any Existing tags that are relevant.
- If there are fewer than 5 relevant Existing tags, add other Allowed tags that fit the Context to reach exactly 5.
- Do NOT choose tags outside the Allowed list.
Return only a comma-separated list of the 5 tags.
"""
                resp = chat_completion_request(
                    api_key,
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg}
                    ],
                    selected_model_text,
                    max_tokens_text,
                    temperature_text,
                    top_p_text,
                    min_p_text,
                    top_k_text,
                    presence_penalty_text,
                    frequency_penalty_text,
                    repetition_penalty_text
                )
                tags_out = custom_postprocess_text(resp)
                return [t.strip() for t in tags_out.split(",")][:5]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_tags) as executor:
                augmented = list(executor.map(process_tags_row, [row for _, row in df_tags.iterrows()]))

            df_tags["final_5_tags"] = [", ".join(tlist) for tlist in augmented]
            st.success("✅ Теги сгенерированы!")
            st.dataframe(df_tags.head(10))

            csv_out = df_tags.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Скачать CSV с 5 тегами",
                data=csv_out,
                file_name="tags_with_5.csv",
                mime="text/csv"
            )
    elif uploaded and not user_tags:
        st.warning("Пожалуйста, введите хотя бы один разрешённый тег выше.")


