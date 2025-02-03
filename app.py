import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pandas as pd
import time
import concurrent.futures
import re
import math

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
    # 1. Удаляем любые фрагменты, начинающиеся с "Note:"
    text = re.sub(r'\s*Note:.*', '', text, flags=re.IGNORECASE)
    # 2. Удаляем нежелательные слова в начале предложения.
    pattern_sentence = re.compile(r'(^|(?<=[.!?]\s))\s*(?:fucking|explicit|intense)[\s,:\-]+', flags=re.IGNORECASE)
    text = pattern_sentence.sub(r'\1', text)
    # 3. Заменяем "F***" на "fuck"
    text = re.sub(r'\bF\*+\b', 'fuck', text, flags=re.IGNORECASE)
    # 4. Удаляем китайские символы
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    # 5. Удаляем эмодзи
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # 6. Убираем двойные кавычки и лишние пробелы
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
    """Параллельная обработка файла построчно без логирования."""
    results = []
    total_rows = len(df)
    for start_idx in range(0, total_rows, chunk_size):
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
    df_out = df.copy()
    df_out["rewrite"] = results
    return df_out

# --- Функции для перевода (аналогичны обработке текста, логика без логирования) ---

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
    results = []
    total_rows = len(df)
    for start_idx in range(0, total_rows, chunk_size):
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
    df_out = df.copy()
    df_out["translated_title"] = results
    return df_out

# --- Новая функция для постобработки файла с удалением вредных паттернов ---
def clean_text(text: str, harmful_patterns: list) -> str:
    """
    Для каждого паттерна из списка удаляет его вхождения из текста.
    """
    for pattern in harmful_patterns:
        # Экранируем спецсимволы, чтобы искать буквальное совпадение (без учета регистра)
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

# Создаем 4 вкладки: Обработка текста, Перевод текста, Разделение файла, Постобработка
tabs = st.tabs(["🔄 Обработка текста", "🌐 Перевод текста", "📂 Разделение файла", "🧹 Постобработка"])

########################################
# Вкладка 1: Обработка текста
########################################
with tabs[0]:
    st.header("🔄 Обработка текста")

    with st.expander("🎨 Выбор пресета модели", expanded=True):
        preset_names = list(PRESETS.keys())
        selected_preset = st.selectbox("Выберите пресет", preset_names, index=0)
        preset = PRESETS[selected_preset]
        system_prompt_text = preset["system_prompt"]
        max_tokens_text = preset["max_tokens"]
        temperature_text = preset["temperature"]
        top_p_text = preset["top_p"]
        min_p_text = preset["min_p"]
        top_k_text = preset["top_k"]
        presence_penalty_text = preset["presence_penalty"]
        frequency_penalty_text = preset["frequency_penalty"]
        repetition_penalty_text = preset["repetition_penalty"]
        if st.button("Сбросить настройки пресета", key="reset_preset_text"):
            selected_preset = "Default"
            preset = PRESETS[selected_preset]
            system_prompt_text = preset["system_prompt"]
            max_tokens_text = preset["max_tokens"]
            temperature_text = preset["temperature"]
            top_p_text = preset["top_p"]
            min_p_text = preset["min_p"]
            top_k_text = preset["top_k"]
            presence_penalty_text = preset["presence_penalty"]
            frequency_penalty_text = preset["frequency_penalty"]
            repetition_penalty_text = preset["repetition_penalty"]

    st.markdown("---")
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("📚 Список моделей для обработки текста")
        st.caption("Список моделей загружается из API Novita AI")
        if st.button("🔄 Обновить список моделей (Обработка текста)", key="refresh_models_text"):
            if not api_key:
                st.error("❌ Ключ API пуст")
                st.session_state["model_list_text"] = []
            else:
                model_list_text = get_model_list(api_key)
                st.session_state["model_list_text"] = model_list_text
        if "model_list_text" not in st.session_state:
            st.session_state["model_list_text"] = []
        if st.session_state["model_list_text"]:
            selected_model_text = st.selectbox("✅ Выберите модель", st.session_state["model_list_text"], key="select_model_text")
        else:
            selected_model_text = st.selectbox("✅ Выберите модель", ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"], key="select_model_default_text")
    with right_col:
        with st.expander("⚙️ Настройки генерации", expanded=True):
            st.subheader("⚙️ Параметры генерации")
            system_prompt_text = st.text_area("📝 System Prompt", value=system_prompt_text, key="system_prompt_text")
            max_tokens_text = st.slider("🔢 max_tokens", min_value=0, max_value=64000, value=max_tokens_text, step=1, key="max_tokens_text")
            temperature_text = st.slider("🌡️ temperature", min_value=0.0, max_value=2.0, value=temperature_text, step=0.01, key="temperature_text")
            top_p_text = st.slider("📊 top_p", min_value=0.0, max_value=1.0, value=top_p_text, step=0.01, key="top_p_text")
            min_p_text = st.slider("📉 min_p", min_value=0.0, max_value=1.0, value=min_p_text, step=0.01, key="min_p_text")
            top_k_text = st.slider("🔝 top_k", min_value=0, max_value=100, value=top_k_text, step=1, key="top_k_text")
            presence_penalty_text = st.slider("⚖️ presence_penalty", min_value=0.0, max_value=2.0, value=presence_penalty_text, step=0.01, key="presence_penalty_text")
            frequency_penalty_text = st.slider("📉 frequency_penalty", min_value=0.0, max_value=2.0, value=frequency_penalty_text, step=0.01, key="frequency_penalty_text")
            repetition_penalty_text = st.slider("🔁 repetition_penalty", min_value=0.0, max_value=2.0, value=repetition_penalty_text, step=0.01, key="repetition_penalty_text")

    st.subheader("📝 Одиночный промпт")
    user_prompt_single_text = st.text_area("Введите промпт для одиночной генерации", key="user_prompt_single_text")
    if st.button("🚀 Отправить одиночный промпт (Обработка текста)", key="submit_single_text"):
        if not api_key:
            st.error("❌ API Key не указан!")
        elif not user_prompt_single_text.strip():
            st.error("❌ Промпт не может быть пустым!")
        else:
            messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt_single_text}
            ]
            raw_response = chat_completion_request(
                api_key=api_key,
                messages=messages,
                model=selected_model_text,
                max_tokens=max_tokens_text,
                temperature=temperature_text,
                top_p=top_p_text,
                min_p=min_p_text,
                top_k=top_k_text,
                presence_penalty=presence_penalty_text,
                frequency_penalty=frequency_penalty_text,
                repetition_penalty=repetition_penalty_text
            )
            final_response = custom_postprocess_text(raw_response)
            st.success("✅ Результат получен!")
            st.text_area("📄 Ответ от модели", value=final_response, height=200)

    st.markdown("---")
    st.subheader("📂 Обработка данных из файла")
    user_prompt_text = st.text_area("Пользовательский промпт (дополнительно к заголовку)", key="user_prompt_text")
    st.markdown("##### Настройка парсинга TXT/CSV")
    with st.expander("📑 Настройки парсинга файла", expanded=True):
        delimiter_input_text = st.text_input("🔸 Разделитель (delimiter)", value="|", key="delimiter_input_text")
        column_input_text = st.text_input("🔸 Названия колонок (через запятую)", value="id,title", key="column_input_text")
    uploaded_file_text = st.file_uploader("📤 Прикрепить файл (CSV или TXT, до 100000 строк)", type=["csv", "txt"], key="uploaded_file_text")
    df_text = None
    if uploaded_file_text is not None:
        file_extension = uploaded_file_text.name.split(".")[-1].lower()
        try:
            if file_extension == "csv":
                df_text = pd.read_csv(uploaded_file_text)
            else:
                content = uploaded_file_text.read().decode("utf-8")
                lines = content.splitlines()
                columns = [c.strip() for c in column_input_text.split(",")]
                parsed_lines = []
                for line in lines:
                    splitted = line.split(delimiter_input_text, maxsplit=len(columns) - 1)
                    if len(splitted) < len(columns):
                        splitted += [""] * (len(columns) - len(splitted))
                    parsed_lines.append(splitted)
                df_text = pd.DataFrame(parsed_lines, columns=columns)
            st.write("### 📋 Предпросмотр файла")
            num_preview = st.number_input("🔍 Количество строк для предпросмотра", min_value=1, max_value=100, value=10)
            st.dataframe(df_text.head(num_preview))
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {e}")
            df_text = None
    if df_text is not None:
        cols_text = df_text.columns.tolist()
        with st.expander("📂 Выбор колонок файла", expanded=True):
            title_col_text = st.selectbox("📌 Какая колонка является заголовком?", cols_text, key="title_col_text")
            max_workers_text = st.slider("🔄 Потоки (max_workers)", min_value=1, max_value=20, value=5, key="max_workers_text")
        if st.button("▶️ Запустить обработку файла (Обработка текста)", key="process_file_text"):
            if not api_key:
                st.error("❌ API Key не указан!")
            else:
                df_out_text = process_file(
                    api_key=api_key,
                    model=selected_model_text,
                    system_prompt=system_prompt_text,
                    user_prompt=user_prompt_text,
                    df=df_text,
                    title_col=title_col_text,
                    response_format="csv",
                    max_tokens=max_tokens_text,
                    temperature=temperature_text,
                    top_p=top_p_text,
                    min_p=min_p_text,
                    top_k=top_k_text,
                    presence_penalty=presence_penalty_text,
                    frequency_penalty=frequency_penalty_text,
                    repetition_penalty=repetition_penalty_text,
                    chunk_size=10,
                    max_workers=max_workers_text
                )
                st.success("✅ Обработка завершена!")
                output_format = st.selectbox("📥 Формат вывода", ["csv", "txt"], key="output_format_text")
                if output_format == "csv":
                    csv_out_text = df_out_text.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Скачать результат (CSV)", data=csv_out_text, file_name="result.csv", mime="text/csv")
                else:
                    txt_out_text = df_out_text.to_csv(index=False, sep="|", header=False).encode("utf-8")
                    st.download_button("📥 Скачать результат (TXT)", data=txt_out_text, file_name="result.txt", mime="text/plain")

########################################
# Вкладка 2: Перевод текста
########################################
with tabs[1]:
    st.header("🌐 Перевод текста")
    with st.expander("🎨 Выбор пресета модели для перевода", expanded=True):
        preset_names_translate = list(PRESETS.keys())
        selected_preset_translate = st.selectbox("Выберите пресет для перевода", preset_names_translate, index=0)
        preset_translate = PRESETS[selected_preset_translate]
        system_prompt_translate = preset_translate["system_prompt"]
        max_tokens_translate = preset_translate["max_tokens"]
        temperature_translate = preset_translate["temperature"]
        top_p_translate = preset_translate["top_p"]
        min_p_translate = preset_translate["min_p"]
        top_k_translate = preset_translate["top_k"]
        presence_penalty_translate = preset_translate["presence_penalty"]
        frequency_penalty_translate = preset_translate["frequency_penalty"]
        repetition_penalty_translate = preset_translate["repetition_penalty"]
        if st.button("Сбросить настройки пресета для перевода", key="reset_preset_translate"):
            selected_preset_translate = "Default"
            preset_translate = PRESETS[selected_preset_translate]
            system_prompt_translate = preset_translate["system_prompt"]
            max_tokens_translate = preset_translate["max_tokens"]
            temperature_translate = preset_translate["temperature"]
            top_p_translate = preset_translate["top_p"]
            min_p_translate = preset_translate["min_p"]
            top_k_translate = preset_translate["top_k"]
            presence_penalty_translate = preset_translate["presence_penalty"]
            frequency_penalty_translate = preset_translate["frequency_penalty"]
            repetition_penalty_translate = preset_translate["repetition_penalty"]
    st.markdown("---")
    left_col_trans, right_col_trans = st.columns(2)
    with left_col_trans:
        st.subheader("📚 Список моделей для перевода")
        st.caption("Список моделей загружается из API Novita AI")
        if st.button("🔄 Обновить список моделей (Перевод текста)", key="refresh_models_translate"):
            if not api_key:
                st.error("❌ Ключ API пуст")
                st.session_state["model_list_translate"] = []
            else:
                model_list_translate = get_model_list(api_key)
                st.session_state["model_list_translate"] = model_list_translate
        if "model_list_translate" not in st.session_state:
            st.session_state["model_list_translate"] = []
        if st.session_state["model_list_translate"]:
            selected_model_translate = st.selectbox("✅ Выберите модель", st.session_state["model_list_translate"], key="select_model_translate")
        else:
            selected_model_translate = st.selectbox("✅ Выберите модель", ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"], key="select_model_default_translate")
    with right_col_trans:
        with st.expander("⚙️ Настройки генерации для перевода", expanded=True):
            st.subheader("⚙️ Параметры генерации для перевода")
            translate_output_format = st.selectbox("📥 Формат вывода перевода", ["csv", "txt"], key="translate_output_format")
            system_prompt_translate = st.text_area("📝 System Prompt для перевода", value=system_prompt_translate, key="system_prompt_translate")
            max_tokens_translate = st.slider("🔢 max_tokens (перевод)", min_value=0, max_value=64000, value=max_tokens_translate, step=1, key="max_tokens_translate")
            temperature_translate = st.slider("🌡️ temperature (перевод)", min_value=0.0, max_value=2.0, value=temperature_translate, step=0.01, key="temperature_translate")
            top_p_translate = st.slider("📊 top_p (перевод)", min_value=0.0, max_value=1.0, value=top_p_translate, step=0.01, key="top_p_translate")
            min_p_translate = st.slider("📉 min_p (перевод)", min_value=0.0, max_value=1.0, value=min_p_translate, step=0.01, key="min_p_translate")
            top_k_translate = st.slider("🔝 top_k (перевод)", min_value=0, max_value=100, value=top_k_translate, step=1, key="top_k_translate")
            presence_penalty_translate = st.slider("⚖️ presence_penalty (перевод)", min_value=0.0, max_value=2.0, value=presence_penalty_translate, step=0.01, key="presence_penalty_translate")
            frequency_penalty_translate = st.slider("📉 frequency_penalty (перевод)", min_value=0.0, max_value=2.0, value=frequency_penalty_translate, step=0.01, key="frequency_penalty_translate")
            repetition_penalty_translate = st.slider("🔁 repetition_penalty (перевод)", min_value=0.0, max_value=2.0, value=repetition_penalty_translate, step=0.01, key="repetition_penalty_translate")
    st.markdown("---")
    st.subheader("📝 Настройки перевода")
    languages = ["English", "Chinese", "Japanese", "Hindi"]
    source_language = st.selectbox("🔠 Исходный язык", languages, index=0, key="source_language")
    target_language = st.selectbox("🔡 Целевой язык", languages, index=1, key="target_language")
    if source_language == target_language:
        st.warning("⚠️ Исходный и целевой языки должны отличаться!")
    st.markdown("---")
    st.subheader("📂 Перевод данных из файла")
    with st.expander("📑 Настройки парсинга файла для перевода", expanded=True):
        delimiter_input_translate = st.text_input("🔸 Разделитель (delimiter) для перевода", value="|", key="delimiter_input_translate")
        column_input_translate = st.text_input("🔸 Названия колонок (через запятую) для перевода", value="id,title", key="column_input_translate")
    uploaded_file_translate = st.file_uploader("📤 Прикрепить файл для перевода (CSV или TXT, до 100000 строк)", type=["csv", "txt"], key="uploaded_file_translate")
    df_translate = None
    if uploaded_file_translate is not None:
        file_extension_translate = uploaded_file_translate.name.split(".")[-1].lower()
        try:
            if file_extension_translate == "csv":
                df_translate = pd.read_csv(uploaded_file_translate)
            else:
                content_translate = uploaded_file_translate.read().decode("utf-8")
                lines_translate = content_translate.splitlines()
                columns_translate = [c.strip() for c in column_input_translate.split(",")]
                parsed_lines_translate = []
                for line in lines_translate:
                    splitted_translate = line.split(delimiter_input_translate, maxsplit=len(columns_translate) - 1)
                    if len(splitted_translate) < len(columns_translate):
                        splitted_translate += [""] * (len(columns_translate) - len(splitted_translate))
                    parsed_lines_translate.append(splitted_translate)
                df_translate = pd.DataFrame(parsed_lines_translate, columns=columns_translate)
            st.write("### 📋 Предпросмотр файла для перевода")
            num_preview_translate = st.number_input("🔍 Количество строк для предпросмотра", min_value=1, max_value=100, value=10, key="num_preview_translate")
            st.dataframe(df_translate.head(num_preview_translate))
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла для перевода: {e}")
            df_translate = None
    if df_translate is not None:
        cols_translate = df_translate.columns.tolist()
        with st.expander("📂 Выбор колонок файла для перевода", expanded=True):
            id_col_translate = st.selectbox("🆔 Какая колонка является ID?", cols_translate, key="id_col_translate")
            title_col_translate = st.selectbox("📌 Какая колонка является заголовком для перевода?", cols_translate, key="title_col_translate")
            max_workers_translate = st.slider("🔄 Потоки (max_workers) для перевода", min_value=1, max_value=20, value=5, key="max_workers_translate")
        if st.button("▶️ Начать перевод", key="start_translation"):
            if not api_key:
                st.error("❌ API Key не указан!")
            elif source_language == target_language:
                st.error("❌ Исходный и целевой языки должны отличаться!")
            elif not title_col_translate:
                st.error("❌ Не выбрана колонка для перевода!")
            else:
                user_prompt_translate = f"Translate the following text from {source_language} to {target_language}:"
                df_translated = process_translation_file(
                    api_key=api_key,
                    model=selected_model_translate,
                    system_prompt=system_prompt_translate,
                    user_prompt=user_prompt_translate,
                    df=df_translate,
                    title_col=title_col_translate,
                    max_tokens=max_tokens_translate,
                    temperature=temperature_translate,
                    top_p=top_p_translate,
                    min_p=min_p_translate,
                    top_k=top_k_translate,
                    presence_penalty=presence_penalty_translate,
                    frequency_penalty=frequency_penalty_translate,
                    repetition_penalty=repetition_penalty_translate,
                    chunk_size=10,
                    max_workers=max_workers_translate
                )
                st.success("✅ Перевод завершен!")
                if translate_output_format == "csv":
                    csv_translated = df_translated.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Скачать переведенный файл (CSV)", data=csv_translated, file_name="translated_result.csv", mime="text/csv")
                else:
                    txt_translated = df_translated.to_csv(index=False, sep="|", header=False).encode("utf-8")
                    st.download_button("📥 Скачать переведенный файл (TXT)", data=txt_translated, file_name="translated_result.txt", mime="text/plain")

########################################
# Вкладка 3: Разделение файла
########################################
with tabs[2]:
    st.header("📂 Разделение файла")
    st.markdown("Загрузите файл (CSV или TXT) и укажите, на сколько строк нужно разбить файл.")
    split_size = st.number_input("Количество строк на часть", min_value=1, value=5000, step=100)
    uploaded_file_split = st.file_uploader("📤 Загрузите файл для разделения", type=["csv", "txt"], key="uploaded_file_split")
    if uploaded_file_split is not None:
        file_extension_split = uploaded_file_split.name.split(".")[-1].lower()
        try:
            if file_extension_split == "csv":
                df_split = pd.read_csv(uploaded_file_split)
                total_rows = len(df_split)
                st.write(f"Загружено строк: {total_rows}")
                num_parts = math.ceil(total_rows / split_size)
                st.write(f"Файл будет разбит на {num_parts} частей.")
                for i in range(num_parts):
                    part_df = df_split.iloc[i*split_size:(i+1)*split_size]
                    csv_part = part_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Скачать часть {i+1} (CSV)",
                        data=csv_part,
                        file_name=f"part_{i+1}.csv",
                        mime="text/csv"
                    )
            else:
                content = uploaded_file_split.read().decode("utf-8")
                lines = content.splitlines()
                total_lines = len(lines)
                st.write(f"Загружено строк: {total_lines}")
                num_parts = math.ceil(total_lines / split_size)
                st.write(f"Файл будет разбит на {num_parts} частей.")
                for i in range(num_parts):
                    part_lines = lines[i*split_size:(i+1)*split_size]
                    part_content = "\n".join(part_lines)
                    st.download_button(
                        label=f"Скачать часть {i+1} (TXT)",
                        data=part_content.encode("utf-8"),
                        file_name=f"part_{i+1}.txt",
                        mime="text/plain"
                    )
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

########################################
# Вкладка 4: Постобработка
########################################
with tabs[3]:
    st.header("🧹 Постобработка")
    st.markdown("Загрузите файл (CSV или TXT) и задайте паттерны, которые необходимо удалить из текста.")
    default_patterns = """Intense Deep Ass Fucking Session with Alex Greene and Brendan Patrick - Icon Male
Intense Black-on-Black Sauna Sex: Micah Brandt and Sean Zevran Heat Up the Hot House
Explicit Threesome Adventures: Clay Towers & Hans Berlin Heat Up Pride Studios
Explicit Anal Encounter: Hot Mess with Justin Brody & Boomer Banks from Cocky Boys"""
    harmful_patterns_input = st.text_area("Введите паттерны (по одному в строке)", value=default_patterns, height=150)
    harmful_patterns = [line.strip() for line in harmful_patterns_input.splitlines() if line.strip()]
    uploaded_file_post = st.file_uploader("📤 Загрузите файл для постобработки (CSV или TXT)", type=["csv", "txt"], key="uploaded_file_post")
    if uploaded_file_post is not None:
        file_extension_post = uploaded_file_post.name.split(".")[-1].lower()
        if file_extension_post == "csv":
            try:
                df_post = pd.read_csv(uploaded_file_post)
                st.write("### 📋 Предпросмотр файла")
                st.dataframe(df_post.head(10))
                cols_post = df_post.columns.tolist()
                text_col = st.selectbox("Выберите колонку для очистки", cols_post, key="text_col_post")
                if st.button("▶️ Запустить постобработку (CSV)", key="process_post_csv"):
                    df_cleaned = process_postprocessing_file(df_post, text_col, harmful_patterns)
                    st.success("✅ Постобработка завершена!")
                    csv_cleaned = df_cleaned.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Скачать очищенный файл (CSV)", data=csv_cleaned, file_name="cleaned_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Ошибка при чтении CSV: {e}")
        else:
            try:
                content_post = uploaded_file_post.read().decode("utf-8")
                lines_post = content_post.splitlines()
                st.write(f"Загружено строк: {len(lines_post)}")
                # Очищаем каждую строку по заданным паттернам
                cleaned_lines = [clean_text(line, harmful_patterns) for line in lines_post]
                cleaned_content = "\n".join(cleaned_lines)
                st.text_area("Предпросмотр очищенного текста", value=cleaned_content[:1000], height=200)
                st.download_button("📥 Скачать очищенный файл (TXT)", data=cleaned_content.encode("utf-8"), file_name="cleaned_result.txt", mime="text/plain")
            except Exception as e:
                st.error(f"Ошибка при обработке TXT: {e}")

