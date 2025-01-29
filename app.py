import streamlit as st
import requests
import json
import pandas as pd
import time
import concurrent.futures
import re

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

# Языковые настройки
SOURCE_LANGUAGES = {
    "Auto": "Auto-detect language",
    "English": "English source text",
    "Japanese": "Japanese source text (日本語)",
    "Chinese": "Chinese source text (中文)",
    "Hindi": "Hindi source text (हिन्दी)"
}

TARGET_LANGUAGES = {
    "English": "Translate to English",
    "Japanese": "Translate to Japanese (日本語)",
    "Chinese": "Translate to Chinese (中文)",
    "Hindi": "Translate to Hindi (हिन्दी)"
}

st.set_page_config(page_title="Novita AI Translation", layout="wide")

#######################################
# 2) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#######################################

def detect_primary_language(text: str) -> str:
    """
    Определяет основной язык текста по базовым признакам
    """
    japanese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f'])
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_chars = len([c for c in text if ord('a') <= ord(c.lower()) <= ord('z')])
    
    if japanese_chars > len(text) * 0.3:
        return "Japanese"
    elif chinese_chars > len(text) * 0.3:
        return "Chinese"
    elif english_chars > len(text) * 0.3:
        return "English"
    return "Auto"

def get_language_system_prompt(source_lang: str, target_lang: str, base_prompt: str) -> str:
    """
    Enhanced system prompt generator with better mixed language handling
    """
    if source_lang == target_lang:
        return base_prompt
    
    base_rules = """
    Translation Rules:
    1. Preserve all names exactly as they appear
    2. Keep technical terms intact
    3. Maintain the original tone and style
    4. Handle mixed language content appropriately
    5. Ensure consistent translation of repeated phrases
    """
    
    language_specific = {
        ("Auto", "Chinese"): """
            将所有文本翻译成中文：
            - 保持人名不变
            - 保持技术术语不变
            - 对于混合语言文本，将非中文部分翻译成中文
            - 保持原文的语气和风格
            - 使用自然的中文表达
            - 对于成人内容，使用适当委婉的表达方式
        """,
        ("Auto", "Japanese"): """
            すべてのテキストを日本語に翻訳：
            - 人名はそのまま保持
            - 技術用語はそのまま保持
            - 混合言語テキストの場合、日本語以外の部分を翻訳
            - 原文のトーンとスタイルを維持
            - 自然な日本語表現を使用
            - アダルトコンテンツには適切な婉曲表現を使用
        """,
        ("Auto", "English"): """
            Translate all text to English:
            - Keep personal names unchanged
            - Preserve technical terms
            - For mixed language text, translate non-English parts
            - Maintain original tone and style
            - Use natural English expressions
            - Use appropriate euphemisms for adult content
        """,
        ("English", "Chinese"): """
            Identify English text and translate to Chinese:
            - Keep all names in original form
            - Preserve technical terms
            - Translate only confirmed English parts
            - Use natural Chinese expressions
            - For adult content, use appropriate Chinese terms
        """,
        ("Japanese", "English"): """
            Identify Japanese text and translate to English:
            - Keep Japanese names in original form
            - Preserve technical terms
            - Translate only confirmed Japanese parts
            - Use natural English expressions
            - For adult content, use appropriate English terms
        """
    }
    
    mixed_language_handling = """
    Mixed Language Handling:
    1. Identify the primary language in each segment
    2. Preserve any intentionally mixed language elements
    3. Translate only the parts that match the source language
    4. Keep formatting and structure intact
    5. Handle adult content appropriately in target language
    """
    
    pair_key = (source_lang, target_lang)
    specific_instructions = language_specific.get(pair_key, "")
    
    full_prompt = f"""
    {base_prompt}
    
    {base_rules}
    
    {specific_instructions}
    
    {mixed_language_handling}
    
    Current translation direction: {source_lang} → {target_lang}
    """
    
    return full_prompt

def custom_postprocess_text(text: str) -> str:
    """
    Убираем 'fucking' (в любом регистре) только в начале строки.
    """
    pattern_start = re.compile(r'^(fucking\s*)', re.IGNORECASE)
    text = pattern_start.sub('', text)
    return text

def get_model_list(api_key: str):
    """Загружаем список доступных моделей через эндпоинт Novita AI"""
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
    """Функция для синхронного (не-стримингового) chat-комплишена с retries на 429."""
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
    source_lang: str,
    target_lang: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float
):
    """Обновленная функция обработки одной строки с поддержкой языков"""
    if source_lang == "Auto":
        detected_lang = detect_primary_language(row_text)
        actual_source = detected_lang
    else:
        actual_source = source_lang

    final_prompt = get_language_system_prompt(actual_source, target_lang, system_prompt)
    
    messages = [
        {"role": "system", "content": final_prompt},
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

    return custom_postprocess_text(raw_response)

def process_file(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    title_col: str,
    source_lang: str,
    target_lang: str,
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
    """Параллельно обрабатываем загруженный файл построчно (или чанками)."""

    progress_bar = st.progress(0)
    time_placeholder = st.empty()

    results = []
    total_rows = len(df)

    start_time = time.time()
    lines_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start_time = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)

        chunk_indices = list(df.index[range(start_idx, end_idx)])
        chunk_size_actual = len(chunk_indices)
        chunk_results = [None] * chunk_size_actual

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
                    source_lang,
                    target_lang,
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

        lines_processed += chunk_size_actual
        progress_bar.progress(lines_processed / total_rows)

        time_for_chunk = time.time() - chunk_start_time
        if chunk_size_actual > 0:
            time_per_line = time_for_chunk / chunk_size_actual
            lines_left = total_rows - lines_processed
            if time_per_line > 0:
                est_time_left_sec = lines_left * time_per_line
                if est_time_left_sec < 60:
                    time_text = f"~{est_time_left_sec:.1f} сек."
                else:
                    est_time_left_min = est_time_left_sec / 60.0
                    time_text = f"~{est_time_left_min:.1f} мин."
                time_placeholder.info(f"Примерное оставшееся время: {time_text}")

    df_out = df.copy()
    df_out["translation"] = results

    elapsed = time.time() - start_time
    time_placeholder.success(f"Обработка завершена за {elapsed:.1f} секунд.")

    return df_out

#######################################
# 4) ИНТЕРФЕЙС
#######################################

st.title("🌎 Novita AI Batch Translation")

# Три колонки для лучшей организации
left_col, middle_col, right_col = st.columns([1, 1, 1])

########################################
# Левая колонка: Список моделей
########################################
with left_col:
    st.markdown("#### Model Selection")
    st.caption("Models are loaded from Novita AI API")

    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

    if st.button("Update Model List"):
        if not api_key:
            st.error("API Key is empty")
            model_list = []
        else:
            model_list = get_model_list(api_key)
            st.session_state["model_list"] = model_list

    if "model_list" not in st.session_state:
        st.session_state["model_list"] = []

    if len(st.session_state["model_list"]) > 0:
        selected_model = st.selectbox("Select Model", st.session_state["model_list"])
    else:
        selected_model = st.selectbox(
            "Select Model",
            ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"]
        )

########################################
# Средняя колонка: Настройки генерации
########################################
with middle_col:
    st.markdown("#### Generation Parameters")
    output_format = st.selectbox("Output Format", ["csv", "txt"])
    system_prompt = st.text_area(
        "System Prompt", 
        value="You are a professional translator. Translate the following text while preserving names, terms and maintaining the original style."
    )

########################################
# Правая колонка: Дополнительные параметры
########################################
with right_col:
    st.markdown("#### Additional Parameters")
    max_tokens = st.slider("max_tokens", min_value=0, max_value=64000, value=512, step=1)
    temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    min_p = st.slider("min_p", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    top_k = st.slider("top_k", min_value=0, max_value=100, value=40, step=1)
    presence_penalty = st.slider("presence_penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    frequency_penalty = st.slider("frequency_penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    repetition_penalty = st.slider("repetition_penalty", min_value=0.0, max_value=2.0, value=1.0, step=0.01)

# Разделительная линия
st.markdown("---")

########################################
# Блок одиночного перевода
########################################
st.subheader("Single Text Translation")
user_prompt_single = st.text_area("Enter text to translate")

# Выбор языков для одиночного перевода
single_source_lang = st.selectbox(
    "Source Language",
    options=list(SOURCE_LANGUAGES.keys()),
    format_func=lambda x: SOURCE_LANGUAGES[x],
    key="single_source"
)

single_target_lang = st.selectbox(
    "Target Language",
    options=list(TARGET_LANGUAGES.keys()),
    format_func=lambda x: TARGET_LANGUAGES[x],
    key="single_target"
)

if st.button("Translate Single Text"):
    if not api_key:
        st.error("API Key is missing!")
    else:
        final_prompt = get_language_system_prompt(single_source_lang, single_target_lang, system_prompt)
        from_text = [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": user_prompt_single}
        ]
        st.info("Sending request...")
        raw_response = chat_completion_request(
            api_key=api_key,
            messages=from_text,
            model=selected_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty
        )
        final_response = custom_postprocess_text(raw_response)
        st.success("Translation complete!")
        st.text_area("Translated Text", value=final_response, height=200)

# Разделительная линия
st.markdown("---")

########################################
# Блок обработки файла
########################################
st.subheader("Batch File Translation")

user_prompt = st.text_area("Additional translation instructions (optional)")

st.markdown("##### File Parsing Settings")
delimiter_input = st.text_input("Delimiter", value="|")
column_input = st.text_input("Column names (comma-separated)", value="id,title")

uploaded_file = st.file_uploader("Upload file (CSV or TXT, up to 100000 lines)", type=["csv", "txt"])

# Выбор языков для пакетного перевода
batch_source_lang = st.selectbox(
    "Source Language for Batch Translation",
    options=list(SOURCE_LANGUAGES.keys()),
    format_func=lambda x: SOURCE_LANGUAGES[x],
    key="batch_source"
)

batch_target_lang = st.selectbox(
    "Target Language for Batch Translation",
    options=list(TARGET_LANGUAGES.keys()),
    format_func=lambda x: TARGET_LANGUAGES[x],
    key="batch_target"
)

df = None
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    try:
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
            lines = content.splitlines()

            columns = [c.strip() for c in column_input.split(",")]

            parsed_lines = []
            for line in lines:
                splitted = line.split(delimiter_input, maxsplit=len(columns) - 1)
                parsed_lines.append(splitted)

            df = pd.DataFrame(parsed_lines, columns=columns)

        st.write("### File Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None

if df is not None:
    cols = df.columns.tolist()
    title_col = st.selectbox("Which column contains text to translate?", cols)

    max_workers = st.slider("Threads (max_workers)", min_value=1, max_value=20, value=5)

    if st.button("Start Batch Translation"):
        if not api_key:
            st.error("API Key is missing!")
        else:
            row_count = len(df)
            if row_count > 100000:
                st.warning(f"File contains {row_count} rows. This exceeds the recommended limit of 100000.")
            st.info("Starting translation, please wait...")

            df_out = process_file(
                api_key=api_key,
                model=selected_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                df=df,
                title_col=title_col,
                source_lang=batch_source_lang,
                target_lang=batch_target_lang,
                response_format=output_format,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                chunk_size=10,
                max_workers=max_workers
            )

            st.success("Translation complete!")

            # Download options
            if output_format == "csv":
                csv_out = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download Result (CSV)", data=csv_out, file_name="translations.csv", mime="text/csv")
            else:
                txt_out = df_out.to_csv(index=False, sep="|", header=False).encode("utf-8")
                st.download_button("Download Result (TXT)", data=txt_out, file_name="translations.txt", mime="text/plain")

            st.write("### Results")
            st.write("Processing completed, rows processed:", len(df_out))
