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

# Словарь доступных языков
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

st.set_page_config(page_title="Novita AI Batch Processor", layout="wide")

#######################################
# 2) Вспомогательные ФУНКЦИИ
#######################################

def get_language_system_prompt(source_lang: str, target_lang: str, base_prompt: str) -> str:
    """
    Generate system prompt based on source and target languages
    """
    if source_lang == target_lang:
        return base_prompt
    
    language_instructions = {
        ("English", "Japanese"): """
            Translate the following English text to Japanese.
            Rules:
            - Keep names unchanged
            - Preserve technical terms
            - Maintain the original style and tone
            - Ensure natural Japanese expression
        """,
        ("English", "Chinese"): """
            Translate the following English text to Chinese.
            Rules:
            - Keep names unchanged
            - Preserve technical terms
            - Maintain the original style and tone
            - Use appropriate Chinese characters
        """,
        ("English", "Hindi"): """
            Translate the following English text to Hindi.
            Rules:
            - Keep names unchanged
            - Preserve technical terms
            - Maintain the original style and tone
            - Use proper Hindi grammar
        """,
        # Add other language pair combinations here
    }
    
    # Get instructions for the language pair, or generate generic instructions
    pair_key = (source_lang, target_lang)
    if pair_key in language_instructions:
        return f"{base_prompt}\n\n{language_instructions[pair_key]}"
    else:
        return f"{base_prompt}\n\nTranslate from {source_lang} to {target_lang}, keeping names and technical terms unchanged."

def custom_postprocess_text(text: str) -> str:
    """
    Убираем 'fucking' (в любом регистре) только в начале строки.
    Если в середине — оставляем.
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
                # rate limit exceeded, ждем 2 сек
                time.sleep(2)
                # и попробуем снова
                continue
            else:
                return f"Ошибка: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Исключение: {e}"
    # Если все попытки исчерпаны
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
    """Функция-обёртка для параллельного вызова."""
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

    # Постобработка: убираем banned words
    final_response = custom_postprocess_text(raw_response)
    return final_response

def process_file(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    title_col: str,  # Название колонки, которую надо переписать
    response_format: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    chunk_size: int = 10,  # фиксируем 10 строк в чанке
    max_workers: int = 5  # Количество потоков
):
    """Параллельно обрабатываем загруженный файл построчно (или чанками)."""

    progress_bar = st.progress(0)
    time_placeholder = st.empty()  # для отображения оставшегося времени

    results = []
    total_rows = len(df)

    start_time = time.time()
    lines_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start_time = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)

        # Берём индексы строк в этом чанке
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

        # Расширяем общий список результатов
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

    # Создаем копию df с новым столбцом
    df_out = df.copy()
    df_out["rewrite"] = results

    elapsed = time.time() - start_time
    time_placeholder.success(f"Обработка завершена за {elapsed:.1f} секунд.")

    return df_out

#######################################
# 4) ИНТЕРФЕЙС
#######################################

st.title("🧠 Novita AI Batch Processing")

# Три колонки для лучшей организации
left_col, middle_col, right_col = st.columns([1, 1, 1])

########################################
# Левая колонка: Список моделей и язык
########################################
with left_col:
    st.markdown("#### Модели и язык")
    st.caption("Список моделей загружается из API Novita AI")

    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    
    # Добавляем выбор языка
    source_language = st.selectbox(
    "Source Language",
    options=list(SOURCE_LANGUAGES.keys()),
    format_func=lambda x: SOURCE_LANGUAGES[x]
)

target_language = st.selectbox(
    "Target Language",
    options=list(TARGET_LANGUAGES.keys()),
    format_func=lambda x: TARGET_LANGUAGES[x]
)

if st.button("Обновить список моделей"):
        if not api_key:
            st.error("Ключ API пуст")
            model_list = []
        else:
            model_list = get_model_list(api_key)
            st.session_state["model_list"] = model_list

if "model_list" not in st.session_state:
        st.session_state["model_list"] = []

if len(st.session_state["model_list"]) > 0:
        selected_model = st.selectbox("Выберите модель", st.session_state["model_list"])
else:
        selected_model = st.selectbox(
            "Выберите модель",
            ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"]
        )

########################################
# Средняя колонка: Настройки генерации
########################################
with middle_col:
    st.markdown("#### Параметры генерации")
    output_format = st.selectbox("Output Format", ["csv", "txt"])
    system_prompt = st.text_area("System Prompt", value="Act like you are a helpful assistant.")

########################################
# Правая колонка: Дополнительные параметры
########################################
with right_col:
    st.markdown("#### Дополнительные параметры")
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
# Поле одиночного промпта
########################################
st.subheader("Одиночный промпт")
user_prompt_single = st.text_area("Введите ваш промпт для одиночной генерации")

if st.button("Отправить одиночный промпт"):
    if not api_key:
        st.error("API Key не указан!")
    else:
        from_text = [
            {"role": "system", "content": get_language_system_prompt(target_language, system_prompt)},
            {"role": "user", "content": user_prompt_single}
        ]
        st.info("Отправляем запрос...")
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
        st.success("Результат получен!")
        st.text_area("Ответ от модели", value=final_response, height=200)

# Разделительная линия
st.markdown("---")

########################################
# Блок обработки файла
########################################
st.subheader("Обработка данных из файла")

user_prompt = st.text_area("Пользовательский промпт (дополнительно к заголовку)")

st.markdown("##### Настройка парсинга TXT/CSV")
delimiter_input = st.text_input("Разделитель (delimiter)", value="|")
column_input = st.text_input("Названия колонок (через запятую)", value="id,title")

uploaded_file = st.file_uploader("Прикрепить файл (CSV или TXT, до 100000 строк)", type=["csv", "txt"])

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

        st.write("### Предпросмотр файла")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        df = None

if df is not None:
    cols = df.columns.tolist()
    title_col = st.selectbox("Какая колонка является заголовком?", cols)

    # Ползунок для выбора кол-ва потоков
    max_workers = st.slider("Потоки (max_workers)", min_value=1, max_value=20, value=5)

    if st.button("Запустить обработку файла"):
        if not api_key:
            st.error("API Key не указан!")
        else:
            row_count = len(df)
            if row_count > 100000:
                st.warning(f"Файл содержит {row_count} строк. Это превышает рекомендованный лимит в 100000.")
            st.info("Начинаем обработку, пожалуйста подождите...")

            df_out = process_file(
                api_key=api_key,
                model=selected_model,
                system_prompt=get_language_system_prompt(source_language, target_language, system_prompt),
                user_prompt=user_prompt,
                df=df,
                title_col=title_col,
                response_format="csv",
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

            st.success("Обработка завершена!")

            # Скачивание
            if output_format == "csv":
                csv_out = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Скачать результат (CSV)", data=csv_out, file_name="result.csv", mime="text/csv")
            else:
                txt_out = df_out.to_csv(index=False, sep="|", header=False).encode("utf-8")
                st.download_button("Скачать результат (TXT)", data=txt_out, file_name="result.txt", mime="text/plain")

            st.write("### Логи")
            st.write("Обработка завершена, строк обработано:", len(df_out))
