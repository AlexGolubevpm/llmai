import streamlit as st
import streamlit_server_state as server_state
from streamlit_autorefresh import st_autorefresh
import concurrent.futures
import uuid
import time
import pandas as pd
import requests
import json
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

st.set_page_config(page_title="🧠 Novita AI Batch Processor", layout="wide")

#######################################
# 2) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#######################################

def custom_postprocess_text(text: str) -> str:
    """
    Убираем 'fucking' (в любом регистре) только в начале строки.
    Также удаляем все двойные кавычки из текста.
    """
    # Удаляем 'fucking' в начале строки
    pattern_start = re.compile(r'^(fucking\s*)', re.IGNORECASE)
    text = pattern_start.sub('', text)
    
    # Удаляем все двойные кавычки
    text = text.replace('"', '')
    
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

    # Постобработка: убираем banned words и двойные кавычки
    final_response = custom_postprocess_text(raw_response)
    return final_response

def process_file(
    task_id: str,
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

    tasks = server_state.get('tasks')
    tasks[task_id]['status'] = 'running'
    tasks[task_id]['start_time'] = time.time()
    server_state.set('tasks', tasks)

    results = []
    total_rows = len(df)

    start_time = time.time()
    lines_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start_time = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)

        # Берём индексы строк в этом чанке
        chunk_indices = list(df.index[start_idx:end_idx])
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
        progress = lines_processed / total_rows
        tasks[task_id]['progress'] = progress
        server_state.set('tasks', tasks)

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
                tasks[task_id]['estimated_time_left'] = time_text
                server_state.set('tasks', tasks)

    # Создаем копию df с новым столбцом
    df_out = df.copy()
    df_out["rewrite"] = results

    elapsed = time.time() - start_time
    tasks[task_id]['status'] = 'completed'
    tasks[task_id]['end_time'] = time.time()
    server_state.set('tasks', tasks)

    return df_out

# ======= Новые функции для перевода =======

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
    """Функция для перевода текста с retries на 429."""
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

    # Постобработка: убираем banned words и двойные кавычки
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
    """Функция-обёртка для параллельного вызова перевода."""
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
    task_id: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    title_col: str,  # Название колонки, которую надо перевести
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
    """Параллельно переводим загруженный файл построчно (или чанками)."""

    tasks = server_state.get('tasks')
    tasks[task_id]['status'] = 'running'
    tasks[task_id]['start_time'] = time.time()
    server_state.set('tasks', tasks)

    results = []
    total_rows = len(df)

    start_time = time.time()
    lines_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start_time = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)

        # Берём индексы строк в этом чанке
        chunk_indices = list(df.index[start_idx:end_idx])
        chunk_size_actual = len(chunk_indices)
        chunk_results = [None] * chunk_size_actual

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

        # Расширяем общий список результатов
        results.extend(chunk_results)

        lines_processed += chunk_size_actual
        progress = lines_processed / total_rows
        tasks[task_id]['progress'] = progress
        server_state.set('tasks', tasks)

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
                tasks[task_id]['estimated_time_left'] = time_text
                server_state.set('tasks', tasks)

    # Создаем копию df с новым столбцом
    df_out = df.copy()
    df_out["translated_title"] = results

    elapsed = time.time() - start_time
    tasks[task_id]['status'] = 'completed'
    tasks[task_id]['end_time'] = time.time()
    server_state.set('tasks', tasks)

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
    # Можно добавить больше пресетов по необходимости
}

#######################################
# 4) ИНТЕРФЕЙС
#######################################

# Инициализация хранилища задач
if not server_state.get('tasks'):
    server_state.set('tasks', {})

# Создаем пул потоков для выполнения задач
executor_main = concurrent.futures.ThreadPoolExecutor(max_workers=5)

st.title("🧠 Novita AI Batch Processor")

# Поле ввода API Key, доступное во всех вкладках
st.sidebar.header("🔑 Настройки API")
api_key = st.sidebar.text_input("API Key", value=DEFAULT_API_KEY, type="password")

# Создаем вкладки для разделения функционала
tabs = st.tabs(["🔄 Обработка текста", "🌐 Перевод текста", "📊 Отслеживание задач"])

########################################
# Вкладка 1: Обработка текста
########################################
with tabs[0]:
    st.header("🔄 Обработка текста")

    # Добавляем выбор пресета
    with st.expander("🎨 Выбор пресета модели", expanded=True):
        preset_names = list(PRESETS.keys())
        selected_preset = st.selectbox("Выберите пресет", preset_names, index=0)

        # Получаем параметры выбранного пресета
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

        # Кнопка для сброса к стандартным настройкам (опционально)
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

    # Две колонки
    left_col, right_col = st.columns([1, 1])

    ########################################
    # Левая колонка: Список моделей
    ########################################
    with left_col:
        st.subheader("📚 Список моделей для обработки текста")
        st.caption("🔄 Список моделей загружается из API Novita AI")

        if st.button("🔄 Обновить список моделей (Обработка текста)", key="refresh_models_text"):
            if not api_key:
                st.error("❌ Ключ API пуст")
            else:
                # Создаём задачу для загрузки моделей
                task_id = uuid.uuid4().hex
                tasks = server_state.get('tasks')
                tasks[task_id] = {
                    'status': 'loading_models_text',
                    'progress': 0.0,
                    'result': None,
                    'start_time': time.time(),
                    'end_time': None,
                    'type': 'loading_models_text'
                }
                server_state.set('tasks', tasks)

                # Отправляем задачу в пул потоков
                executor_main.submit(
                    load_models_task,
                    task_id,
                    api_key
                )

                st.success(f"✅ Задача загрузки моделей запущена! Ваш Task ID: {task_id}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

        # Автоматическое обновление списка моделей через autorefresh
        st_autorefresh(interval=2000, limit=100, key="autorefresh_models_text")

        # Получаем список моделей из server_state
        tasks = server_state.get('tasks')
        model_list_text = []
        for task_id, task in tasks.items():
            if task.get('status') == 'completed' and task.get('type') == 'loading_models_text':
                model_list_text = task.get('result', [])
                break

        if len(model_list_text) > 0:
            selected_model_text = st.selectbox("✅ Выберите модель для обработки текста", model_list_text, key="select_model_text")
        else:
            selected_model_text = st.selectbox(
                "✅ Выберите модель для обработки текста",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_text"
            )

    ########################################
    # Правая колонка: Настройки генерации
    ########################################
    with right_col:
        with st.expander("⚙️ Настройки генерации", expanded=True):
            st.subheader("⚙️ Параметры генерации для обработки текста")

            # Здесь пользователи могут изменить параметры пресета при необходимости
            system_prompt_text = st.text_area("📝 System Prompt", value=system_prompt_text, key="system_prompt_text")

            max_tokens_text = st.slider("🔢 max_tokens", min_value=0, max_value=64000, value=max_tokens_text, step=1, key="max_tokens_text")
            temperature_text = st.slider("🌡️ temperature", min_value=0.0, max_value=2.0, value=temperature_text, step=0.01, key="temperature_text")
            top_p_text = st.slider("📊 top_p", min_value=0.0, max_value=1.0, value=top_p_text, step=0.01, key="top_p_text")
            min_p_text = st.slider("📉 min_p", min_value=0.0, max_value=1.0, value=min_p_text, step=0.01, key="min_p_text")
            top_k_text = st.slider("🔝 top_k", min_value=0, max_value=100, value=top_k_text, step=1, key="top_k_text")
            presence_penalty_text = st.slider("⚖️ presence_penalty", min_value=0.0, max_value=2.0, value=presence_penalty_text, step=0.01, key="presence_penalty_text")
            frequency_penalty_text = st.slider("📉 frequency_penalty", min_value=0.0, max_value=2.0, value=frequency_penalty_text, step=0.01, key="frequency_penalty_text")
            repetition_penalty_text = st.slider("🔁 repetition_penalty", min_value=0.0, max_value=2.0, value=repetition_penalty_text, step=0.01, key="repetition_penalty_text")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Поле одиночного промпта (не обязательно)
    ########################################
    st.subheader("📝 Одиночный промпт")
    user_prompt_single_text = st.text_area("Введите ваш промпт для одиночной генерации", key="user_prompt_single_text")

    if st.button("🚀 Отправить одиночный промпт (Обработка текста)", key="submit_single_text"):
        if not api_key:
            st.error("❌ API Key не указан!")
        elif not user_prompt_single_text.strip():
            st.error("❌ Промпт не может быть пустым!")
        else:
            from_text = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt_single_text}
            ]
            st.info("🔄 Отправляем запрос...")
            raw_response = chat_completion_request(
                api_key=api_key,
                messages=from_text,
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
            # Можем вызвать custom_postprocess_text, если нужно
            final_response = custom_postprocess_text(raw_response)
            st.success("✅ Результат получен!")
            st.text_area("📄 Ответ от модели", value=final_response, height=200)

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Блок обработки файла
    ########################################
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
                        # Заполняем недостающие колонки пустыми строками
                        splitted += [""] * (len(columns) - len(splitted))
                    parsed_lines.append(splitted)

                df_text = pd.DataFrame(parsed_lines, columns=columns)

            st.write("### 📋 Предпросмотр файла")
            num_preview = st.number_input("🔍 Количество строк для предпросмотра", min_value=1, max_value=100, value=10, key="num_preview_text")
            st.dataframe(df_text.head(num_preview))
        except Exception as e:
            st.error(f"❌ Ошибка при чтении файла: {e}")
            df_text = None

    if df_text is not None:
        cols_text = df_text.columns.tolist()
        with st.expander("📂 Выбор колонок файла", expanded=True):
            title_col_text = st.selectbox("📌 Какая колонка является заголовком?", cols_text, key="title_col_text")

            # Ползунок для выбора кол-ва потоков
            max_workers_text = st.slider("🔄 Потоки (max_workers)", min_value=1, max_value=20, value=5, key="max_workers_text")

        if st.button("▶️ Запустить обработку файла (Обработка текста)", key="process_file_text"):
            if not api_key:
                st.error("❌ API Key не указан!")
            else:
                row_count = len(df_text)
                if row_count > 100000:
                    st.warning(f"⚠️ Файл содержит {row_count} строк. Это превышает рекомендованный лимит в 100000.")
                st.info("🔄 Начинаем обработку, пожалуйста подождите...")

                # Создаем уникальный ID для задачи
                task_id = uuid.uuid4().hex

                # Инициализируем задачу в server_state
                tasks = server_state.get('tasks')
                tasks[task_id] = {
                    'status': 'queued',
                    'progress': 0.0,
                    'result': None,
                    'start_time': None,
                    'end_time': None,
                    'type': 'processing_text'
                }
                server_state.set('tasks', tasks)

                # Отправляем задачу в пул потоков
                executor_main.submit(
                    process_file,
                    task_id,
                    api_key,
                    selected_model_text,
                    system_prompt_text,
                    user_prompt_text,
                    df_text,
                    title_col_text,
                    "csv",  # response_format, можно использовать при необходимости
                    max_tokens_text,
                    temperature_text,
                    top_p_text,
                    min_p_text,
                    top_k_text,
                    presence_penalty_text,
                    frequency_penalty_text,
                    repetition_penalty_text,
                    chunk_size=10,
                    max_workers=max_workers_text
                )

                st.success(f"✅ Задача запущена! Ваш Task ID: {task_id}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

    # Разделительная линия
    st.markdown("---")

########################################
# Вкладка 2: Перевод текста
########################################
with tabs[1]:
    st.header("🌐 Перевод текста")

    # Добавляем выбор пресета
    with st.expander("🎨 Выбор пресета модели для перевода", expanded=True):
        preset_names_translate = list(PRESETS.keys())
        selected_preset_translate = st.selectbox("Выберите пресет для перевода", preset_names_translate, index=0)

        # Получаем параметры выбранного пресета
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

        # Кнопка для сброса к стандартным настройкам (опционально)
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

    # Две колонки
    left_col_trans, right_col_trans = st.columns([1, 1])

    ########################################
    # Левая колонка: Список моделей для перевода
    ########################################
    with left_col_trans:
        st.subheader("📚 Список моделей для перевода текста")
        st.caption("🔄 Список моделей загружается из API Novita AI")

        if st.button("🔄 Обновить список моделей (Перевод текста)", key="refresh_models_translate"):
            if not api_key:
                st.error("❌ Ключ API пуст")
            else:
                # Создаём задачу для загрузки моделей
                task_id = uuid.uuid4().hex
                tasks = server_state.get('tasks')
                tasks[task_id] = {
                    'status': 'loading_models_translate',
                    'progress': 0.0,
                    'result': None,
                    'start_time': time.time(),
                    'end_time': None,
                    'type': 'loading_models_translate'
                }
                server_state.set('tasks', tasks)

                # Отправляем задачу в пул потоков
                executor_main.submit(
                    load_models_translate_task,
                    task_id,
                    api_key
                )

                st.success(f"✅ Задача загрузки моделей для перевода запущена! Ваш Task ID: {task_id}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

        # Автоматическое обновление списка моделей через autorefresh
        st_autorefresh(interval=2000, limit=100, key="autorefresh_models_translate")

        # Получаем список моделей из server_state
        tasks = server_state.get('tasks')
        model_list_translate = []
        for task_id, task in tasks.items():
            if task.get('status') == 'completed' and task.get('type') == 'loading_models_translate':
                model_list_translate = task.get('result', [])
                break

        if len(model_list_translate) > 0:
            selected_model_translate = st.selectbox("✅ Выберите модель для перевода текста", model_list_translate, key="select_model_translate")
        else:
            selected_model_translate = st.selectbox(
                "✅ Выберите модель для перевода текста",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_translate"
            )

    ########################################
    # Правая колонка: Настройки генерации для перевода
    ########################################
    with right_col_trans:
        with st.expander("⚙️ Настройки генерации для перевода", expanded=True):
            st.subheader("⚙️ Параметры генерации для перевода текста")
            translate_output_format = st.selectbox("📥 Формат вывода перевода", ["csv", "txt"], key="translate_output_format")  # CSV или TXT
            system_prompt_translate = st.text_area("📝 System Prompt для перевода", value=system_prompt_translate, key="system_prompt_translate")

            max_tokens_translate = st.slider("🔢 max_tokens (перевод)", min_value=0, max_value=64000, value=max_tokens_translate, step=1, key="max_tokens_translate")
            temperature_translate = st.slider("🌡️ temperature (перевод)", min_value=0.0, max_value=2.0, value=temperature_translate, step=0.01, key="temperature_translate")
            top_p_translate = st.slider("📊 top_p (перевод)", min_value=0.0, max_value=1.0, value=top_p_translate, step=0.01, key="top_p_translate")
            min_p_translate = st.slider("📉 min_p (перевод)", min_value=0.0, max_value=1.0, value=min_p_translate, step=0.01, key="min_p_translate")
            top_k_translate = st.slider("🔝 top_k (перевод)", min_value=0, max_value=100, value=top_k_translate, step=1, key="top_k_translate")
            presence_penalty_translate = st.slider("⚖️ presence_penalty (перевод)", min_value=0.0, max_value=2.0, value=presence_penalty_translate, step=0.01, key="presence_penalty_translate")
            frequency_penalty_translate = st.slider("📉 frequency_penalty (перевод)", min_value=0.0, max_value=2.0, value=frequency_penalty_translate, step=0.01, key="frequency_penalty_translate")
            repetition_penalty_translate = st.slider("🔁 repetition_penalty (перевод)", min_value=0.0, max_value=2.0, value=repetition_penalty_translate, step=0.01, key="repetition_penalty_translate")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Поле выбора языков и настройки перевода
    ########################################
    st.subheader("📝 Настройки перевода")

    languages = ["English", "Chinese", "Japanese", "Hindi"]
    source_language = st.selectbox("🔠 Исходный язык", languages, index=0, key="source_language")
    target_language = st.selectbox("🔡 Целевой язык", languages, index=1, key="target_language")

    if source_language == target_language:
        st.warning("⚠️ Исходный и целевой языки должны отличаться!")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Блок обработки файла для перевода
    ########################################
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
                        # Заполняем недостающие колонки пустыми строками
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

            # Ползунок для выбора кол-ва потоков
            max_workers_translate = st.slider("🔄 Потоки (max_workers) для перевода", min_value=1, max_value=20, value=5, key="max_workers_translate")

        if st.button("▶️ Начать перевод", key="start_translation"):
            if not api_key:
                st.error("❌ API Key не указан!")
            elif source_language == target_language:
                st.error("❌ Исходный и целевой языки должны отличаться!")
            elif not title_col_translate:
                st.error("❌ Не выбрана колонка для перевода!")
            else:
                row_count_translate = len(df_translate)
                if row_count_translate > 100000:
                    st.warning(f"⚠️ Файл содержит {row_count_translate} строк. Это превышает рекомендованный лимит в 100000.")
                st.info("🔄 Начинаем перевод, пожалуйста подождите...")

                # Создаем уникальный ID для задачи
                task_id = uuid.uuid4().hex

                # Инициализируем задачу в server_state
                tasks = server_state.get('tasks')
                tasks[task_id] = {
                    'status': 'queued',
                    'progress': 0.0,
                    'result': None,
                    'start_time': None,
                    'end_time': None,
                    'type': 'translation'
                }
                server_state.set('tasks', tasks)

                # Пользовательский промпт для перевода
                user_prompt_translate = f"Translate the following text from {source_language} to {target_language}:"

                # Отправляем задачу в пул потоков
                executor_main.submit(
                    process_translation_file,
                    task_id,
                    api_key,
                    selected_model_translate,
                    system_prompt_translate,
                    user_prompt_translate,
                    df_translate,
                    title_col_translate,
                    max_tokens_translate,
                    temperature_translate,
                    top_p_translate,
                    min_p_translate,
                    top_k_translate,
                    presence_penalty_translate,
                    frequency_penalty_translate,
                    repetition_penalty_translate,
                    chunk_size=10,
                    max_workers=max_workers_translate
                )

                st.success(f"✅ Задача запущена! Ваш Task ID: {task_id}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

    # Разделительная линия
    st.markdown("---")

########################################
# Вкладка 3: Отслеживание задач
########################################
with tabs[2]:
    st.header("📊 Отслеживание задач")

    tasks = server_state.get('tasks')

    if not tasks:
        st.info("Нет активных задач.")
    else:
        task_ids = list(tasks.keys())
        selected_task_id = st.selectbox("🆔 Выберите Task ID для просмотра", task_ids, key="selected_task_id_tracking")

        if selected_task_id:
            task = tasks[selected_task_id]
            st.write(f"**Task ID:** {selected_task_id}")
            st.write(f"**Тип задачи:** {task.get('type')}")
            st.write(f"**Статус:** {task.get('status')}")
            progress = task.get('progress', 0.0)
            st.progress(progress)

            if task.get('start_time'):
                st.write(f"**Начало:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['start_time']))}")
            if task.get('end_time'):
                st.write(f"**Завершение:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['end_time']))}")
            else:
                est_time_left = task.get('estimated_time_left', 'Неизвестно')
                st.write(f"**Примерное оставшееся время:** {est_time_left}")

            if task.get('status') == 'completed':
                if task.get('type') == 'processing_text':
                    st.write("**Результат:** Обработка текста завершена.")
                    # Предложить скачать результат, если нужно
                    # Здесь можно добавить ссылку на файл или другие действия
                elif task.get('type') == 'translation':
                    st.write("**Результат:** Перевод завершен.")
                    # Предложить скачать результат, если нужно
            elif task.get('status') == 'failed':
                st.error(f"**Ошибка:** {task.get('result')}")

    # Автоматическое обновление страницы для обновления статуса задач
    st_autorefresh(interval=5000, limit=100, key="autorefresh_tasks_tracking")

#######################################
# Дополнительные Функции для Загрузки Моделей
#######################################

def load_models_task(task_id: str, api_key: str):
    """Функция для загрузки списка моделей и обновления состояния задачи."""
    tasks = server_state.get('tasks')
    tasks[task_id]['status'] = 'running'
    server_state.set('tasks', tasks)

    models = get_model_list(api_key)

    if isinstance(models, list):
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = models
    else:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['result'] = models  # Здесь models содержит сообщение об ошибке

    tasks[task_id]['end_time'] = time.time()
    server_state.set('tasks', tasks)

def load_models_translate_task(task_id: str, api_key: str):
    """Функция для загрузки списка моделей для перевода и обновления состояния задачи."""
    tasks = server_state.get('tasks')
    tasks[task_id]['status'] = 'running'
    server_state.set('tasks', tasks)

    models = get_model_list(api_key)

    if isinstance(models, list):
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = models
    else:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['result'] = models  # Здесь models содержит сообщение об ошибке

    tasks[task_id]['end_time'] = time.time()
    server_state.set('tasks', tasks)

