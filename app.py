import streamlit as st
import streamlit_server_state as server_state
import concurrent.futures
import uuid
import time
import pandas as pd
import re
from streamlit_autorefresh import st_autorefresh

# Ваши импортированные модули и функции
import requests
import json

# Инициализация хранилища задач
if 'tasks' not in server_state:
    server_state.tasks = {}

# Инициализация пула потоков
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Функция для отправки задач в фоновый режим
def submit_task(function, *args, **kwargs):
    task_id = str(uuid.uuid4())
    server_state.tasks[task_id] = {
        'status': 'running',
        'progress': 0.0,
        'result': None,
        'start_time': time.time(),
        'end_time': None
    }
    
    def task_wrapper(task_id, *args, **kwargs):
        try:
            result = function(*args, task_id=task_id, **kwargs)
            server_state.tasks[task_id]['status'] = 'completed'
            server_state.tasks[task_id]['result'] = result
            server_state.tasks[task_id]['end_time'] = time.time()
        except Exception as e:
            server_state.tasks[task_id]['status'] = 'failed'
            server_state.tasks[task_id]['result'] = str(e)
            server_state.tasks[task_id]['end_time'] = time.time()
    
    executor.submit(task_wrapper, task_id, *args, **kwargs)
    st.session_state.current_task_id = task_id
    
    return task_id

# Функция обработки файла
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
    max_workers: int = 5,
    task_id: str = None
):
    total_rows = len(df)
    results = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_indices = list(df.index[start_idx:end_idx])
        chunk_size_actual = len(chunk_indices)
        chunk_results = [None] * chunk_size_actual

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor_inner:
            future_to_i = {}
            for i, row_idx in enumerate(chunk_indices):
                row_text = str(df.loc[row_idx, title_col])
                future = executor_inner.submit(
                    process_single_row,  # Ваша функция обработки строки
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
        progress = len(results) / total_rows
        server_state.tasks[task_id]['progress'] = progress

    df_out = df.copy()
    df_out["rewrite"] = results

    return "Файл успешно обработан."

# Функция обработки перевода файла
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
    max_workers: int = 5,
    task_id: str = None
):
    total_rows = len(df)
    results = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_indices = list(df.index[start_idx:end_idx])
        chunk_size_actual = len(chunk_indices)
        chunk_results = [None] * chunk_size_actual

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor_inner:
            future_to_i = {}
            for i, row_idx in enumerate(chunk_indices):
                row_text = str(df.loc[row_idx, title_col])
                future = executor_inner.submit(
                    process_translation_single_row,  # Ваша функция перевода строки
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
        progress = len(results) / total_rows
        server_state.tasks[task_id]['progress'] = progress

    df_out = df.copy()
    df_out["translated_title"] = results

    return "Перевод завершен."

# Основной интерфейс Streamlit
st.title("🧠 Novita AI Batch Processor")

# Поле ввода API Key, доступное во всех вкладках
st.sidebar.header("🔑 Настройки API")
api_key = st.sidebar.text_input("API Key", value=st.secrets.get("novita_api_key", ""), type="password")

# Боковая панель: Отслеживание задач
st.sidebar.header("📈 Отслеживание задач")

with st.sidebar.expander("🔍 Проверить статус задачи"):
    input_task_id = st.text_input("Введите Task ID", key="input_task_id")
    if st.button("Проверить статус", key="check_status"):
        if not input_task_id:
            st.error("❌ Task ID не может быть пустым!")
        else:
            task = server_state.tasks.get(input_task_id)
            if task:
                status = task['status']
                progress = task['progress']
                result = task['result']
                start_time = task['start_time']
                end_time = task['end_time']
    
                st.write(f"**Task ID:** {input_task_id}")
                st.write(f"**Статус:** {status}")
                st.progress(progress)
                st.write(f"**Начало:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
                if end_time:
                    st.write(f"**Завершение:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
                else:
                    st.write("**Завершение:** В процессе")
    
                if status == 'completed':
                    st.write(f"**Результат:** {result}")
                elif status == 'failed':
                    st.error(f"**Ошибка:** {result}")
            else:
                st.error("❌ Задача с таким ID не найдена.")

# Вкладки приложения
tabs = st.tabs(["🔄 Обработка текста", "🌐 Перевод текста"])

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
                server_state.tasks['model_list_text'] = []
            else:
                model_list_text = get_model_list(api_key)
                server_state.tasks['model_list_text'] = model_list_text

        if 'model_list_text' not in server_state.tasks:
            server_state.tasks['model_list_text'] = []

        if len(server_state.tasks['model_list_text']) > 0:
            selected_model_text = st.selectbox("✅ Выберите модель для обработки текста", server_state.tasks['model_list_text'], key="select_model_text")
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
            num_preview = st.number_input("🔍 Количество строк для предпросмотра", min_value=1, max_value=100, value=10)
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

                # Отправка задачи в фоновый режим
                task_id = submit_task(
                    process_file,
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

                st.success(f"✅ Обработка начата! Ваш Task ID: {task_id}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

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
                server_state.tasks['model_list_translate'] = []
            else:
                model_list_translate = get_model_list(api_key)
                server_state.tasks['model_list_translate'] = model_list_translate

        if 'model_list_translate' not in server_state.tasks:
            server_state.tasks['model_list_translate'] = []

        if len(server_state.tasks['model_list_translate']) > 0:
            selected_model_translate = st.selectbox("✅ Выберите модель для перевода текста", server_state.tasks['model_list_translate'], key="select_model_translate")
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

                # Пользовательский промпт для перевода
                user_prompt_translate = f"Translate the following text from {source_language} to {target_language}:"

                # Отправка задачи перевода в фоновый режим
                task_id_translate = submit_task(
                    process_translation_file,
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

                st.success(f"✅ Перевод начат! Ваш Task ID: {task_id_translate}")
                st.info("Сохраните этот ID, чтобы отслеживать прогресс.")

# Отображение текущего статуса задачи
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None

if st.session_state.current_task_id:
    task = server_state.tasks.get(st.session_state.current_task_id)
    if task:
        status = task['status']
        progress = task['progress']
        result = task['result']
        start_time = task['start_time']
        end_time = task['end_time']

        st.write(f"**Текущая задача:** {st.session_state.current_task_id}")
        st.write(f"**Статус:** {status}")
        st.progress(progress)
        st.write(f"**Начало:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        if end_time:
            st.write(f"**Завершение:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        else:
            st.write("**Завершение:** В процессе")

        if status == 'completed':
            st.write(f"**Результат:** {result}")
            st.session_state.current_task_id = None  # Сброс после завершения
            st.success("✅ Задача завершена!")
        elif status == 'failed':
            st.error(f"**Ошибка:** {result}")
            st.session_state.current_task_id = None  # Сброс после ошибки
        else:
            # Автообновление каждые 5 секунд
            st_autorefresh(interval=5000, key="progress_refresh_current_task")

