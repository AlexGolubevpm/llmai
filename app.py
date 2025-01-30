import streamlit as st
import requests
import json
import pandas as pd
import time
import concurrent.futures
import re
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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

st.set_page_config(page_title="Novita AI Batch Processor", layout="wide")

#######################################
# 2) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#######################################

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

    # === Конец цикла ===

    # Создаем копию df с новым столбцом
    df_out = df.copy()
    df_out["rewrite"] = results

    elapsed = time.time() - start_time
    time_placeholder.success(f"Обработка завершена за {elapsed:.1f} секунд.")

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

    # Постобработка: убираем banned words
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

    # === Конец цикла ===

    # Создаем копию df с новым столбцом
    df_out = df.copy()
    df_out["translated_title"] = results

    elapsed = time.time() - start_time
    time_placeholder.success(f"Перевод завершен за {elapsed:.1f} секунд.")

    return df_out

# ======= Новые функции для RewritePro =======

def evaluate_rewrite(api_key: str, model: str, rewrite_text: str):
    """
    Функция для оценки качества рерайта.
    Возвращает оценку от 0 до 10.
    """
    system_prompt = "You are an expert in evaluating text rewrites."
    user_prompt = f"Оцени качество следующего рерайта по шкале от 0 до 10, где 10 - отличный рерайт, а 0 - очень плохой:\n\n{rewrite_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = chat_completion_request(
        api_key=api_key,
        messages=messages,
        model=model,
        max_tokens=10,  # Небольшой ответ
        temperature=0.0,  # Для более детерминированного ответа
        top_p=1.0,
        min_p=0.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0
    )

    # Предполагаем, что ответ - число от 0 до 10
    try:
        score = float(re.findall(r'\d+', raw_response)[0])
        return min(max(score, 0.0), 10.0)  # Ограничиваем от 0 до 10
    except:
        return 0.0  # Если не удалось распознать, ставим 0

def rewrite_specific_row(
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
    """Функция для рерайтинга конкретной строки."""
    return process_single_row(
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

def postprocess_rewrites(
    api_key: str,
    model: str,
    df: pd.DataFrame,
    rewrite_col: str,
    status_col: str,
    threshold: float = 7.0
):
    """Функция для оценки и переписывания строк с низкой оценкой."""
    for idx, row in df.iterrows():
        current_score = row[status_col]
        if current_score < threshold:
            original_text = row[rewrite_col]
            # Рерайтим текст
            new_rewrite = rewrite_specific_row(
                api_key=api_key,
                model=model,
                system_prompt="Act like you are a helpful assistant.",
                user_prompt="Rewrite the following title:",
                row_text=original_text,
                max_tokens=512,
                temperature=0.7,
                top_p=1.0,
                min_p=0.0,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0
            )
            df.at[idx, rewrite_col] = new_rewrite
            # Оцениваем новый рерайт
            new_score = evaluate_rewrite(
                api_key=api_key,
                model=model,
                rewrite_text=new_rewrite
            )
            df.at[idx, status_col] = new_score
    return df

def postprocess_by_words(
    api_key: str,
    model: str,
    df: pd.DataFrame,
    rewrite_col: str,
    status_col: str,
    words: list
):
    """Функция для переписывания строк, содержащих определённые слова."""
    for idx, row in df.iterrows():
        text = row[rewrite_col]
        if any(word.lower() in text.lower() for word in words):
            # Рерайтим текст
            new_rewrite = rewrite_specific_row(
                api_key=api_key,
                model=model,
                system_prompt="Act like you are a helpful assistant.",
                user_prompt="Rewrite the following title:",
                row_text=text,
                max_tokens=512,
                temperature=0.7,
                top_p=1.0,
                min_p=0.0,
                top_k=40,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0
            )
            df.at[idx, rewrite_col] = new_rewrite
            # Оцениваем новый рерайт
            new_score = evaluate_rewrite(
                api_key=api_key,
                model=model,
                rewrite_text=new_rewrite
            )
            df.at[idx, status_col] = new_score
    return df

#######################################
# 3) ИНТЕРФЕЙС
#######################################

st.title("🧠 Novita AI Batch Processor")

# Создаем вкладки для разделения функционала
tabs = st.tabs(["Обработка текста", "Перевод текста", "RewritePro"])

########################################
# Вкладка 1: Обработка текста
########################################
with tabs[0]:
    st.header("🔄 Обработка текста")

    # Две колонки
    left_col, right_col = st.columns([1, 1])

    ########################################
    # Левая колонка: Список моделей
    ########################################
    with left_col:
        st.markdown("#### Модели для обработки текста")
        st.caption("Список моделей загружается из API Novita AI")

        if st.button("Обновить список моделей (Обработка текста)", key="refresh_models_text"):
            if not st.session_state.get("api_key"):
                st.error("Ключ API пуст")
                model_list_text = []
            else:
                model_list_text = get_model_list(st.session_state["api_key"])
                st.session_state["model_list_text"] = model_list_text

        if "model_list_text" not in st.session_state:
            st.session_state["model_list_text"] = []

        if len(st.session_state["model_list_text"]) > 0:
            selected_model_text = st.selectbox("Выберите модель для обработки текста", st.session_state["model_list_text"], key="select_model_text")
        else:
            selected_model_text = st.selectbox(
                "Выберите модель для обработки текста",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_text"
            )

    ########################################
    # Правая колонка: Настройки генерации
    ########################################
    with right_col:
        st.markdown("#### Параметры генерации для обработки текста")
        output_format = st.selectbox("Формат вывода", ["csv", "txt"], key="output_format_text")  # CSV или TXT
        system_prompt_text = st.text_area("System Prompt", value="Act like you are a helpful assistant.", key="system_prompt_text")

        max_tokens_text = st.slider("max_tokens", min_value=0, max_value=64000, value=512, step=1, key="max_tokens_text")
        temperature_text = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.01, key="temperature_text")
        top_p_text = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="top_p_text")
        min_p_text = st.slider("min_p", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="min_p_text")
        top_k_text = st.slider("top_k", min_value=0, max_value=100, value=40, step=1, key="top_k_text")
        presence_penalty_text = st.slider("presence_penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="presence_penalty_text")
        frequency_penalty_text = st.slider("frequency_penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="frequency_penalty_text")
        repetition_penalty_text = st.slider("repetition_penalty", min_value=0.0, max_value=2.0, value=1.0, step=0.01, key="repetition_penalty_text")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Поле одиночного промпта (не обязательно)
    ########################################
    st.subheader("📝 Одиночный промпт")
    user_prompt_single_text = st.text_area("Введите ваш промпт для одиночной генерации", key="user_prompt_single_text")

    if st.button("Отправить одиночный промпт (Обработка текста)", key="submit_single_text"):
        if not st.session_state.get("api_key"):
            st.error("API Key не указан!")
        elif not user_prompt_single_text.strip():
            st.error("Промпт не может быть пустым!")
        else:
            from_text = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt_single_text}
            ]
            st.info("Отправляем запрос...")
            raw_response = chat_completion_request(
                api_key=st.session_state["api_key"],
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
            st.success("Результат получен!")
            st.text_area("Ответ от модели", value=final_response, height=200)

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Блок обработки файла
    ########################################
    st.subheader("📂 Обработка данных из файла")

    user_prompt_text = st.text_area("Пользовательский промпт (дополнительно к заголовку)", key="user_prompt_text")

    st.markdown("##### Настройка парсинга TXT/CSV")
    delimiter_input_text = st.text_input("Разделитель (delimiter)", value="|", key="delimiter_input_text")
    column_input_text = st.text_input("Названия колонок (через запятую)", value="id,title", key="column_input_text")

    uploaded_file_text = st.file_uploader("Прикрепить файл (CSV или TXT, до 100000 строк)", type=["csv", "txt"], key="uploaded_file_text")

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

            st.write("### Предпросмотр файла")
            st.dataframe(df_text.head())
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")
            df_text = None

    if df_text is not None:
        cols_text = df_text.columns.tolist()
        title_col_text = st.selectbox("Какая колонка является заголовком?", cols_text, key="title_col_text")

        # Ползунок для выбора кол-ва потоков
        max_workers_text = st.slider("Потоки (max_workers)", min_value=1, max_value=20, value=5, key="max_workers_text")

        if st.button("Запустить обработку файла (Обработка текста)", key="process_file_text"):
            if not st.session_state.get("api_key"):
                st.error("API Key не указан!")
            else:
                row_count = len(df_text)
                if row_count > 100000:
                    st.warning(f"Файл содержит {row_count} строк. Это превышает рекомендованный лимит в 100000.")
                st.info("Начинаем обработку, пожалуйста подождите...")

                df_out_text = process_file(
                    api_key=st.session_state["api_key"],
                    model=selected_model_text,
                    system_prompt=system_prompt_text,
                    user_prompt=user_prompt_text,
                    df=df_text,
                    title_col=title_col_text,
                    response_format="csv",  # уже не используем, но пусть есть
                    max_tokens=max_tokens_text,
                    temperature=temperature_text,
                    top_p=top_p_text,
                    min_p=min_p_text,
                    top_k=top_k_text,
                    presence_penalty=presence_penalty_text,
                    frequency_penalty=frequency_penalty_text,
                    repetition_penalty=repetition_penalty_text,
                    chunk_size=10,  # фиксируем 10 строк в чанке
                    max_workers=max_workers_text
                )

                st.success("Обработка завершена!")

                # Скачивание
                if output_format == "csv":
                    csv_out_text = df_out_text.to_csv(index=False).encode("utf-8")
                    st.download_button("Скачать результат (CSV)", data=csv_out_text, file_name="result.csv", mime="text/csv")
                else:
                    txt_out_text = df_out_text.to_csv(index=False, sep="|", header=False).encode("utf-8")
                    st.download_button("Скачать результат (TXT)", data=txt_out_text, file_name="result.txt", mime="text/plain")

                st.write("### Логи")
                st.write("Обработка завершена, строк обработано:", len(df_out_text))

########################################
# Вкладка 2: Перевод текста
########################################
with tabs[1]:
    st.header("🌐 Перевод текста")

    # Две колонки
    left_col_trans, right_col_trans = st.columns([1, 1])

    ########################################
    # Левая колонка: Список моделей для перевода
    ########################################
    with left_col_trans:
        st.markdown("#### Модели для перевода текста")
        st.caption("Список моделей загружается из API Novita AI")

        if st.button("Обновить список моделей (Перевод текста)", key="refresh_models_translate"):
            if not st.session_state.get("api_key"):
                st.error("Ключ API пуст")
                model_list_translate = []
            else:
                model_list_translate = get_model_list(st.session_state["api_key"])
                st.session_state["model_list_translate"] = model_list_translate

        if "model_list_translate" not in st.session_state:
            st.session_state["model_list_translate"] = []

        if len(st.session_state["model_list_translate"]) > 0:
            selected_model_translate = st.selectbox("Выберите модель для перевода текста", st.session_state["model_list_translate"], key="select_model_translate")
        else:
            selected_model_translate = st.selectbox(
                "Выберите модель для перевода текста",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_translate"
            )

    ########################################
    # Правая колонка: Настройки генерации для перевода
    ########################################
    with right_col_trans:
        st.markdown("#### Параметры генерации для перевода текста")
        translate_output_format = st.selectbox("Формат вывода перевода", ["csv", "txt"], key="translate_output_format")  # CSV или TXT
        system_prompt_translate = st.text_area("System Prompt для перевода", value="You are a professional translator.", key="system_prompt_translate")

        max_tokens_translate = st.slider("max_tokens (перевод)", min_value=0, max_value=64000, value=512, step=1, key="max_tokens_translate")
        temperature_translate = st.slider("temperature (перевод)", min_value=0.0, max_value=2.0, value=0.3, step=0.01, key="temperature_translate")
        top_p_translate = st.slider("top_p (перевод)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="top_p_translate")
        min_p_translate = st.slider("min_p (перевод)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="min_p_translate")
        top_k_translate = st.slider("top_k (перевод)", min_value=0, max_value=100, value=40, step=1, key="top_k_translate")
        presence_penalty_translate = st.slider("presence_penalty (перевод)", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="presence_penalty_translate")
        frequency_penalty_translate = st.slider("frequency_penalty (перевод)", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="frequency_penalty_translate")
        repetition_penalty_translate = st.slider("repetition_penalty (перевод)", min_value=0.0, max_value=2.0, value=1.0, step=0.01, key="repetition_penalty_translate")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Поле выбора языков и настройки перевода
    ########################################
    st.subheader("📝 Настройки перевода")

    languages = ["English", "Chinese", "Japanese", "Hindi"]
    source_language = st.selectbox("Исходный язык", languages, index=0, key="source_language")
    target_language = st.selectbox("Целевой язык", languages, index=1, key="target_language")

    if source_language == target_language:
        st.warning("Исходный и целевой языки должны отличаться!")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Блок обработки файла для перевода
    ########################################
    st.subheader("📂 Перевод данных из файла")

    st.markdown("##### Настройка парсинга TXT/CSV для перевода")
    delimiter_input_translate = st.text_input("Разделитель (delimiter) для перевода", value="|", key="delimiter_input_translate")
    column_input_translate = st.text_input("Названия колонок (через запятую) для перевода", value="id,title", key="column_input_translate")

    uploaded_file_translate = st.file_uploader("Прикрепить файл для перевода (CSV или TXT, до 100000 строк)", type=["csv", "txt"], key="uploaded_file_translate")

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

            st.write("### Предпросмотр файла для перевода")
            st.dataframe(df_translate.head())
        except Exception as e:
            st.error(f"Ошибка при чтении файла для перевода: {e}")
            df_translate = None

    if df_translate is not None:
        cols_translate = df_translate.columns.tolist()
        id_col_translate = st.selectbox("Какая колонка является ID?", cols_translate, key="id_col_translate")
        title_col_translate = st.selectbox("Какая колонка является заголовком для перевода?", cols_translate, key="title_col_translate")

        # Ползунок для выбора кол-ва потоков
        max_workers_translate = st.slider("Потоки (max_workers) для перевода", min_value=1, max_value=20, value=5, key="max_workers_translate")

        if st.button("Начать перевод", key="start_translation"):
            if not st.session_state.get("api_key"):
                st.error("API Key не указан!")
            elif source_language == target_language:
                st.error("Исходный и целевой языки должны отличаться!")
            else:
                row_count_translate = len(df_translate)
                if row_count_translate > 100000:
                    st.warning(f"Файл содержит {row_count_translate} строк. Это превышает рекомендованный лимит в 100000.")
                st.info("Начинаем перевод, пожалуйста подождите...")

                # Пользовательский промпт для перевода
                user_prompt_translate = f"Translate the following text from {source_language} to {target_language}:"

                # Обработка перевода
                df_translated = process_translation_file(
                    api_key=st.session_state["api_key"],
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

                st.success("Перевод завершен!")

                # Скачивание
                if translate_output_format == "csv":
                    csv_translated = df_translated.to_csv(index=False).encode("utf-8")
                    st.download_button("Скачать переведенный файл (CSV)", data=csv_translated, file_name="translated_result.csv", mime="text/csv")
                else:
                    txt_translated = df_translated.to_csv(index=False, sep="|", header=False).encode("utf-8")
                    st.download_button("Скачать переведенный файл (TXT)", data=txt_translated, file_name="translated_result.txt", mime="text/plain")

                st.write("### Логи")
                st.write("Перевод завершен, строк переведено:", len(df_translated))

########################################
# Вкладка 3: RewritePro
########################################
with tabs[2]:
    st.header("🛠 RewritePro")

    ########################################
    # Блок выбора моделей для RewritePro
    ########################################
    st.subheader("🔧 Настройки моделей RewritePro")

    # Две колонки для выбора моделей и настроек
    model_col, helper_col = st.columns([1, 1])

    with model_col:
        st.markdown("#### Модель для рерайтинга")
        if st.button("Обновить список моделей (RewritePro)", key="refresh_models_rewritepro"):
            if not st.session_state.get("api_key"):
                st.error("Ключ API пуст")
                model_list_rewritepro = []
            else:
                model_list_rewritepro = get_model_list(st.session_state["api_key"])
                st.session_state["model_list_rewritepro"] = model_list_rewritepro

        if "model_list_rewritepro" not in st.session_state:
            st.session_state["model_list_rewritepro"] = []

        if len(st.session_state["model_list_rewritepro"]) > 0:
            selected_model_rewritepro = st.selectbox("Выберите модель для рерайтинга", st.session_state["model_list_rewritepro"], key="select_model_rewritepro")
        else:
            selected_model_rewritepro = st.selectbox(
                "Выберите модель для рерайтинга",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_rewritepro"
            )

        # Настройки генерации для рерайтинга
        st.markdown("##### Параметры генерации для рерайтинга")
        system_prompt_rewritepro = st.text_area("System Prompt для рерайтинга", value="Act like you are a helpful assistant.", key="system_prompt_rewritepro")

        max_tokens_rewritepro = st.slider("max_tokens (рерайт)", min_value=0, max_value=64000, value=512, step=1, key="max_tokens_rewritepro")
        temperature_rewritepro = st.slider("temperature (рерайт)", min_value=0.0, max_value=2.0, value=0.7, step=0.01, key="temperature_rewritepro")
        top_p_rewritepro = st.slider("top_p (рерайт)", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="top_p_rewritepro")
        min_p_rewritepro = st.slider("min_p (рерайт)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="min_p_rewritepro")
        top_k_rewritepro = st.slider("top_k (рерайт)", min_value=0, max_value=100, value=40, step=1, key="top_k_rewritepro")
        presence_penalty_rewritepro = st.slider("presence_penalty (рерайт)", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="presence_penalty_rewritepro")
        frequency_penalty_rewritepro = st.slider("frequency_penalty (рерайт)", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="frequency_penalty_rewritepro")
        repetition_penalty_rewritepro = st.slider("repetition_penalty (рерайт)", min_value=0.0, max_value=2.0, value=1.0, step=0.01, key="repetition_penalty_rewritepro")

    with helper_col:
        st.markdown("#### Модель для оценки рерайтинга")
        if st.button("Обновить список моделей для хелпера", key="refresh_models_helper"):
            if not st.session_state.get("api_key"):
                st.error("Ключ API пуст")
                model_list_helper = []
            else:
                model_list_helper = get_model_list(st.session_state["api_key"])
                st.session_state["model_list_helper"] = model_list_helper

        if "model_list_helper" not in st.session_state:
            st.session_state["model_list_helper"] = []

        if len(st.session_state["model_list_helper"]) > 0:
            selected_model_helper = st.selectbox("Выберите модель для оценки рерайтинга", st.session_state["model_list_helper"], key="select_model_helper")
        else:
            selected_model_helper = st.selectbox(
                "Выберите модель для оценки рерайтинга",
                ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"],
                key="select_model_default_helper"
            )

        # Настройки генерации для хелпера
        st.markdown("##### Параметры генерации для хелпера")
        system_prompt_helper = st.text_area("System Prompt для хелпера", value="You are an expert in evaluating text rewrites.", key="system_prompt_helper")

    # Разделительная линия
    st.markdown("---")

    ########################################
    # Блок загрузки файла для RewritePro
    ########################################
    st.subheader("📂 Загрузка файла для рерайтинга")

    st.markdown("##### Настройка парсинга TXT/CSV для рерайтинга")
    delimiter_input_rewrite = st.text_input("Разделитель (delimiter) для рерайтинга", value="|", key="delimiter_input_rewrite")
    column_input_rewrite = st.text_input("Названия колонок (через запятую) для рерайтинга", value="id,title", key="column_input_rewrite")

    uploaded_file_rewrite = st.file_uploader("Прикрепить файл для рерайтинга (CSV или TXT, до 100000 строк)", type=["csv", "txt"], key="uploaded_file_rewrite")

    df_rewrite = None
    if uploaded_file_rewrite is not None:
        file_extension_rewrite = uploaded_file_rewrite.name.split(".")[-1].lower()
        try:
            if file_extension_rewrite == "csv":
                df_rewrite = pd.read_csv(uploaded_file_rewrite)
            else:
                content_rewrite = uploaded_file_rewrite.read().decode("utf-8")
                lines_rewrite = content_rewrite.splitlines()

                columns_rewrite = [c.strip() for c in column_input_rewrite.split(",")]

                parsed_lines_rewrite = []
                for line in lines_rewrite:
                    splitted_rewrite = line.split(delimiter_input_rewrite, maxsplit=len(columns_rewrite) - 1)
                    if len(splitted_rewrite) < len(columns_rewrite):
                        # Заполняем недостающие колонки пустыми строками
                        splitted_rewrite += [""] * (len(columns_rewrite) - len(splitted_rewrite))
                    parsed_lines_rewrite.append(splitted_rewrite)

                df_rewrite = pd.DataFrame(parsed_lines_rewrite, columns=columns_rewrite)

            st.write("### Предпросмотр файла для рерайтинга")
            st.dataframe(df_rewrite.head())
        except Exception as e:
            st.error(f"Ошибка при чтении файла для рерайтинга: {e}")
            df_rewrite = None

    ########################################
    # Блок выбора колонок и настроек
    ########################################
    if df_rewrite is not None:
        cols_rewrite = df_rewrite.columns.tolist()
        id_col_rewrite = st.selectbox("Какая колонка является ID?", cols_rewrite, key="id_col_rewrite")
        title_col_rewrite = st.selectbox("Какая колонка является заголовком для рерайтинга?", cols_rewrite, key="title_col_rewrite")

        # Инициализация дополнительных колонок, если их нет
        if "rewrite" not in df_rewrite.columns:
            df_rewrite["rewrite"] = ""
        if "status" not in df_rewrite.columns:
            df_rewrite["status"] = 0.0

        # Сохранение DataFrame в session_state для дальнейшего обновления
        if "df_rewrite" not in st.session_state:
            st.session_state["df_rewrite"] = df_rewrite.copy()
        else:
            # Обновляем DataFrame, если файл был загружен заново
            if st.session_state.get("uploaded_file_rewrite") != uploaded_file_rewrite:
                st.session_state["df_rewrite"] = df_rewrite.copy()

        df_rewrite = st.session_state["df_rewrite"]

        # Отображение таблицы с кнопками для рерайтинга
        st.write("### Таблица для рерайтинга")

        # Настройка AgGrid
        gb = GridOptionsBuilder.from_dataframe(df_rewrite)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)  # Отображать 50 строк на странице
        gb.configure_side_bar()
        gb.configure_default_column(editable=False, sortable=True, filter=True)
        # Добавление кнопки "Переписать" в каждую строку
        gb.configure_column("rewrite", editable=False)
        gb.configure_column("status", editable=False)
        # Добавляем специальный столбец для кнопки
        gb.configure_column("rewrite_button", headerName="", cellRenderer='''function(params) {
            return '<button style="padding: 5px 10px;">Переписать</button>';
        }''', width=120, suppressMenu=True)

        gridOptions = gb.build()

        # Отображение таблицы с AgGrid
        grid_response = AgGrid(
            df_rewrite,
            gridOptions=gridOptions,
            height=500,
            width='100%',
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True,  # Разрешить выполнение пользовательского JS
            enable_enterprise_modules=False,
            theme='light'
        )

        # Обработка кликов на кнопки "Переписать"
        # В текущей версии st-aggrid нет прямой поддержки обработки кнопок внутри ячеек,
        # поэтому необходимо использовать пользовательский JavaScript или альтернативные методы.
        # Для упрощения, можно использовать стандартные кнопки под таблицей.

        st.write("### Переписать конкретные строки")
        for idx, row in df_rewrite.iterrows():
            cols = st.columns([1, 3, 3, 1, 1, 1])  # Настройка ширины колонок
            with cols[0]:
                st.write(idx + 1)  # Номер строки
            with cols[1]:
                st.write(row[id_col_rewrite])  # ID
            with cols[2]:
                st.write(row[title_col_rewrite])  # Title
            with cols[3]:
                st.write(row["rewrite"])  # Rewrite
            with cols[4]:
                st.write(f"{row['status']}/10")  # Оценка

            # Добавление кнопки "Переписать" рядом с каждой строкой
            with cols[5]:
                button_key = f"rewrite_button_{idx}"
                if st.button("Переписать", key=button_key):
                    rewrite_text = row[title_col_rewrite]
                    st.info(f"Переписываем строку ID: {row[id_col_rewrite]}")
                    new_rewrite = rewrite_specific_row(
                        api_key=st.session_state["api_key"],
                        model=selected_model_rewritepro,
                        system_prompt=system_prompt_rewritepro,
                        user_prompt="Rewrite the following title:",
                        row_text=rewrite_text,  # Убедитесь, что здесь нет запятой
                        max_tokens=max_tokens_rewritepro,
                        temperature=temperature_rewritepro,
                        top_p=top_p_rewritepro,
                        min_p=min_p_rewritepro,
                        top_k=top_k_rewritepro,
                        presence_penalty=presence_penalty_rewritepro,
                        frequency_penalty=frequency_penalty_rewritepro,
                        repetition_penalty=repetition_penalty_rewritepro
                    )
                    df_rewrite.at[idx, "rewrite"] = new_rewrite
                    # Оцениваем рерайт
                    score = evaluate_rewrite(
                        api_key=st.session_state["api_key"],
                        model=selected_model_helper,
                        rewrite_text=new_rewrite
                    )
                    df_rewrite.at[idx, "status"] = score
                    st.success(f"Рерайт завершён. Оценка: {score}/10")
                    # Обновляем session_state
                    st.session_state["df_rewrite"] = df_rewrite.copy()

        # Разделительная линия
        st.markdown("---")

        ########################################
        # Блок автоматической оценки и рерайтинга
        ########################################
        st.subheader("🔍 Автоматическая оценка и рерайтинг низко оцененных строк")

        if st.button("Оценить и переписать низко оцененные строки (ниже 7)", key="auto_rewrite"):
            if not st.session_state.get("api_key"):
                st.error("API Key не указан!")
            else:
                st.info("Начинаем оценку и рерайтинг низко оцененных строк...")
                df_rewrite = postprocess_rewrites(
                    api_key=st.session_state["api_key"],
                    model=selected_model_rewritepro,
                    df=df_rewrite,
                    rewrite_col="rewrite",
                    status_col="status",
                    threshold=7.0
                )
                st.success("Автоматическая оценка и рерайтинг завершены.")
                st.session_state["df_rewrite"] = df_rewrite.copy()

        # Разделительная линия
        st.markdown("---")

        ########################################
        # Блок постобработчика по словам
        ########################################
        st.subheader("🔠 Постобработчик по ключевым словам")

        words_input = st.text_input("Введите слова для переписывания (через запятую)", value="", key="words_input_rewrite")
        if st.button("Переписать строки, содержащие указанные слова", key="rewrite_by_words"):
            if not st.session_state.get("api_key"):
                st.error("API Key не указан!")
            elif not words_input.strip():
                st.error("Пожалуйста, введите хотя бы одно слово.")
            else:
                words = [word.strip() for word in words_input.split(",") if word.strip()]
                if not words:
                    st.error("Пожалуйста, введите хотя бы одно валидное слово.")
                else:
                    st.info("Начинаем переписывание строк, содержащих указанные слова...")
                    df_rewrite = postprocess_by_words(
                        api_key=st.session_state["api_key"],
                        model=selected_model_rewritepro,
                        df=df_rewrite,
                        rewrite_col="rewrite",
                        status_col="status",
                        words=words
                    )
                    st.success("Переписывание завершено.")
                    st.session_state["df_rewrite"] = df_rewrite.copy()

        # Разделительная линия
        st.markdown("---")

        ########################################
        # Скачивание результата
        ########################################
        st.subheader("💾 Скачивание результата")

        download_format_rewrite = st.selectbox("Формат скачивания файла", ["csv", "txt"], key="download_format_rewrite")

        if download_format_rewrite == "csv":
            csv_rewrite = df_rewrite.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать файл (CSV)", data=csv_rewrite, file_name="rewrite_result.csv", mime="text/csv")
        else:
            txt_rewrite = df_rewrite.to_csv(index=False, sep="|", header=True).encode("utf-8")
            st.download_button("Скачать файл (TXT)", data=txt_rewrite, file_name="rewrite_result.txt", mime="text/plain")

        st.write("### Логи")
        st.write("Рерайтинг завершен, строк обработано:", len(df_rewrite))

########################################
# Боковая панель: Ввод API Key
########################################
# Перемещаем ввод API Key в боковую панель для удобства
if "api_key" not in st.session_state:
    st.session_state["api_key"] = DEFAULT_API_KEY

with st.sidebar:
    st.header("🔑 Настройки API")
    st.session_state["api_key"] = st.text_input("API Key", value=st.session_state["api_key"], type="password")
