import streamlit as st
import requests
import json
import pandas as pd
import time

#######################################
# 1) НАСТРОЙКИ ПРИЛОЖЕНИЯ
#######################################

# Базовый URL API Novita
API_BASE_URL = "https://api.novita.ai/v3/openai"
LIST_MODELS_ENDPOINT = f"{API_BASE_URL}/models"
CHAT_COMPLETIONS_ENDPOINT = f"{API_BASE_URL}/chat/completions"

# Ключ по умолчанию (НЕБЕЗОПАСНО в реальном проде)
DEFAULT_API_KEY = "sk_MyidbhnT9jXzw-YDymhijjY8NF15O0Qy7C36etNTAxE"

st.set_page_config(page_title="Novita AI Batch Processor", layout="wide")

#######################################
# 2) ФУНКЦИИ
#######################################

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
    """Функция для синхронного (не-стримингового) chat-комплишена."""
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

    try:
        resp = requests.post(CHAT_COMPLETIONS_ENDPOINT, headers=headers, data=json.dumps(payload))
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"].get("content", "")
        else:
            return f"Ошибка: {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"Исключение: {e}"


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
    chunk_size: int = 10  # фиксируем 10 строк в чанке
):
    """Обрабатываем загруженный файл построчно (или чанками) с отображением примерного оставшегося времени."""

    progress_bar = st.progress(0)
    time_placeholder = st.empty()  # для отображения оставшегося времени

    results = []
    total_rows = len(df)

    start_time = time.time()
    lines_processed = 0

    for start_idx in range(0, total_rows, chunk_size):
        chunk_start_time = time.time()
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        chunk_size_actual = end_idx - start_idx

        for idx, row in chunk.iterrows():
            # Берем колонку, которую пользователь выбрал как 'title'
            row_text = str(row[title_col])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n{row_text}"}
            ]

            content = chat_completion_request(
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
            results.append(content)

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
    # Переименуем столбец с результатом, чтобы было видно, что это переписанный title
    df_out["rewrite"] = results

    elapsed = time.time() - start_time
    time_placeholder.success(f"Обработка завершена за {elapsed:.1f} секунд.")

    return df_out

#######################################
# 4) ИНТЕРФЕЙС
#######################################

st.title("🧠 Novita AI Batch Processing")

# Две колонки
left_col, right_col = st.columns([1,1])

########################################
# Левая колонка: Список моделей
########################################
with left_col:
    st.markdown("#### Модели")
    st.caption("Список моделей загружается из API Novita AI")

    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

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
# Правая колонка: Настройки генерации
########################################
with right_col:
    st.markdown("#### Параметры генерации")
    response_format = st.selectbox("Response Format", ["text", "csv"])
    system_prompt = st.text_area("System Prompt", value="Act like you are a helpful assistant.")

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
# Поле одиночного промпта (не обязательно)
########################################
st.subheader("Одиночный промпт")
user_prompt_single = st.text_area("Введите ваш промпт для одиночной генерации")

if st.button("Отправить одиночный промпт"):
    if not api_key:
        st.error("API Key не указан!")
    else:
        from_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_single}
        ]
        st.info("Отправляем запрос...")
        response = chat_completion_request(
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
        st.success("Результат получен!")
        st.text_area("Ответ от модели", value=response, height=200)

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
            # Если человек хочет сам выбирать delimiter / колонки, можно учесть, иначе просто .read_csv
            # тут в примере пусть csv читаем обычным способом:
            df = pd.read_csv(uploaded_file)
        else:
            # Предполагаем TXT, парсим с учётом delimiter и колонок
            content = uploaded_file.read().decode("utf-8")
            lines = content.splitlines()

            # Получаем список колонок
            columns = [c.strip() for c in column_input.split(",")]

            # Разбиваем каждую строку по delimiter, сохраняя кол-во столбцов
            parsed_lines = []
            for line in lines:
                splitted = line.split(delimiter_input, maxsplit=len(columns) - 1)
                # если не совпадает кол-во столбцов, можно доп. обработку сделать
                parsed_lines.append(splitted)

            df = pd.DataFrame(parsed_lines, columns=columns)

        st.write("### Предпросмотр файла")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        df = None

if df is not None:
    # Выбор, какая колонка считается заголовком (title)
    cols = df.columns.tolist()
    title_col = st.selectbox("Какая колонка является заголовком?", cols)

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
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                df=df,
                title_col=title_col,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                chunk_size=10  # фиксируем 10 строк в чанке
            )

            st.success("Обработка завершена!")

            # df_out["rewrite"] уже есть
            if response_format == "text":
                st.text_area("Результат (колонка 'rewrite')", value="\n".join(df_out["rewrite"].astype(str)), height=300)
            else:
                csv_out = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Скачать результат (CSV)", data=csv_out, file_name="result.csv", mime="text/csv")

            st.write("### Логи")
            st.write("Обработка завершена, строк обработано:", len(df_out))
