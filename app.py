import streamlit as st
import requests
import json
import pandas as pd

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
            # Предполагаем, что data["data"] содержит список моделей
            models = [m["id"] for m in data.get("data", [])]
            return models
        else:
            st.error(f"Не удалось получить список моделей. Код: {resp.status_code}. Текст: {resp.text}")
            return []
    except Exception as e:
        st.error(f"Ошибка при получении списка моделей: {e}")
        return []


def process_file(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    df: pd.DataFrame,
    response_format: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    chunk_size: int = 5000
):
    """Обрабатываем загруженный файл построчно (или чанками)"""

    # Прогресс-бар
    progress_bar = st.progress(0)
    results = []

    # Чтобы избежать слишком больших запросов, разобьем на чанки
    total_rows = len(df)
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        
        for idx, row in chunk.iterrows():
            # Формируем сообщение
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n{str(row[0])}"}  # Предполагаем, что текст в первой колонке
            ]

            # Параметры запроса
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
                # Некоторые движки могут не поддерживать min_p;
                # если Novita AI поддерживает, оставим, иначе уберем.
                "min_p": min_p
            }

            try:
                resp = requests.post(CHAT_COMPLETIONS_ENDPOINT, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }, data=json.dumps(payload))

                if resp.status_code == 200:
                    data = resp.json()
                    # Извлекаем ответ из data
                    content = data["choices"][0]["message"].get("content", "")
                    results.append(content)
                else:
                    results.append(f"Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                results.append(f"Exception: {str(e)}")

        # Обновляем прогресс
        progress_bar.progress(end_idx / total_rows)

    # Готовим итоги
    df_out = df.copy()
    df_out["response"] = results

    return df_out

#######################################
# 3) ИНТЕРФЕЙС
#######################################

st.title("🧠 Novita AI Batch Processing")

left_col, right_col = st.columns([1,1])

# Левая колонка: список моделей
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
        selected_model = st.selectbox("Выберите модель", ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"])

# Правая колонка: настройки модели
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

st.markdown("---")
st.subheader("Обработка данных")

user_prompt = st.text_area("Пользовательский промпт")

uploaded_file = st.file_uploader("Прикрепить файл (CSV или TXT, до 100000 строк)", type=["csv", "txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    try:
        if file_extension == "csv":
            # читаем CSV обычным способом
            df = pd.read_csv(uploaded_file)
        else:
            # читаем TXT с помощью engine='python' и delimiter='\n'
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, engine='python')
        st.write("### Предпросмотр файла")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        df = None

    if df is not None:
        if st.button("Запустить"):
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
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                )

                st.success("Обработка завершена!")

                # Вывод результата
                if response_format == "text":
                    st.text_area("Результат", value="\n".join(df_out["response"].astype(str)), height=300)
                else:
                    csv_out = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Скачать результат (CSV)", data=csv_out, file_name="result.csv", mime="text/csv")

                st.write("### Логи")
                st.write("Обработка завершена, строк обработано:", len(df_out))
