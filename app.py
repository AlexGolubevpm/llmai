git add app.py
git commit -m "Add app.py"
git push origin main
import streamlit as st
import requests
import json
import pandas as pd

# API Configuration
API_BASE_URL = "https://api.novita.ai/v3/openai/chat/completions"

# Streamlit UI
st.set_page_config(page_title="Novita AI Batch Processor", layout="centered")
st.title("🧠 Novita AI Batch Processing")

# Sidebar for API Settings
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 API Key:", type="password")
model_options = ["meta-llama/llama-3.1-8b-instruct", "Nous-Hermes-2-Mixtral-8x7B-DPO"]
selected_model = st.sidebar.selectbox("🤖 Choose a Model:", model_options)
max_tokens = st.sidebar.slider("🔢 Max Tokens:", min_value=50, max_value=4096, value=512, step=50)
stream_response = st.sidebar.checkbox("📡 Stream Response")

# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload a file (TXT or CSV):", type=["txt", "csv"])

# User Prompt Input
st.subheader("💬 Enter Prompt for Processing")
user_prompt = st.text_area("✍️ Enter your prompt:")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["text"])
    
    st.write("### Preview of Uploaded File:")
    st.dataframe(df.head())
    
    if st.button("🚀 Process File"):
        if not api_key:
            st.error("❌ API Key is required.")
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            results = []
            for index, row in df.iterrows():
                messages = [
                    {"role": "system", "content": "Act like you are a helpful assistant."},
                    {"role": "user", "content": f"{user_prompt} {row['text']}"}
                ]
                
                payload = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": stream_response
                }
                
                response = requests.post(API_BASE_URL, headers=headers, data=json.dumps(payload))
                
                if response.status_code == 200:
                    response_data = response.json()
                    results.append(response_data["choices"][0]["message"]["content"])
                else:
                    results.append(f"Error: {response.status_code}")
            
            df["response"] = results
            
            st.success("✅ Processing Completed!")
            st.write("### Processed Results:")
            st.dataframe(df.head())
            
            # Allow Download
            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download Results", data=csv_output, file_name="processed_results.csv", mime="text/csv")
