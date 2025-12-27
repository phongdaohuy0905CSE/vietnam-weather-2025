import streamlit as st
import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import io
import soundfile as sf

# -----------------------------
# Load model (cache để không load lại)
# -----------------------------
@st.cache_resource
def load_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-vie")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------
# TTS function
# -----------------------------
def text2speech(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform.squeeze().cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, output, model.config.sampling_rate, format="WAV")
    return buffer.getvalue()

# -----------------------------
# Weather logic
# -----------------------------
def response_weather(day: str, location: str):
    if day == "0":
        if location == "HCM":
            return "Hôm nay trời nắng, nhiệt độ ba mươi độ."
        elif location == "HN":
            return "No information"
    elif day == "1":
        if location == "HCM":
            return "Ngày mai trời nắng râm, nhiệt độ khoảng hai mươi tám độ."
        elif location == "No information":
            return ""
    elif day == "2":
        if location == "HCM":
            return "Ba ngày tới ở thành phố Hồ Chí Minh có nắng gián đoạn, nhiệt độ trung bình ba mươi mốt độ."
        elif location == "No information":
            return ""

    return "Xin lỗi, trung tâm không có dữ liệu cho lựa chọn này."

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌦️ Trung tâm dự báo thời tiết Con Nai Nù")

st.write("Chọn thông tin để nghe dự báo thời tiết bằng giọng nói.")

day = st.selectbox(
    "Chọn thời gian:",
    options=["0", "1", "2"],
    format_func=lambda x: {
        "0": "Hôm nay",
        "1": "Ngày mai",
        "2": "3 ngày tới"
    }[x]
)

location = st.selectbox(
    "Chọn thành phố:",
    options=["HCM", "HN"],
    format_func=lambda x: {
        "HCM": "TP. Hồ Chí Minh",
        "HN": "Hà Nội"
    }[x]
)

if st.button("🔊 Nghe dự báo"):
    text = response_weather(day, location)
    st.success(text)

    audio_bytes = text2speech(text)
    st.audio(audio_bytes, format="audio/wav")
