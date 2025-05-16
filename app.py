import os
import streamlit as st
import openai
import librosa
import tempfile

from openai import OpenAI

# Set up OpenAI client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🎤 AI Lyric Writer from Music")
st.write("Upload a music track and get AI-generated lyrics based on the mood, tempo, and key of the song.")

uploaded_file = st.file_uploader("Upload your music file (MP3/WAV)", type=["mp3", "wav"])

theme = st.text_input("What is the theme of the song? (e.g., heartbreak, hope, party)")
language = st.text_input("Preferred language (e.g., English, Hindi, Spanish)", value="English")

if uploaded_file and theme and language:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    y, sr = librosa.load(file_path)

    # Extract musical features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][chroma.argmax()]
    mood = "energetic" if tempo
