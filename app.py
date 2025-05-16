import os
import openai
import streamlit as st

openai.api_key = os.getenv("OPENAI_API_KEY")

import librosa
import numpy as np
import openai
import soundfile as sf
import tempfile

st.title("🎵 AI Music-to-Lyrics Generator")
st.markdown("Upload a track and get original, high-quality lyrics that match the vibe.")

uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    y, sr = librosa.load(file_path)
    
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    key = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1).argmax()
    mood = "energetic" if tempo > 120 else "calm"  # crude mood logic

    st.success(f"Detected tempo: {int(tempo)} BPM | Mood: {mood} | Key: {key}")

    theme = st.text_input("What's the theme or emotion you want? (e.g. heartbreak, summer love)")
    language = st.selectbox("Language", ["English", "Hindi", "Punjabi", "Hinglish"])

    if st.button("Generate Lyrics"):
        prompt = f"""
You are a professional songwriter. Write original, emotional, industry-grade lyrics for a song.

Music details:
- Tempo: {int(tempo)} BPM
- Key: {key}
- Mood: {mood}
- Theme: {theme}
- Language: {language}

Structure it as a song with [Verse 1], [Chorus], [Verse 2], etc.
Make it catchy, rhyming, and emotionally resonant.
"""


        with st.spinner("Generating lyrics..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            lyrics = response["choices"][0]["message"]["content"]
            st.text_area("🎤 Generated Lyrics:", lyrics, height=300)
