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
    mood = "energetic" if tempo > 120 else "calm"

    st.write(f"🎼 Tempo: {int(tempo)} BPM")
    st.write(f"🎹 Key: {key}")
    st.write(f"🧠 Inferred Mood: {mood}")

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional lyricist."},
                {"role": "user", "content": prompt}
            ]
        )
        lyrics = response.choices[0].message.content
        st.success("Lyrics generated!")
        st.text_area("🎶 Your AI-generated lyrics:", value=lyrics, height=300)
