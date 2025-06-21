import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Transcribe audio
filename = "app/db/audio/generated_f7297a70-524f-4829-a52b-bfbe1d44ac7f.wav"
with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
    )
    captured_text = transcription.text
    print("Transcribed Text:", captured_text)

# Step 2: Translate using LLM
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a translator. Translate the input text into clear English. "
                "If it's already in English, return it as-is. Do not explain anything. Just return the result."
            ),
        },
        {"role": "user", "content": captured_text},
    ],
    temperature=0.3,
)

# Final result
translated_text = response.choices[0].message.content.strip()
print("Translated English Text:", translated_text)