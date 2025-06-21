import os
import requests
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from groq import Groq
from app.services.chat_services import get_answer_for_session

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# /start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to our Grocery Bot! Ask me about products, prices, or what's in stock!"
    )

# TEXT message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    question = update.message.text

    answer = get_answer_for_session(session_id=user_id, question=question)
    await update.message.reply_text(answer)

# VOICE message handler
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)

    try:
        voice_file = await update.message.voice.get_file()

        # Create temp folder if not exists (fixes Windows no /tmp folder)
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, f"{user_id}_voice.ogg")

        # Download voice file to local_path
        await voice_file.download_to_drive(custom_path=local_path)

        # Transcribe audio file
        with open(local_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(local_path, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                # NO language param means auto-detect mixed languages
                # language=None
            )
        question = transcription.text  # This contains the original transcribed text, mixed languages

        print(f"[DEBUG] Original transcription (auto language detection): {question}")

        # Get answer from your chatbot
        answer = get_answer_for_session(session_id=user_id, question=question)
        await update.message.reply_text(answer)

    except Exception as e:
        import traceback
        traceback.print_exc()
        await update.message.reply_text("⚠️ Sorry, I couldn't process your voice message.")

# Run the Telegram bot, clearing webhook before polling to avoid conflicts
def run_telegram_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Delete webhook before polling (fixes Conflict error)
    async def on_startup(app_instance):
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.delete_webhook(drop_pending_updates=True)
        print("✅ Webhook cleared before polling.")

    app.post_init = on_startup

    print("🤖 Telegram bot running...")
    app.run_polling()

if __name__ == "__main__":
    run_telegram_bot()
