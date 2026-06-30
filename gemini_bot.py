import os
import logging
import io
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from PIL import Image

# ========================
# CONFIGURATION
# ========================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY environment variables!")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use stable Gemini 1.5 Pro model
model = genai.GenerativeModel('gemini-1.5-pro')

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# BOT HANDLERS
# ========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when /start is issued."""
    welcome_message = (
        "🎨 Welcome to CreatePixelBot!\n\n"
        "Send me a text description, and I'll generate an image for you.\n\n"
        "✨ Tips for best results:\n"
        "• Be specific and descriptive\n"
        "• Mention the style (photorealistic, cartoon, oil painting, etc.)\n"
        "• Include lighting, colors, and composition details\n\n"
        "Example: 'A majestic dragon flying over snow-capped mountains at sunset, photorealistic'"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a help message."""
    help_text = (
        "🖼️ How to use CreatePixelBot:\n\n"
        "Simply send any text message describing the image you want.\n\n"
        "Examples:\n"
        "• 'Photorealistic sunset over calm ocean waters with seagulls flying'\n"
        "• 'Digital art of a cyberpunk city with neon signs and rain'\n"
        "• 'Oil painting style portrait of an elderly wise wizard'\n\n"
        "Commands:\n"
        "/start - Show welcome message\n"
        "/help - Show this help"
    )
    await update.message.reply_text(help_text)


async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an image from the user's prompt using Gemini."""
    prompt = update.message.text
    
    await update.message.reply_text(f"🎨 Generating image for: '{prompt}'\n⏳ Please wait...")

    try:
        # Use Gemini 1.5 Pro with a different approach - request image generation
        response = model.generate_content(
            f"Generate a high-quality image based on this description. "
            f"Description: {prompt}. "
            f"Create a detailed, visually appealing image. "
            f"Respond with the image data.",
            generation_config={
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Check if response contains image data
        if hasattr(response, '_result') and response._result.candidates:
            for candidate in response._result.candidates:
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            # Extract image data
                            image_data = part.inline_data.data
                            image_bytes = io.BytesIO(image_data)
                            
                            # Send image back to user
                            await update.message.reply_photo(
                                photo=image_bytes,
                                caption=f"✅ Generated from: '{prompt}'\n\n🤖 @CreatePixelBot"
                            )
                            return
        
        # If no image data found, try fallback method
        await update.message.reply_text(
            "⚠️ The image couldn't be generated with the current model. "
            "Please try a different prompt or use DALL-E version."
        )
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        await update.message.reply_text(
            f"❌ Sorry, an error occurred: {str(e)[:100]}...\n\n"
            "Please try again with a different prompt."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
    logger.warning(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("⚠️ Something went wrong. Please try again.")


# ========================
# MAIN
# ========================

def main():
    """Start the bot."""
    app = Application.builder().token(TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    
    # Add message handler for text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))
    
    # Add error handler
    app.add_error_handler(error_handler)

    logger.info("🤖 @CreatePixelBot is running on Gemini 1.5 Pro!")
    app.run_polling()


if __name__ == "__main__":
    main()
