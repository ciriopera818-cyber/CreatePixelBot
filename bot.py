import os
import io
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Try importing the image generation libraries
try:
    from diffusers import DiffusionPipeline
    import torch
    from PIL import Image
    DIFFUSION_AVAILABLE = True
    print("✅ Diffusers imported successfully")
except ImportError as e:
    DIFFUSION_AVAILABLE = False
    print(f"❌ Diffusers not available: {e}")

# ========================
# CONFIGURATION
# ========================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variables
pipe = None
generation_in_progress = False

# ========================
# INITIALIZE MODEL (SMALL VERSION)
# ========================

def initialize_model():
    """Initialize the model with a smaller version that works on Railway."""
    global pipe
    
    if not DIFFUSION_AVAILABLE:
        logger.error("Diffusers not available. Cannot load model.")
        return False
    
    try:
        logger.info("🔄 Loading model... This may take 2-3 minutes.")
        
        # Use a MUCH smaller model that works on Railway free tier
        # This model is only 1.4GB instead of 5-6GB
        model_id = "OFA-Sys/small-stable-diffusion-v0"
        
        # Load with CPU only and minimal memory usage
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp32"
        )
        
        # Use CPU
        pipe.to("cpu")
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

# ========================
# GENERATE IMAGE
# ========================

def generate_image(prompt):
    """Generate an image from text prompt using Stable Diffusion."""
    global pipe, generation_in_progress
    
    if not DIFFUSION_AVAILABLE or pipe is None:
        return None, "Image generation is not available. Please try again later."
    
    if generation_in_progress:
        return None, "Another image is being generated. Please wait."
    
    try:
        generation_in_progress = True
        
        logger.info(f"🎨 Generating image for: {prompt}")
        
        # Enhanced prompt for better results
        enhanced_prompt = f"{prompt}, high quality, detailed"
        
        # Generate the image with fewer steps for speed
        with torch.no_grad():
            image = pipe(
                enhanced_prompt,
                negative_prompt="blurry, ugly, low quality, distorted, deformed, worst quality",
                num_inference_steps=15,  # Reduced for speed
                guidance_scale=7.0,
                height=256,  # Smaller size for faster generation
                width=256
            ).images[0]
        
        # Resize to 512x512 for better quality
        image = image.resize((512, 512))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        generation_in_progress = False
        return img_byte_arr, None
        
    except Exception as e:
        generation_in_progress = False
        logger.error(f"❌ Generation error: {e}")
        return None, f"Error: {str(e)[:100]}"

# ========================
# FALLBACK IMAGE GENERATOR
# ========================

def create_fallback_image(prompt):
    """Create a simple image when model fails."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create gradient background
    img = Image.new('RGB', (512, 512), color=(30, 40, 80))
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    for i, color in enumerate(colors):
        x = 50 + i * 150
        draw.ellipse([x, 100, x+80, 180], fill=color, outline=(255,255,255), width=3)
    
    # Add text
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text = f"🎨 CreatePixelBot\n\nPrompt:\n{prompt[:30]}..."
    draw.text((50, 300), text, fill=(255, 255, 255))
    
    # Add status message
    draw.text((50, 420), "Model is loading...", fill=(200, 200, 100))
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

# ========================
# TELEGRAM HANDLERS
# ========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message."""
    welcome = (
        "🎨 **Welcome to CreatePixelBot!**\n\n"
        "I generate images from your text using AI.\n\n"
        "**How to use:**\n"
        "• Send any text description\n"
        "• Wait 30-60 seconds\n"
        "• Get your image!\n\n"
        "**Examples:**\n"
        "• 'A cat wearing a spacesuit'\n"
        "• 'Beautiful sunset over mountains'\n"
        "• 'Cyberpunk city at night'\n\n"
        "⚠️ First generation takes 2-3 minutes to load."
    )
    await update.message.reply_text(welcome)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    help_text = (
        "🖼️ **CreatePixelBot Help**\n\n"
        "**Commands:**\n"
        "/start - Welcome message\n"
        "/help - This help\n"
        "/status - Bot status\n\n"
        "**Tips:**\n"
        "• Be descriptive\n"
        "• Mention art style\n"
        "• Include colors/mood\n\n"
        "**Examples:**\n"
        "• 'Oil painting of a fox'\n"
        "• 'Futuristic city with neon'\n"
        "• 'Watercolor portrait of a woman'"
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check bot status."""
    global pipe, generation_in_progress
    
    status = "🔍 **Bot Status:**\n\n"
    
    if DIFFUSION_AVAILABLE:
        status += "✅ Image generation: Available\n"
    else:
        status += "❌ Image generation: Unavailable\n"
    
    if pipe is not None:
        status += "✅ AI Model: Loaded\n"
    else:
        status += "⚠️ AI Model: Not loaded\n"
    
    if generation_in_progress:
        status += "🔄 Currently: Generating\n"
    else:
        status += "🟢 Status: Ready\n"
    
    status += f"\n🔧 Model: Small Stable Diffusion"
    
    await update.message.reply_text(status)

async def generate_image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages and generate images."""
    prompt = update.message.text
    
    # Send initial message
    status_msg = await update.message.reply_text(
        f"🎨 **Generating...**\n\nPrompt: \"{prompt}\"\n⏳ Please wait 30-60 seconds."
    )
    
    # Try to generate image
    image_bytes, error = generate_image(prompt)
    
    if image_bytes is not None:
        # Send the generated image
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"✅ **Generated!**\n\n📝 {prompt}\n\n🤖 @CreatePixelBot"
        )
        await status_msg.delete()
    else:
        # Use fallback
        fallback_image = create_fallback_image(prompt)
        await update.message.reply_photo(
            photo=fallback_image,
            caption=f"⚠️ **Using placeholder**\n\nModel is loading or unavailable.\n\n📝 {prompt}\n\nTry again in a few minutes!"
        )
        await status_msg.delete()

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors."""
    logger.error(f"Error: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "❌ **Error!**\n\nPlease try again in a few minutes."
        )

# ========================
# MAIN
# ========================

def main():
    """Start the bot."""
    # Try to initialize the model
    model_loaded = initialize_model()
    
    if not model_loaded:
        logger.warning("⚠️ Model not loaded. Bot will use fallback images.")
    
    # Create application
    app = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    
    # Add message handler for text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image_handler))
    
    # Add error handler
    app.add_error_handler(error_handler)
    
    logger.info("🤖 @CreatePixelBot is running!")
    logger.info("🚀 Bot is ready to generate images!")
    
    # Start the bot
    app.run_polling()

if __name__ == "__main__":
    main()
