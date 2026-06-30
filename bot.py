import os
import io
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Try importing the image generation libraries
try:
    from diffusers import StableDiffusionPipeline
    import torch
    from PIL import Image
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    print("Diffusers not available. Using fallback mode.")

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
# INITIALIZE AI MODEL
# ========================

def initialize_model():
    """Initialize the Stable Diffusion model."""
    global pipe
    
    if not DIFFUSION_AVAILABLE:
        return False
    
    try:
        logger.info("Loading Stable Diffusion model... This may take a moment.")
        
        # Use a smaller, faster model that works well
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Use CPU for Railway (no GPU needed)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to CPU
        pipe = pipe.to("cpu")
        
        # Enable memory efficient attention
        pipe.enable_attention_slicing()
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# ========================
# IMAGE GENERATION
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
        
        # Generate image
        logger.info(f"Generating image for: {prompt}")
        
        # Enhanced prompt for better results
        enhanced_prompt = f"{prompt}, high quality, detailed, 4k, photorealistic"
        
        # Generate the image
        with torch.no_grad():
            image = pipe(
                enhanced_prompt,
                negative_prompt="blurry, ugly, low quality, distorted, deformed",
                num_inference_steps=20,  # Fewer steps for faster generation
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        generation_in_progress = False
        return img_byte_arr, None
        
    except Exception as e:
        generation_in_progress = False
        logger.error(f"Image generation error: {e}")
        return None, f"Error generating image: {str(e)[:100]}"

# ========================
# FALLBACK: Simple Placeholder Images
# ========================

def create_placeholder_image(prompt):
    """Create a simple placeholder image when model isn't available."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a colorful gradient background
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Draw some decorative elements
    for i in range(10):
        color = (i * 25 % 255, (i * 40) % 255, (i * 60) % 255)
        draw.rectangle([i*50, i*40, i*50+40, i*40+30], fill=color)
    
    # Add text
    text = f"🎨 CreatePixelBot\n\nPrompt:\n{prompt[:50]}..."
    draw.text((50, 200), text, fill=(255, 255, 255), font=font)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

# ========================
# TELEGRAM BOT HANDLERS
# ========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message."""
    welcome_message = (
        "🎨 **Welcome to CreatePixelBot!**\n\n"
        "I generate images from your text descriptions using AI.\n\n"
        "**How to use:**\n"
        "• Send any text description\n"
        "• Wait a few seconds for the image\n"
        "• Be specific for best results\n\n"
        "**Example prompts:**\n"
        "• 'A cat wearing a spacesuit'\n"
        "• 'Beautiful sunset over mountains'\n"
        "• 'Cyberpunk city at night'\n\n"
        "⚠️ **Note:** First generation may take 2-3 minutes to load the model.\n"
        "Enjoy! 🚀"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    help_text = (
        "🖼️ **CreatePixelBot Help**\n\n"
        "**Commands:**\n"
        "/start - Show welcome message\n"
        "/help - Show this help\n"
        "/status - Check bot status\n\n"
        "**Tips for better images:**\n"
        "• Be descriptive (colors, style, mood)\n"
        "• Mention the art style (realistic, cartoon, painting)\n"
        "• Include lighting and atmosphere details\n\n"
        "**Examples:**\n"
        "• 'Oil painting of a fox in a forest'\n"
        "• 'Futuristic city with neon lights'\n"
        "• 'Watercolor portrait of a woman'"
    )
    await update.message.reply_text(help_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check bot status."""
    global pipe, generation_in_progress
    
    status = "🔍 **Bot Status:**\n\n"
    
    if DIFFUSION_AVAILABLE:
        status += "✅ Image generation: **Available**\n"
    else:
        status += "❌ Image generation: **Unavailable** (missing libraries)\n"
    
    if pipe is not None:
        status += "✅ AI Model: **Loaded**\n"
    else:
        status += "⚠️ AI Model: **Not loaded** (will load on first request)\n"
    
    if generation_in_progress:
        status += "🔄 Currently: **Generating image**\n"
    else:
        status += "🟢 Currently: **Ready**\n"
    
    await update.message.reply_text(status)


async def generate_image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages and generate images."""
    prompt = update.message.text
    
    # Send initial message
    status_msg = await update.message.reply_text(
        f"🎨 **Generating image...**\n\nPrompt: \"{prompt}\"\n⏳ Please wait, this may take 30-60 seconds."
    )
    
    # Try to generate image
    image_bytes, error = generate_image(prompt)
    
    if image_bytes is not None:
        # Send the generated image
        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"✅ **Generated successfully!**\n\n📝 Prompt: {prompt}\n\n🤖 @CreatePixelBot"
        )
        await status_msg.delete()
    else:
        # Use fallback if generation failed
        logger.warning(f"Using fallback image for: {prompt}")
        fallback_image = create_placeholder_image(prompt)
        await update.message.reply_photo(
            photo=fallback_image,
            caption=f"⚠️ **Using placeholder image**\n\nModel is still loading or unavailable.\n\n📝 Prompt: {prompt}\n\nTry again in a few minutes! 🚀"
        )
        await status_msg.delete()


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors."""
    logger.error(f"Error: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "❌ **Something went wrong!**\n\nPlease try again or contact the bot owner."
        )

# ========================
# MAIN
# ========================

def main():
    """Start the bot."""
    # Initialize the model
    initialize_model()
    
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
    logger.info("🔧 Using Stable Diffusion 1.5 for image generation")
    
    # Start the bot
    app.run_polling()


if __name__ == "__main__":
    main()
