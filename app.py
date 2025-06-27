from dotenv import load_dotenv

load_dotenv()
import gradio as gr
import together
import os
import base64
import io
import re
import requests
from PIL import Image
import datetime
import random
from typing import Optional
from bs4 import BeautifulSoup

# --- Configuration ---
# Initialize the Together client
# It will automatically look for the TOGETHER_API_KEY environment variable
try:
    client = together.Together()
except Exception as e:
    print(f"Error initializing Together client: {e}")
    # Potentially raise an error or handle it gracefully in the UI
    client = None

# --- Configuration ---
# Models that do not support disabling the safety checker
MODELS_NO_SAFETY_DISABLE = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1.1-pro",
    "black-forest-labs/FLUX.1-kontext",
    "black-forest-labs/FLUX.1-kontext-max"
]

# Model-specific constraints
MODEL_CONSTRAINTS = {
    "black-forest-labs/FLUX.1-schnell": {
        "max_steps": 12
    },
    "black-forest-labs/FLUX.1-kontext": {
        "max_steps": 12
    },
    "black-forest-labs/FLUX.1-kontext-max": {
        "max_steps": 50
    }
}

# Models that typically expect an input image
IMAGE_INPUT_MODELS = [
    "black-forest-labs/FLUX.1-canny",
    "black-forest-labs/FLUX.1-depth",
    "black-forest-labs/FLUX.1-redux",
    "black-forest-labs/FLUX.1-kontext",
    "black-forest-labs/FLUX.1-kontext-max"
]

# List of available models with descriptions
MODEL_INFO = {
    "black-forest-labs/FLUX.1.1-pro": {
        "description": "Premium high-quality generation model for photorealistic and artistic images",
        "details": """FLUX.1.1-Pro is Black Forest Labs' flagship model, designed for maximum image quality and detail.
        It excels at photorealism, complex scenes, and accurate representations of subjects. Best for portfolio-quality
        images when quality is more important than generation speed. Supports up to 50 inference steps."""
    },
    "black-forest-labs/FLUX.1-schnell": {
        "description": "Fast generation model with good quality - ideal for rapid iterations",
        "details": """FLUX.1-Schnell (German for 'fast') is optimized for speed while maintaining respectable image quality.
        It's perfect for quickly testing concepts or when you need multiple variations in less time. Limited to a maximum of
        12 inference steps, it generates images significantly faster than other models in the FLUX family."""
    },
    "black-forest-labs/FLUX.1-dev": {
        "description": "Development model offering a well-balanced mix of quality and speed",
        "details": """FLUX.1-Dev provides a middle ground between the Pro and Schnell models. It offers a good balance of
        generation quality and speed, making it suitable for everyday use and development work. It handles a wide range of
        prompts reliably and supports safety filter disabling for artistic freedom."""
    },
    "black-forest-labs/FLUX.1-canny": {
        "description": "Uses edge detection for precise composition and structure control",
        "details": """FLUX.1-Canny is an image-to-image model that preserves the structural outlines of your reference image.
        It works by detecting edges in your uploaded image and using them as a skeleton for the new creation. Perfect for
        maintaining exact poses, compositions, or converting sketches and line drawings into fully rendered images."""
    },
    "black-forest-labs/FLUX.1-depth": {
        "description": "Uses depth maps for accurate spatial relationships and 3D consistency",
        "details": """FLUX.1-Depth extracts depth information from your reference image to maintain the spatial layout and
        perspective in the generated image. It's ideal for preserving the 3D structure of scenes while changing their style,
        materials, or theme. Excellent for architectural visualization, interior design transformations, or landscape restyling."""
    },
    "black-forest-labs/FLUX.1-redux": {
        "description": "Creates variations of input images while maintaining core elements",
        "details": """FLUX.1-Redux specializes in creating alternative interpretations of your reference images. It maintains
        the essential elements and composition while allowing for creative reinterpretation based on your prompt. Perfect for
        exploring different artistic styles, lighting conditions, or mood variations of an existing image."""
    },
    "black-forest-labs/FLUX.1-kontext": {
        "description": "A fast image-to-image model that excels at understanding visual context",
        "details": """FLUX.1-Kontext is a specialized image-to-image model designed to understand and replicate the visual
        context of a reference image. It's highly effective at capturing the style, mood, and composition of the input,
        making it ideal for tasks like style transfer, character consistency, or generating variations that are visually
        harmonious with the source. It operates with a maximum of 12 inference steps for rapid results."""
    },
    "black-forest-labs/FLUX.1-kontext-max": {
        "description": "A professional-grade version of Kontext for maximum quality and detail",
        "details": """FLUX.1-Kontext-Max offers the same powerful visual context understanding as the base Kontext model but
        is designed for higher fidelity and more detailed outputs. It supports up to 50 inference steps, allowing for
        more refined and intricate results. Use this model when you need the absolute best quality for style transfer,
        character replication, or other context-aware generation tasks."""
    }
}

# Extract just the model IDs for the dropdown
AVAILABLE_MODELS = list(MODEL_INFO.keys())

# Example prompts
EXAMPLE_PROMPTS = [
    "A serene mountain landscape at sunset with a calm lake reflecting the sky",
    "A futuristic cyberpunk cityscape with neon lights and flying vehicles",
    "An enchanted forest with glowing mushrooms and fairy lights",
    "A cozy coffee shop interior on a rainy day",
    "A majestic space scene showing planets and nebulae"
]

# --- Backend Functions ---

# --- Helper Functions ---
def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    buffered = io.BytesIO()
    # Save image to buffer in PNG format to preserve transparency if any
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def url_to_pil(url: str) -> Image.Image:
    """Fetches an image from a URL and converts it to a PIL Image."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        raise ValueError(f"Failed to download image from URL: {e}")

def is_valid_url(url: str) -> bool:
    """Check if the given string is a valid URL."""
    if not url:
        return False
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http://, https://, ftp://, ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IPv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def get_random_example_prompt() -> str:
    """Returns a random example prompt."""
    return random.choice(EXAMPLE_PROMPTS)

# --- Backend Functions ---
def generate_image(
    prompt: str,
    negative_prompt: str,
    model_name: str,
    steps: int,
    cfg_scale: float,
    seed: int,
    width: int,
    height: int,
    num_outputs: int,
    input_image: Optional[Image.Image],
    image_url: str,
    disable_safety: bool,
    progress=gr.Progress(track_tqdm=True)
):
    """Generates images using the Together AI API."""
    if not client:
        raise gr.Error("Together AI client not initialized. Check API Key.")
    if not prompt:
        raise gr.Error("Prompt cannot be empty. Please provide a description of the image you want to generate.")

    processed_image = None
    # Process image inputs (uploaded or URL)
    if input_image is not None:
        processed_image = input_image
        print(f"Using uploaded image for model {model_name}")
    elif image_url and is_valid_url(image_url):
        try:
            processed_image = url_to_pil(image_url)
            print(f"Successfully downloaded image from URL for model {model_name}")
        except Exception as e:
            raise gr.Error(f"Failed to process image URL: {str(e)}")
    elif model_name in IMAGE_INPUT_MODELS:
        # Warn if an image-expecting model is used without an image
        print(f"Note: Model {model_name} typically works better with an input image, but none was provided.")
        gr.Warning(f"Model {model_name} typically works better with an input image, but none was provided.")
    
    # Build API arguments
    # Apply model-specific constraints
    adjusted_steps = steps
    if model_name in MODEL_CONSTRAINTS and "max_steps" in MODEL_CONSTRAINTS[model_name]:
        max_steps = MODEL_CONSTRAINTS[model_name]["max_steps"]
        if steps > max_steps:
            print(f"Warning: Model {model_name} has a maximum of {max_steps} steps. Adjusting from {steps} to {max_steps}.")
            adjusted_steps = max_steps
            
    args = {
        "model": model_name,
        "prompt": prompt,
        "steps": int(adjusted_steps),
        "n": int(num_outputs),
        "width": int(width),
        "height": int(height),
    }

    if negative_prompt:
        args["negative_prompt"] = negative_prompt
    if cfg_scale > 0:
        args["cfg_scale"] = cfg_scale
    if seed != -1:
        args["seed"] = int(seed)
    
    # Add image data if available
    if processed_image is not None:
        try:
            # Resize the image if it's too large (prevent "Request Entity Too Large" errors)
            if processed_image.width > 1024 or processed_image.height > 1024:
                aspect_ratio = processed_image.width / processed_image.height
                if processed_image.width > processed_image.height:
                    new_width = 1024
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = 1024
                    new_width = int(new_height * aspect_ratio)
                processed_image = processed_image.resize((new_width, new_height))
                print(f"Resized image to {new_width}x{new_height} for API compatibility")
            
            # Convert image to base64
            base64_img = pil_to_base64(processed_image)
            
            # Handle image differently based on model type
            if model_name in IMAGE_INPUT_MODELS:
                # For image-to-image models, use image_url with data URL format
                # This matches the format used in the Together AI documentation
                data_url = f"data:image/png;base64,{base64_img}"
                args["image_url"] = data_url
                print(f"Using image_url with data URL for {model_name}, image size: {processed_image.size}")
            else:
                # For text-to-image models (if they support image input)
                args["image_base64"] = base64_img
                print(f"Using image_base64 for {model_name}")
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            raise gr.Error(f"Error encoding image: {str(e)}")

    # Handle safety checker logic
    if disable_safety and model_name not in MODELS_NO_SAFETY_DISABLE:
        args["disable_safety_checker"] = True
        print("Safety checker disabled for this generation.")
    elif disable_safety and model_name in MODELS_NO_SAFETY_DISABLE:
        gr.Warning(f"Note: Safety checker cannot be disabled for model {model_name}, continuing with safety checks enabled.")

    try:
        # Log generation parameters for debugging
        print(f"--- Generating Image ---")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt}")
        print(f"Arguments: {str({k: v for k, v in args.items() if k != 'image_base64'})}")  # Don't print base64 string
        
        progress(0.1, desc="Sending request to Together AI")
        
        # Workaround for the Together API limitation:
        # It seems the API only returns one image regardless of the 'n' parameter
        # So we'll make multiple individual requests instead
        original_n = args["n"]
        args["n"] = 1  # Set to 1 for individual requests
        
        images = []
        for req_idx in range(original_n):
            progress_val = 0.1 + (req_idx / original_n * 0.4)
            progress(progress_val, desc=f"Generating image {req_idx+1} of {original_n}")
            print(f"Requesting image {req_idx+1} of {original_n}...")
            
            # Use a different seed for each image if the original seed wasn't specified
            if seed == -1 and req_idx > 0:
                args["seed"] = random.randint(1, 1000000)
                
            response = client.images.generate(**args)
            print(f"Response received. Data length: {len(response.data) if hasattr(response, 'data') and response.data else 'No data'}")
            
            if response and response.data:
                # Process the response data for each API call
                for i, image_data in enumerate(response.data):
                    progress_val = 0.5 + (req_idx / original_n * 0.5)
                    progress(progress_val, desc=f"Processing image {req_idx+1} of {original_n}")
                     
                    # First try b64_json for direct data
                    if hasattr(image_data, 'b64_json') and image_data.b64_json:
                        img_bytes = base64.b64decode(image_data.b64_json)
                        img = Image.open(io.BytesIO(img_bytes))
                        images.append(img)
                    
                    # Fall back to URL if b64_json isn't available
                    elif hasattr(image_data, 'url') and image_data.url:
                        try:
                            img = url_to_pil(image_data.url)
                            images.append(img)
                        except Exception as url_error:
                            print(f"Failed to fetch image from URL: {url_error}")
                            # Continue to the next image rather than failing completely
                    else:
                        print(f"Warning: No image data found for result {i}")
            
        if not images:
            raise gr.Error("No images were returned from the API")
            
        print(f"Successfully generated {len(images)} images.")
        # Return both the images and an updated count message
        image_count_message = f"Generated {len(images)} image{'s' if len(images) > 1 else ''}"
        return images, image_count_message

    except Exception as e:
        error_message = str(e)
        print(f"Error during image generation: {error_message}")
        
        # Provide more helpful error messages based on common error patterns
        if "rate limit" in error_message.lower():
            raise gr.Error("Rate limit exceeded. Please try again after a short wait.")
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            raise gr.Error("Authentication error. Please check your Together API key.")
        elif "timed out" in error_message.lower() or "timeout" in error_message.lower():
            raise gr.Error("The request timed out. The Together API may be experiencing high traffic.")
        else:
            # NOTE: Do not add 'title' parameter here - it will cause TypeErrors in Gradio 4.19.2
            raise gr.Error(f"Failed to generate image: {error_message}")

def suggest_prompt(theme: str, progress=gr.Progress(track_tqdm=True)):
    """Generates a prompt suggestion using a powerful LLM."""
    if not client:
        raise gr.Error("Together AI client not initialized. Check API Key.")
    if not theme:
        return "", ""

    progress(0.1, desc="Sending request to LLM for prompt suggestion...")
    print(f"Suggesting prompt for theme: {theme}")

    try:
        # Use a more advanced model for higher quality suggestions
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528-tput",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class expert in crafting artistic and effective prompts for AI image generators. "
                        "Your task is to take a user's simple theme or idea and transform it into a rich, detailed, and "
                        "evocative prompt. The prompt should be a single, continuous block of text, not a list. "
                        "Focus on sensory details, composition, lighting, artistic style, and mood. "
                        "For example, if the user says 'a cat in a library', you might create: "
                        "'A photorealistic image of a fluffy ginger cat sleeping peacefully on a pile of old, leather-bound books. "
                        "Sunlight streams through a dusty library window, illuminating the scene with a warm, golden glow. "
                        "The style should be cozy and academic, with a shallow depth of field focusing on the cat. "
                        "The mood is tranquil and serene.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Based on this theme, create a single, detailed, and artistic prompt for an AI image generator: '{theme}'"
                }
            ],
            max_tokens=400,
            temperature=0.75,
            repetition_penalty=1.05,
        )
        
        progress(0.8, desc="Processing LLM response...")
        
        suggestion = response.choices[0].message.content.strip()
        
        # Clean up the suggestion if it includes conversational text
        # For example, removing "Here is a prompt:" from the beginning
        cleaned_suggestion = re.sub(r'^(Here is a detailed prompt based on your theme:|Here is a prompt:|Sure, here is a prompt for you:)\s*', '', suggestion, flags=re.IGNORECASE)
        
        # Further clean up by removing markdown quotes if present
        cleaned_suggestion = cleaned_suggestion.replace('"', '')

        print(f"Generated suggestion: {cleaned_suggestion}")
        
        # Return the suggestion and a confirmation message
        return cleaned_suggestion, f"Suggestion for '{theme}' created successfully."

    except Exception as e:
        error_message = str(e)
        print(f"Error suggesting prompt: {error_message}")
        
        if "rate limit" in error_message.lower():
            gr.Warning("Rate limit exceeded for prompt suggestion. Please try again later.")
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            gr.Warning("Authentication error with the LLM service for prompt suggestion.")
        else:
            gr.Warning(f"Could not suggest prompt: {error_message}")
        
        return "", "Failed to generate suggestion."

def random_example():
    """Returns a random example prompt."""
    return get_random_example_prompt()

# --- Gradio UI ---
css = """
/* --- Base Container & Font --- */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: auto;
}

/* --- FIX: Remove Italics & Improve Spacing for General Text --- */
/* This targets all paragraphs and list items to ensure they are not italicized */
.gradio-container p, .gradio-container li {
    font-style: normal !important; /* The key to removing unwanted italics */
    line-height: 1.6;              /* Increases space between lines for readability */
    letter-spacing: 0.1px;         /* Adds subtle space between characters */
}

/* --- App Header Styling --- */
.app-header {
    text-align: center;
    margin-bottom: 10px;
}
.app-title {
    color: #2c7be5;
    margin-bottom: 0px;
    font-weight: 600; /* Make title a bit bolder */
}
.app-subtitle {
    margin-top: 0px;
    color: #6c757d;
}
.app-description {
    color: #4a5568;
    max-width: 900px;
    margin: 0 auto 20px auto;
    text-align: center;
    font-size: 1.1em; /* Make the description text slightly larger */
}

/* --- General Component Styling --- */
.gradio-group {
    margin-bottom: 15px;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.image-preview {
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* --- IMPROVEMENT: Add Padding to Textboxes --- */
textarea[data-testid="textbox"] {
    padding: 10px !important;
    border-radius: 8px !important;
}

/* --- Model Info Styling --- */
.model-info, .model-details {
    margin-top: 5px;
    font-size: 0.9em;
    color: #555; /* Darker text for better contrast */
}

/* --- User Guide Styling --- */
.guide-content {
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 15px;
}
.guide-content li {
    margin-bottom: 10px; /* IMPROVEMENT: Adds space between each bullet point */
}
.guide-heading {
    font-weight: bold;
    margin-bottom: 5px;
    color: #2c7be5;
}

/* --- Footer and Utility --- */
.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 0.9em;
    color: #6c757d;
}
footer { display: none !important; } /* Hides the default Gradio footer */
"""

# Define the Flux Model Guide content
FLUX_GUIDE_CONTENT = """
### How to Use This App
1.  **Select a Model**: Choose a model from the dropdown. Each model has unique strengths. Check the model details below the selector for more info.
2.  **Write a Prompt**: Describe the image you want to create. Be descriptive! Use the **Prompt Helper** if you need ideas.
3.  **(Optional) Add an Input Image**: For models like Canny, Depth, or Kontext, upload an image or provide a URL to guide the generation.
4.  **Adjust Settings**: Fine-tune parameters like steps, guidance scale (CFG), and image dimensions in the **Advanced Settings**.
5.  **Generate**: Click the "Generate Image" button and see your creation!

### Prompting Best Practices
-   **Be Specific**: Instead of "a dog", try "A fluffy golden retriever puppy playing in a field of wildflowers, morning light".
-   **Use Adjectives**: Words like "serene", "vibrant", "dramatic", "minimalist" help set the mood.
-   **Mention Style**: Include phrases like "in the style of a watercolor painting", "photorealistic", "cyberpunk concept art", or "Studio Ghibli animation".
-   **Control the Camera**: Use terms like "wide-angle shot", "close-up portrait", "from a low angle" to control composition.
-   **Use Negative Prompts**: Add things you want to avoid, like "blurry, deformed hands, watermark, text".
"""

with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="blue", secondary_hue="sky")) as demo:
    with gr.Column(elem_classes="app-header"):
        gr.Markdown("# Flux Image Generator", elem_classes="app-title")
        gr.Markdown("### Powered by Together AI", elem_classes="app-subtitle")
    
    gr.Markdown(
        "Create stunning images using various Black Forest Labs FLUX models. "
        "Choose from text-to-image or image-to-image generation with fine control over all parameters.",
        elem_classes="app-description"
    )
    
    # Add Flux Model Guide
    with gr.Accordion("📚 Flux Model Guide", open=False):
        gr.Markdown(FLUX_GUIDE_CONTENT, elem_classes="guide-content")

    with gr.Row():
        with gr.Column(scale=2):
            # Prompt Input section
            with gr.Group(elem_classes="gradio-group"):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description here...",
                    lines=3
                )
                
                with gr.Row():
                    example_btn = gr.Button("Get Example Prompt", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="Enter elements to avoid in the image...",
                    lines=2,
                    elem_classes="textbox-padding"
                )
            
            # Prompt Helper section
            with gr.Accordion("🚀 Prompt Helper (DeepSeek R1)", open=False):
                with gr.Column():
                    prompt_theme_input = gr.Textbox(
                        label="Prompt Idea / Theme",
                        placeholder="e.g., 'a cat in a library', 'a futuristic city at night', 'enchanted forest'",
                        lines=2,
                        elem_classes="textbox-padding"
                    )
                    suggest_button = gr.Button("✨ Generate a Better Prompt", variant="secondary")
                    
                    suggested_prompt_output = gr.Textbox(
                        label="Suggested Prompt",
                        interactive=True,
                        lines=4,
                        placeholder="Your improved prompt will appear here...",
                        elem_classes="textbox-padding"
                    )
                    
                    with gr.Row():
                        use_suggestion_btn = gr.Button("⬇️ Use This Suggestion", size="sm")
                        clear_suggestion_btn = gr.Button("🗑️ Clear Suggestion", size="sm")
                
                suggestion_status = gr.Markdown("")
            
            # Model Selection section
            with gr.Group(elem_classes="gradio-group"):
                model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0], # Default to FLUX Pro
                    label="Select Model"
                )
                
                # Add a dynamic steps slider update when model changes
                def update_steps_slider(model_name, current_steps):
                    max_steps = 50
                    if model_name in MODEL_CONSTRAINTS and "max_steps" in MODEL_CONSTRAINTS[model_name]:
                        max_steps = MODEL_CONSTRAINTS[model_name]["max_steps"]
                    
                    # Adjust current value if it exceeds the new maximum
                    new_value = min(int(current_steps), max_steps)
                    
                    return gr.update(maximum=max_steps, value=new_value)
                model_info = gr.Markdown(
                    MODEL_INFO[AVAILABLE_MODELS[0]]["description"],
                    elem_classes="model-info" 
                )
                model_details = gr.Markdown(
                    MODEL_INFO[AVAILABLE_MODELS[0]]["details"]
                )
            
            # Image Input section
            with gr.Group(elem_classes="gradio-group"):
                with gr.Tab("Upload Image"):
                    input_image_upload = gr.Image(
                        type="pil", 
                        label="Input Image (for Img2Img models)",
                        elem_classes="image-preview"
                    )
                with gr.Tab("Image URL"):
                    image_url_input = gr.Textbox(
                        label="Image URL", 
                        placeholder="https://example.com/image.jpg"
                    )

            # Advanced Settings section
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        steps_slider = gr.Slider(
                            minimum=1, maximum=50, step=1, value=12, 
                            label="Steps"
                        )
                        cfg_slider = gr.Slider(
                            minimum=0, maximum=20, step=0.5, value=4.5, 
                            label="Guidance Scale (CFG)"
                        )
                        seed_input = gr.Number(
                            label="Seed", value=-1, precision=0,
                            info="-1 for random seed"
                        )
                    
                    with gr.Column():
                        with gr.Row():
                            width_slider = gr.Slider(
                                minimum=512, maximum=2048, step=64, value=1024, 
                                label="Width"
                            )
                            height_slider = gr.Slider(
                                minimum=512, maximum=2048, step=64, value=1024, 
                                label="Height"
                            )
                        
                        num_outputs_slider = gr.Slider(
                            minimum=1, maximum=8, step=1, value=1, 
                            label="Number of Images"
                        )
                        
                        safety_checkbox = gr.Checkbox(
                            label="Disable Safety Filter (Not applicable for Schnell/Pro models)", 
                            value=True, 
                            info="When enabled, removes the content filter for applicable models"
                        )

            # Generate Button
            generate_button = gr.Button("Generate Image", variant="primary", size="lg")

        with gr.Column(scale=3):
            # Output Gallery - Modified for better multiple image display
            output_gallery = gr.Gallery(
                label="Generated Images", 
                show_label=True, 
                elem_id="gallery", 
                columns=4,  # Increased from 2 to 4 columns to show more images side by side
                height=600,  # Fixed height instead of "auto" to ensure all images are visible
                object_fit="contain"
            )
            
            with gr.Row():
                image_count_text = gr.Markdown("", elem_id="image-count")
                status_text = gr.Markdown("Ready to generate images", elem_id="status")

    # --- Event Listeners ---
    # Model info update
    def update_model_info(model_name):
        """Updates the model information based on selection."""
        if model_name in MODEL_INFO:
            return MODEL_INFO[model_name]["description"], MODEL_INFO[model_name]["details"]
        return "", ""
    
    model_selector.change(
        fn=update_model_info,
        inputs=[model_selector],
        outputs=[model_info, model_details]
    )
    
    # Update steps slider based on model selection
    model_selector.change(
        fn=update_steps_slider,
        inputs=[model_selector, steps_slider],
        outputs=[steps_slider]
    )
    
    # Random example prompt
    example_btn.click(
        fn=random_example,
        inputs=[],
        outputs=[prompt_input]
    )
    
    # Clear prompt
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[prompt_input]
    )
    
    # Generate image
    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_prompt_input,
            model_selector,
            steps_slider,
            cfg_slider,
            seed_input,
            width_slider,
            height_slider,
            num_outputs_slider,
            input_image_upload,
            image_url_input,
            safety_checkbox
        ],
        outputs=[output_gallery, image_count_text]
    )

    # Prompt suggestion
    suggest_button.click(
        fn=suggest_prompt,
        inputs=[prompt_theme_input],
        outputs=[suggested_prompt_output, suggestion_status]
    )

    # Use suggested prompt
    def use_suggestion(suggestion):
        """Copies the suggested prompt to the main prompt input."""
        return suggestion

    use_suggestion_btn.click(
        fn=use_suggestion,
        inputs=[suggested_prompt_output],
        outputs=[prompt_input]
    )

    # Clear suggestion
    clear_suggestion_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[prompt_theme_input, suggested_prompt_output, suggestion_status]
    )
    
    # Footer
    gr.Markdown(
        "Created with Gradio and Together AI • Black Forest Labs FLUX Models",
        elem_classes="footer"
    )


# --- Launch ---
if __name__ == "__main__":
    if client is None:
         print("\nERROR: Could not initialize Together AI client.")
         print("Please ensure the TOGETHER_API_KEY environment variable is set correctly.")
         # Launch a dummy interface explaining the error
         with gr.Blocks(theme=gr.themes.Default(primary_hue="red")) as error_demo:
             gr.Markdown("# Error: Configuration Problem", elem_classes="app-title")
             gr.Markdown(
                "### Could not initialize the Together AI client\n\n"
                "Please ensure the `TOGETHER_API_KEY` environment variable is set correctly in your environment or Hugging Face Space secrets."
             )
         error_demo.launch()
    else:
        print("Launching Gradio Interface...")
        demo.launch() # Share=True option can be added for public links if needed
