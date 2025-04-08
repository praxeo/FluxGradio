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
    "black-forest-labs/FLUX.1.1-pro"
]

# Models that typically expect an input image
IMAGE_INPUT_MODELS = [
    "black-forest-labs/FLUX.1-canny",
    "black-forest-labs/FLUX.1-depth", 
    "black-forest-labs/FLUX.1-redux"
]

# List of available models with descriptions
MODEL_INFO = {
    "black-forest-labs/FLUX.1.1-pro": {"description": "Premium high-quality generation model"},
    "black-forest-labs/FLUX.1-schnell": {"description": "Fast generation model with good quality"}, 
    "black-forest-labs/FLUX.1-dev": {"description": "Development model, well-balanced"},
    "black-forest-labs/FLUX.1-canny": {"description": "Uses edge detection for precise composition control"},
    "black-forest-labs/FLUX.1-depth": {"description": "Uses depth maps for accurate spatial relationships"},
    "black-forest-labs/FLUX.1-redux": {"description": "Creates variations of input images"}
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
    input_image: Image.Image | None,
    image_url: str,
    disable_safety: bool,
    progress=gr.Progress(track_tqdm=True)
):
    """Generates images using the Together AI API."""
    if not client:
        raise gr.Error("Together AI client not initialized. Check API Key.", title="Authentication Error")
    if not prompt:
        raise gr.Error("Prompt cannot be empty. Please provide a description of the image you want to generate.", title="Missing Input")

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
            raise gr.Error(f"Failed to process image URL: {str(e)}", title="Image Processing Error")
    elif model_name in IMAGE_INPUT_MODELS:
        # Warn if an image-expecting model is used without an image
        print(f"Note: Model {model_name} typically works better with an input image, but none was provided.")
        gr.Warning(f"Model {model_name} typically works better with an input image, but none was provided.")
    
    # Build API arguments
    args = {
        "model": model_name,
        "prompt": prompt,
        "steps": int(steps),
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
            args["image_base64"] = pil_to_base64(processed_image)
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            raise gr.Error(f"Error encoding image: {str(e)}", title="Image Processing Error")

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
        response = client.images.generate(**args)
        progress(0.5, desc="Processing response")

        images = []
        if response and response.data:
            total = len(response.data)
            for i, image_data in enumerate(response.data):
                progress((0.5 + ((i+1) / total * 0.5)), desc=f"Processing image {i+1} of {total}")
                 
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
            raise gr.Error("No images were returned from the API", title="Generation Failed")
            
        print(f"Successfully generated {len(images)} images.")
        # Return both the images and an updated count message
        image_count_message = f"Generated {len(images)} image{'s' if len(images) > 1 else ''}"
        return images, image_count_message

    except Exception as e:
        error_message = str(e)
        print(f"Error during image generation: {error_message}")
        
        # Provide more helpful error messages based on common error patterns
        if "rate limit" in error_message.lower():
            raise gr.Error("Rate limit exceeded. Please try again after a short wait.", title="API Rate Limit")
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            raise gr.Error("Authentication error. Please check your Together API key.", title="Authentication Error")
        elif "timed out" in error_message.lower() or "timeout" in error_message.lower():
            raise gr.Error("The request timed out. The Together API may be experiencing high traffic.", title="Timeout Error")
        else:
            raise gr.Error(f"Failed to generate image: {error_message}", title="Generation Failed")

def suggest_prompt(theme: str):
    """Generates a prompt suggestion using Llama 4 Maverick."""
    if not client:
        raise gr.Error("Together AI client not initialized. Check API Key.", title="Authentication Error")
    if not theme:
        return "" # Return empty if no theme provided

    print(f"Suggesting prompt for theme: {theme}")
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "You are an expert at creating stunning, evocative image generation prompts. Your specialty is crafting prompts that produce vivid, beautiful, and compelling images when fed to AI image generators. Focus on visual elements: subjects, lighting, colors, mood, composition, style, and artistic influences. Be creative, detailed, and articulate. Your prompts should paint a clear picture in the reader's mind."},
                {"role": "user", "content": f"Create a detailed, artistic prompt for an AI image generator based on this theme: {theme}. Include visual details, style references, lighting, and composition guidance."}
            ],
            max_tokens=200, # Increased token limit for more detailed prompts
            temperature=0.9, # Slightly higher temperature for more creative outputs
        )
        suggestion = response.choices[0].message.content.strip()
        print(f"Generated suggestion: {suggestion}")
        return suggestion
    except Exception as e:
        error_message = str(e)
        print(f"Error suggesting prompt: {error_message}")
        
        if "rate limit" in error_message.lower():
            gr.Warning("Rate limit exceeded. Please try again later.")
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            gr.Warning("Authentication error with the LLM service.")
        else:
            gr.Warning(f"Could not suggest prompt: {error_message}")
        
        return "" # Return empty on error

def random_example():
    """Returns a random example prompt."""
    return get_random_example_prompt()

# --- Gradio UI ---
css = """
.gradio-container { 
    font-family: 'IBM Plex Sans', sans-serif; 
    max-width: 1200px;
    margin: auto;
}
.app-header { 
    text-align: center;
    margin-bottom: 10px;
}
.app-title {
    color: #2c7be5;
    margin-bottom: 0px;
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
}
.footer { 
    text-align: center;
    margin-top: 20px;
    font-size: 0.9em;
    color: #6c757d;
}
.image-preview {
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.model-info {
    margin-top: 5px;
    font-size: 0.85em;
    font-style: italic;
    color: #6c757d;
}
footer { display: none !important; }
.prompt-example-btn {
    padding: 2px 8px;
    margin: 2px;
    border-radius: 4px;
    font-size: 0.8em;
}
.guide-content {
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 15px;
}
.guide-section {
    margin-bottom: 15px;
}
.guide-heading {
    font-weight: bold;
    margin-bottom: 5px;
    color: #2c7be5;
}
"""

# Define the Flux Model Guide content
FLUX_GUIDE_CONTENT = """
## Flux Model Guide – Quick Start

### Model Overview
**Flux 1.1 Pro** is a high-quality general-purpose text-to-image model. It generates detailed, aesthetically pleasing images from prompts without needing any reference images. It's ideal for creating everything from product photos and fantasy art to landscapes and portraits based purely on your imagination.

**Flux Redux** is a variation of Flux 1.1 Pro. It also uses text-to-image generation but is tuned differently. Think of it as offering a different visual "taste" or creative style. You may notice it handles composition, lighting, or faces a bit differently. Use Redux when you want variation in the aesthetic or when Flux Pro isn't quite hitting the vibe you want.

**Flux Depth** is an image-to-image model that uses a depth map extracted from a reference image to understand and preserve the 3D structure and spatial layout of the scene. You still provide a prompt, but the model uses the depth information to guide the composition. It's great for scenarios where you want to restyle an image while keeping the layout intact.

**Flux Canny** is another image-to-image model, but instead of depth, it uses Canny edge detection to preserve the outlines and structure of your source image. It works best with clear shapes like sketches, drawings, or strong pose references. Flux Canny is ideal when you care about the silhouette or positioning of objects.

### Prompting Tips
Prompts should be specific but not overly long. For example, "a cozy cabin at night, snow falling, warm lights, cinematic lighting" works well. You can include stylistic cues like "in the style of Studio Ghibli", "oil painting", or "cyberpunk environment". Use lighting terms like "dramatic rim light" or "soft ambient glow" to control mood. Camera terms such as "wide shot", "portrait", or "overhead angle" help shape composition.

Negative prompts are optional but helpful. For example, "no blur, no watermark, no extra limbs" can improve output quality.

### Parameters Explained
**Seed values** control randomness. The same seed with the same prompt will generate the same image. Changing the seed will produce different variations.

**Guidance Scale (CFG)** controls how closely the model follows your prompt. Higher values (6–10) = more literal.

**Steps** is the number of refinement iterations. 20–30 is a good range. More steps = better quality but longer time.

### Usage Summary
- Use **Flux 1.1 Pro or Redux** when you just want to generate from text
- Use **Flux Depth** when you want to preserve the 3D layout of a scene but change the style
- Use **Flux Canny** when you care more about preserving shapes, outlines, or poses from a sketch or image
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
            with gr.Group():
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
                    lines=2
                )
            
            # Prompt Helper section
            with gr.Accordion("Prompt Helper (Llama 4 Maverick)", open=False):
                prompt_theme_input = gr.Textbox(
                    label="Prompt Idea/Theme", 
                    placeholder="e.g., 'cyberpunk cat cafe', 'underwater city', 'desert oasis'"
                )
                suggest_button = gr.Button("Suggest Detailed Prompt", variant="secondary")
                suggested_prompt_output = gr.Textbox(
                    label="Suggested Prompt", 
                    interactive=True,
                    lines=4
                )
            
            # Model Selection section
            with gr.Group():
                model_selector = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=AVAILABLE_MODELS[0], # Default to FLUX Pro
                    label="Select Model"
                )
                model_info = gr.Markdown(
                    MODEL_INFO[AVAILABLE_MODELS[0]]["description"],
                    elem_classes="model-info" 
                )
            
            # Image Input section
            with gr.Group():
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
                            minimum=1, maximum=50, step=1, value=20, 
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
            return MODEL_INFO[model_name]["description"]
        return ""
    
    model_selector.change(
        fn=update_model_info,
        inputs=[model_selector],
        outputs=[model_info]
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
        outputs=[suggested_prompt_output]
    )
    
    # Use suggested prompt
    def use_suggestion(suggestion):
        """Use the suggested prompt."""
        if suggestion:
            return suggestion
        return ""
        
    # Add button to use the suggested prompt
    with gr.Accordion("Prompt Helper (Llama 4 Maverick)", open=False) as prompt_helper:
        use_suggestion_btn = gr.Button("Use This Suggestion", variant="secondary", size="sm")
        
    use_suggestion_btn.click(
        fn=use_suggestion,
        inputs=[suggested_prompt_output],
        outputs=[prompt_input]
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
