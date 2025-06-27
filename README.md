---
title: FluxGradio
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: false
---

# Flux Image Generator (via Together AI)

This Gradio application provides an intuitive interface for generating images using various Black Forest Labs FLUX models hosted on the Together AI platform.

## Features

*   Supports multiple FLUX models: Pro, Schnell, Dev, Canny, Depth, Redux, Kontext, and Kontext-Max
*   Text-to-Image generation with detailed prompt control
*   Image-to-Image generation with specialized models
*   Adjustable parameters: Steps, CFG Scale, Seed, Dimensions, Number of Outputs
*   Optional safety checker disabling (where applicable)
*   Prompt suggestion helper using `deepseek-ai/DeepSeek-R1-0528-tput` for higher-quality creative prompts
*   Comprehensive Flux Model Guide with usage tips and best practices
*   Improved gallery display with support for multiple images (up to 4 columns)

## Model Overview & Use Cases

### Text-to-Image Models

#### FLUX.1.1-Pro
The flagship model offering premium quality generation for detailed, photorealistic images. It handles complex scenes with excellent composition and fine details.

**Best for:**
- Professional illustrations and concept art
- Detailed landscapes and environments
- Photorealistic product visualizations
- Complex scenes with multiple elements

**Example prompt:**
```
A serene Japanese garden at sunset, with a wooden bridge crossing a koi pond. Cherry blossom petals gently float on the water's surface. Golden hour lighting casts long shadows across moss-covered stones. Cinematic composition, f/2.8 aperture, shot on Hasselblad.
```

#### FLUX.1-Schnell
Optimized for speed with a maximum of 12 inference steps. Produces good quality images significantly faster than other models.

**Best for:**
- Rapid prototyping and ideation
- Quick concept testing
- Situations where quantity of variations is prioritized over ultimate quality
- Time-sensitive projects

**Example prompt:**
```
Futuristic cyberpunk street market, neon lights, holographic advertisements, street food vendors, rain-slicked streets
```

#### FLUX.1-Dev
A well-balanced model offering a good compromise between generation quality and speed.

**Best for:**
- Everyday use cases
- Balanced workflow where both quality and speed matter
- General-purpose image generation
- Testing prompts before using Pro for final versions

**Example prompt:**
```
A cozy reading nook in a forest cottage. Warm lighting from vintage lamps illuminates stacks of leather-bound books. A large window overlooks a misty forest. Comfortable armchair with a knitted blanket. Warm color palette.
```

### Image-to-Image Models

#### FLUX.1-Canny
Uses edge detection to preserve the outlines and structure of your source image.

**Best for:**
- Converting sketches to finished artwork
- Maintaining exact poses and compositions
- Preserving structural elements while changing style
- Working from line drawings or wireframes

**Example workflow:**
1. Upload a sketch or line drawing
2. Prompt: "Transform this sketch into a detailed watercolor painting of a fantasy castle with vibrant colors and intricate details"

#### FLUX.1-Depth
Uses depth maps to maintain spatial relationships from the source image.

**Best for:**
- Preserving 3D structure while changing style
- Architectural visualization alternatives
- Interior design transformations
- Landscape redesigns

**Example workflow:**
1. Upload a photo of a room interior
2. Prompt: "Convert this space into a minimalist Scandinavian interior with natural wood elements, neutral color palette, and abundant natural light"

#### FLUX.1-Redux
Creates variations of input images while maintaining core elements.

**Best for:**
- Exploring different artistic interpretations
- Testing different lighting or mood variations
- Style transfers that respect original composition
- Creating multiple variations of a core concept

**Example workflow:**
1. Upload a portrait photograph
2. Prompt: "Reimagine this portrait in the style of a Renaissance oil painting with dramatic Rembrandt lighting, rich dark background, and warm color palette"

#### FLUX.1-Kontext
A fast image-to-image model that excels at understanding and replicating the visual context of a reference image. It's highly effective at capturing style, mood, and composition.

**Best for:**
- Style transfer
- Character consistency
- Generating variations that are visually harmonious with a source image
- Rapid context-aware iterations (max 12 steps)

**Example workflow:**
1. Upload an image with a distinct artistic style (e.g., a Van Gogh painting)
2. Prompt: "A modern cityscape at night, in the style of the reference image"

#### FLUX.1-Kontext-Max
A professional-grade version of Kontext for maximum quality and detail, supporting up to 50 inference steps.

**Best for:**
- High-fidelity style transfer
- Detailed character replication
- Complex context-aware generation tasks where quality is paramount

**Example workflow:**
1. Upload a detailed character portrait
2. Prompt: "The same character, now wearing futuristic armor and standing on a spaceship bridge"

## Technical Implementation Notes

### Image-to-Image Processing
For image-to-image models (FLUX.1-canny, FLUX.1-depth, FLUX.1-redux), we use the `image_url` parameter with a data URI format:

```python
data_url = f"data:image/png;base64,{base64_img}"
args["image_url"] = data_url
```

This ensures that the images are properly passed to the Together AI API in a format it can process.

### Model-Specific Constraints
The application automatically handles model-specific constraints:

- FLUX.1-schnell: Maximum of 12 inference steps
- FLUX.1.1-pro, FLUX.1-schnell, FLUX.1-kontext, and FLUX.1-kontext-max: Cannot disable safety filter
- All models: Dynamic resizing of large input images to prevent "Request Entity Too Large" errors

## Prompt Helper

The application includes a powerful prompt suggestion feature powered by `deepseek-ai/DeepSeek-R1-0528-tput`, a state-of-the-art language model. This helper can transform simple themes into detailed, effective prompts that produce better results with image generation models.

**Example:**
- Input theme: "underwater city"
- Generated prompt: "A magnificent underwater metropolis with bioluminescent buildings made of coral and crystal, connected by transparent tubes. Schools of colorful fish swim between the structures while majestic manta rays glide overhead. The city is illuminated by an ethereal blue-green glow, creating dramatic light rays that pierce through the deep water. Photorealistic rendering with dramatic depth and perspective."

## Setup

1. Clone this repository or create a new Space on Hugging Face
2. Upload `app.py`, `requirements.txt`, and this `README.md`
3. Go to your Space settings -> Secrets -> Add a new secret named `TOGETHER_API_KEY` and paste your Together AI API key as the value
4. The Space should build and launch the Gradio interface

## Troubleshooting

If you encounter any errors:

1. Verify your Together API key is correctly set in the Space secrets
2. Check the model-specific limitations mentioned above
3. For image-to-image models, ensure you've uploaded a compatible image
4. The app automatically adjusts parameters like 'steps' when using specific models with constraints
