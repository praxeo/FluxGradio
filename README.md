---
title: Flux Image Generator
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
python_version: 3.9
app_file: app.py
pinned: false
license: mit
---

# Flux Image Generator (via Together AI)

This is a Gradio application for generating images using various Black Forest Labs FLUX models hosted on the Together AI platform.

## Features

*   Supports multiple FLUX models: Pro, Schnell, Dev, Canny, Depth, Redux.
*   Text-to-Image generation.
*   Image-to-Image generation (using Canny, Depth, Redux models with image upload).
*   Adjustable parameters: Steps, CFG Scale, Seed, Dimensions, Number of Outputs.
*   Optional safety checker disabling (where applicable).
*   Prompt suggestion helper using Llama 4 Maverick for higher-quality creative prompts.
*   Comprehensive Flux Model Guide with usage tips and best practices.
*   Improved gallery display with support for multiple images (up to 4 columns).

## Setup

1.  Clone this repository or create a new Space on Hugging Face.
2.  Upload `app.py`, `requirements.txt`, and this `README.md`.
3.  Go to your Space settings -> Secrets -> Add a new secret named `TOGETHER_API_KEY` and paste your Together AI API key as the value.
4.  The Space should build and launch the Gradio interface.

## Model-Specific Notes

* **FLUX.1-schnell**: This model has a maximum limit of 12 steps. The UI automatically adjusts the steps slider when this model is selected to ensure API compatibility.
* **FLUX.1.1-pro** and **FLUX.1-schnell**: These models do not support disabling the safety filter.
* **FLUX.1-canny**, **FLUX.1-depth** and **FLUX.1-redux**: These models work best with an input image.

## Troubleshooting

If you encounter any errors:

1. Verify your Together API key is correctly set in the Space secrets.
2. Check the model-specific limitations mentioned above.
3. The app automatically adjusts parameters like 'steps' when using specific models with constraints.
