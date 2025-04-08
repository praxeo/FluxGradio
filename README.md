---
title: Flux Image Generator
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2 # Specify a recent Gradio version, adjust if needed
python_version: 3.9 # Specify Python version
app_file: app.py
pinned: false
license: mit # Or apache-2.0, choose an appropriate license
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
