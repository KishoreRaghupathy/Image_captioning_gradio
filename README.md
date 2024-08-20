# Image Captioning Web App

This repository contains a simple web app for generating captions for images using Gradio and the Happy Face library. The app leverages the `Salesforce/blip-image-captioning-base` model for captioning images.

## Features

- Upload any image to the app, and it will generate a caption based on the content of the image.
- Utilizes Gradio for building a user-friendly web interface.
- Powered by Hugging Face's `transformers` library and the `Salesforce/blip-image-captioning-base` model.

## Requirements

- `gradio`: For creating the web interface.
- `numpy`: To handle image data.
- `Pillow (PIL)`: To manipulate images.
- `transformers`: To load and use the pre-trained BLIP model.

You can install the required libraries using:

```bash
pip install gradio numpy pillow transformers
```

## How It Works

1. The app loads a pre-trained model and processor from `Salesforce/blip-image-captioning-base`.
2. You upload an image through the Gradio interface.
3. The image is processed and a caption is generated based on the content of the image.

## Running the App

To run the app locally, simply execute the Python script:

```bash
python app.py
```

This will launch a local Gradio interface in your browser where you can upload images and get their captions.

## Example

![Demo](https://user-images.githubusercontent.com/123456789/abcd.png)

1. Upload an image.
2. The app will generate a caption based on the image.

## Code Explanation

```python
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    # Process the image
    inputs = processor(raw_image, return_tensors="pt")
    # Generate a caption for the image
    out = model.generate(**inputs, max_length=50)
    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Gradio Interface
iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

# Launch the app
iface.launch()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
