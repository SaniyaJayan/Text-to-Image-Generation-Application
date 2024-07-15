from googletrans import Translator
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
from flask import jsonify,Flask, request, render_template
import os
import time

app = Flask(__name__)

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (500, 500)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12
    
def get_model():
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16,
        revision = "fp16", use_auth_token ='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', guidance_scale=9
    )
    image_gen_model = image_gen_model.to(CFG.device)
    return image_gen_model

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator = CFG.generator,
        guidance_scale = CFG.image_gen_guidance_scale
    ).image[0]

    image = image.resize(CFG.image_gen_size)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    print(prompt)
    img = generate_image(get_model())
    save_path = save_image(img)
    response = {
        'prompt':prompt,
        'image_path':save_path
    }
    return jsonify(response)

def generate_unique_filename():
    timestamp = str(int(time.time()))
    filename = f"image_{timestamp}.jpg"
    return filename

def save_image(image):
    save_folder = 'static'
    os.makedirs(save_folder, exist_ok=True)
    filename = generate_unique_filename()
    image_path = os.path.join(save_folder,filename)
    image.save(image_path)
    print('Image is saved at this path', image_path)
    return image_path

if __name__ == "__main__":
    app.run(debug = True)