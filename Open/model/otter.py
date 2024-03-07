import mimetypes
import os
from io import BytesIO
from typing import Union
# import cv2
import requests
import torch
import transformers
from PIL import Image
from transformers import CLIPImageProcessor, infer_auto_device_map
import sys
sys.path.append(r"../microsoft/Otter")
from src.otter_ai import OtterForConditionalGeneration
# from pipeline.benchmarks.models.base_model import BaseModel

def get_pil_image(raw_image_data) -> Image.Image:
    if isinstance(raw_image_data, Image.Image):
        return raw_image_data
    else:
        return Image.open(BytesIO(raw_image_data["bytes"]))

def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def run_otter(question: str, raw_image_data):
    # model_path = '../microsoft/otter3.1.4n_7B'
    precision = {}
    model_path = 'luodian/OTTER-Image-MPT7B'
    precision["torch_dtype"] = torch.bfloat16
    # device_map = infer_auto_device_map(model_path, max_memory={0: "50GiB"})
    model = OtterForConditionalGeneration.from_pretrained(model_path, device_map='sequential', **precision)
    model.text_tokenizer.padding_side = "left"
    tokenizer  = model.text_tokenizer
    model.eval()
    input_data = get_pil_image(raw_image_data)


    if isinstance(input_data, Image.Image):
        if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
            vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(model.parameters()).dtype)
        else:
            vision_x = CLIPImageProcessor.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(
                1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image.")



    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(question),
        ],
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    parsed_output = model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].split("<|endofchunk|>")[
        0].strip()

    return parsed_output

