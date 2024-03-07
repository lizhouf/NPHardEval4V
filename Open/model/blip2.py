import requests
import torch
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration, AutoProcessor
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
# 前期调试把后两行给注释了，后期可以再打开-20240207-2001
with open('model.txt', 'r') as f:
        model_content = f.read()
if model_content == 'blip2':
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="cuda:0")

def run_blip2(txtprompts, imgprompts):
    # for n, p in model.named_parameters():
    #     print(f"{n}: {p.device}")
    # print(txtprompts, imgprompts)
    # img_url = '../microsoft/LLaVA/images/llava_logo.png'
    img_url = imgprompts
    raw_image = Image.open(img_url).convert('RGB')
    model.to('cuda:0')
    # question = "how many dogs are in the picture?"
    question = txtprompts

    inputs = processor(raw_image, question, return_tensors="pt")
    inputs = {k:v.cuda() for k,v in inputs.items()}
    # print(inputs)
    # print(inputs.items(), inputs.keys(), inputs.values())
    out = model.generate(**inputs, max_new_tokens=2048)
    output = processor.decode(out[0], skip_special_tokens=True)
    print(output)
    return output

