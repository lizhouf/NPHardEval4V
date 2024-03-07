import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# model_id = "llava-hf/llava-1.5-13b-hf"
# processor = AutoProcessor.from_pretrained(model_id)

# def llava_run(prompt, imgPATH):

#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     ).to(0)

#     # prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
#     # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     prmopt = "USER: <image>\n" + prompt + "\nASSISTANT:"
#     raw_image = Image.open(imgPATH).convert('RGB')
#     inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

#     output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
#     Final_out = processor.decode(output[0][2:], skip_special_tokens=True)
#     print(Final_out)
#     return Final_out


from transformers import pipeline
from PIL import Image
import requests

with open('model.txt', 'r') as f:
        model_content = f.read()
if model_content == 'llava':
    model_id = "llava-hf/llava-1.5-13b-hf"
    pipe = pipeline("image-to-text", model=model_id, device=0)
def llava_run(prompt, imgPATH):
    # url = "../../Data/BSP/Images/bsp_instance_1_0.png"

    image = Image.open(imgPATH).convert('RGB')
    prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 512})
    print(outputs)
    return outputs



