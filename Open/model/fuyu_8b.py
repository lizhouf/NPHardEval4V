from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

with open('model.txt', 'r') as f:
        model_content = f.read()
if model_content == 'fuyu_8b':
    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")
def run_fuyu_8b(prompt, imgPATH):
    # load model and processor
    # model_id = "../microsoft/fuyu-8b"
    # processor = FuyuProcessor.from_pretrained(model_id)
    # model = FuyuForCausalLM.from_pretrained(model_id, device_map="auto")
    # model = FuyuForCausalLM.from_pretrained(model_id)
    # prepare inputs for the model
    # text_prompt = "What color is the bus?\n"
    # url = "../microsoft/fuyu-8b/bus.png"
    # image = Image.open(url)
    image = Image.open(imgPATH).convert('RGB')
    # url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
    # image = Image.open(requests.get(url, stream=True).raw)

    # prompts = ["What color is the bus?\n",
    #            "What color is the bus?\n",
    #            "What color is the bus?\n",]
    # image1 = Image.open(url)
    # image2 = Image.open(url)
    # image3 = Image.open(url)
    # inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs.to('cuda:0')
    # autoregressively generate text
    generation_output = model.generate(**inputs, max_new_tokens=512)
    generation_text = processor.batch_decode(generation_output[:, -512:], skip_special_tokens=True)


    # print(generation_output)
    print(generation_text)
    return generation_text
    # assert generation_text == ["The bus is blue.\n"]


