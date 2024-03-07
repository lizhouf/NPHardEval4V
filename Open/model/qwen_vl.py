from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# tokenizer = AutoTokenizer.from_pretrained("../microsoft/Qwen-VL", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../microsoft/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
with open('model.txt', 'r') as f:
        model_content = f.read()
if model_content == 'qwen_vl':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

def run_qwen_vl(prompt, imgPATH):
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()


    # image = Image.open(imgPATH).convert('RGB')
    image = imgPATH
    query = tokenizer.from_list_format([
        {'image': image},
        {'text': prompt},
    ])
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, max_new_tokens=1024)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
    print(response)
    return response
    # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
    # image = tokenizer.draw_bbox_on_latest_picture(response)
    # if image:
    #   image.save('2.jpg')
    # else:
    #   print("no box")
