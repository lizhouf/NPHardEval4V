import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

device_map = infer_auto_device_map(model, max_memory={0:'50GiB'},
                                   no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
model = load_checkpoint_and_dispatch(
    model,
    'THUDM/cogvlm-chat-hf',
    # '/home/lixiang/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c',   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
    device_map=device_map,
)
model = model.eval()

# check device for weights if u want to
for n, p in model.named_parameters():
    print(f"{n}: {p.device}")

def run_cogvlm(prompt, imgPATH):
# chat example
#     query = 'Describe this image'
    query = prompt
    # image = Image.open('../microsoft/LLaVA/images/llava_logo.png').convert('RGB')
    image = Image.open(imgPATH).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        Final_out = tokenizer.decode(outputs[0])
        print(Final_out)
    return Final_out
