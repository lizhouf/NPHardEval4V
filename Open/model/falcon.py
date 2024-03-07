from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.cuda.amp import autocast
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
for n, p in model.named_parameters():
    print(f"{n}: {p.device}")
max_memory = get_balanced_memory(
    model,
    max_memory=None,
    no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
    dtype='float16',
    low_zero=False,
)

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
    dtype='float16'
)

model = dispatch_model(model, device_map=device_map)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0,
    "top_p": 0.85,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "min_new_tokens": 10,
    "max_new_tokens": 50,
    "eos_token_id": tokenizer.eos_token_id,
}


with autocast():
    print(tokenizer.decode(model.generate(tokenizer.encode("Hello World!", return_tensors="pt").to("cuda:0"), **generation_kwargs)[0]))