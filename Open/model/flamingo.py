from PIL import Image
import requests
import torch
from huggingface_hub import hf_hub_download
import torch

from Open.microsoft.open_flamingo.open_flamingo.src.factory import create_model_and_transforms

# device = if ("device" in model_args and model_args["device"] >= 0) else "cpu"


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=1,
    cache_dir="/home/lixiang/.cache/huggingface/modules/transformers_modules/anas-awadalla/mpt-7b/b772e556c8e8a17d087db6935e7cd019e5eefb0f"  # Defaults to ~/.cache
)

# grab model checkpoint from huggingface hub

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to('cuda:0')
# demo_image_one = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
#     ).raw
# )

def run_flamingo(prompt, imgPATH):

    img = Image.open(imgPATH).convert('RGB')

    vision_x = [image_processor(img).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    token_pmt = prompt
    lang_x = tokenizer(
        # ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        [token_pmt],
        return_tensors="pt",
    )

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    print("Generated text: ", tokenizer.decode(generated_text[0]))
