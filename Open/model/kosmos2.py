import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
# import Open.prompts as prompts

with open('model.txt', 'r') as f:
        model_content = f.read()
if model_content == 'kosmos2':
  model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
  processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", device='cuda:0')
    

def run_kosmos2(prompt, imgPATH):
    # model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    # processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    # url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"

    image = Image.open(imgPATH).convert('RGB')
    # image.save("new_image.jpg")
    # image = Image.open("new_image.jpg")
    # image = Image.open('../../image_data/SPP/spp_instance_3_0.png')
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k:v.cuda() for k,v in inputs.items()}
    model.cuda()
    generated_ids = model.generate(
      pixel_values=inputs["pixel_values"],
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      image_embeds=None,
      image_embeds_position_mask=inputs["image_embeds_position_mask"],
      use_cache=True,
      max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    _processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    processed_text, entities = processor.post_process_generation(generated_text)
    # print(_processed_text)
    print(processed_text)
    # print(entities)
    return processed_text

# imgPATH = '../../image_data/SPP/spp_instance_3_0.png'
# # prompt = '<grounding> An image of'
# # run_kosmos2(prompt, imgPATH)
# # prompt = "<grounding> Question: What is special about this image? Answer:"
# # print(prompts.sppPrompts['Intro'])
# import json
# def load_data(DATA_PATH):
#     data_path = DATA_PATH
#     with open(data_path + "spp_instances.json", 'r') as f:
#         all_data = json.load(f)
#
#     return all_data
#
# sppData = load_data('../../Data/SPP/')
# q = sppData[0]
# p = prompts.sppPrompts
# start_node = q['nodes'][0]
# end_node = q['nodes'][-1]
# edges = q['edges']
# prompt_text ="<grounding> " + p['Intro'] + '\n' + \
#             p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
#             p['Output_content'].format(start_node=start_node, end_node=end_node) + '\n' + \
#             p['Output_format'] + \
#             "\n The graph's edges and weights are as follows: \n"
# for edge in edges:
#     this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
#     prompt_text += this_line + '\n'
# prompt_text += 'Answer:\n'
#
# # data_prompts = ["<grounding> " + prompts.sppPrompts['Intro'] + "\n" + prompts.sppPrompts['Initial_question'] + \
# #                 prompts.sppPrompts['Output_format'] + \
# #                 "Question:" + prompts.sppPrompts['Output_content'] + "? Answer:"]
# run_kosmos2(prompt_text, imgPATH)

