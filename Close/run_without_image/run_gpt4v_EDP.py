import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
# from models import *
from prompts import edpPrompts
from check.check_p_EDP import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
import re
import ast
from openai import OpenAI
api_key =  ""



def parse_to_dict(xml_string):
        xml_string=str(xml_string)
        final_answer_element = ''
        if '<final_answer>' in xml_string and '</final_answer>' in xml_string:
            final_answer_start = xml_string.index('<final_answer>') + len('<final_answer>')
            final_answer_end = xml_string.index('</final_answer>')
            final_answer_element = xml_string[final_answer_start:final_answer_end].rstrip().strip().rstrip()
            # final_answer_element = ast.literal_eval(final_answer_element)

        reasoning_element = ''
        if '<reasoning>' in xml_string and '</reasoning>' in xml_string:
            reasoning_start = xml_string.index('<reasoning>') + len('<reasoning>')
            reasoning_end = xml_string.index('</reasoning>')
            reasoning_element = xml_string[reasoning_start:reasoning_end].rstrip().strip().rstrip()


        return final_answer_element, reasoning_element



def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def run_gpt4V(prompt, imgPATH):
        base64_image = encode_image(imgPATH)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()



def run_opensource_EDP(qs, p=edpPrompts, imgdir=None):
    imgcnt = 0
    all_prompts = []
    assigned_image_numbers = {}
    outputs = []

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        string_a = q['string_a']
        string_b = q['string_b']
        prompt_text = p['Intro'] + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + '\n'
        # "The details are shown in figure:"
        # prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        prompt_text += 'Answer:\n'

        imgprompts = 'white.png'
        output = run_gpt4V(prompt_text, imgprompts)
        print(output)

        output, reasoning = parse_to_dict(output)
        try:
            extracted_content = re.search(r'\{.*?\}', output).group()
            print('aoligei')
            extracted_content = ast.literal_eval(extracted_content)
            print(extracted_content)
        except:
            extracted_content = ''

        a = edp_check(q, extracted_content)
        print(a)

        output_dict = {}
        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning
        outputs += [output_dict]
        imgcnt += 1

    return outputs



DATA_PATH = '../../../Data/EDP/'
sppData = load_data()
IMGDATA_PATH = ''

outputs = run_opensource_EDP(sppData, imgdir=IMGDATA_PATH)
RESULT_PATH = ''
print(outputs)
with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")