import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
# from models import *
from prompts import bspPrompts
from check.check_p_BSP import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
import re
from openai import OpenAI
api_key =  ""
import ast




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
    with open(data_path + "bsp_instances.json", 'r') as f:
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

def extract_number(filename):
    # This regular expression will match any number in the file name, even if it's not at the end
    parts = re.findall(r'\d+', filename)
    return int(parts[-1]) if parts else 0

def run_opensource_BSP(qs, p=bspPrompts, imgdir=None):
    imgcnt = 0
    all_prompts = []
    assigned_image_numbers = {}
    image_files_unsorted = [file for file in os.listdir(imgdir) if file.endswith('.png')]
    image_files = sorted(image_files_unsorted, key=extract_number)
    print(image_files)
    outputs = []
    print(bspPrompts)

    for i, q in enumerate(tqdm(qs)):
        target_value = q['target']
        # TO-DO: fix data not being sorted
        array = sorted(q['array'])
        prompt_text = p['Intro'] + '\n' + \
                      p['Initial_question'].format(target_value=target_value) + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + '\n' + \
                      '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."

        print(image_files[imgcnt])
        imgprompts = '' + image_files[imgcnt]
        output = run_gpt4V(prompt_text, imgprompts)

        output, reasoning = parse_to_dict(output)
        try:
            extracted_content = re.search(r'\{.*?\}', output).group()
            print('aoligei')
            extracted_content = ast.literal_eval(extracted_content)
            print(extracted_content)
        except:
            extracted_content = ''

        a = bsp_check(q, extracted_content,reasoning)
        print(a)

        output_dict = {}
        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]
        imgcnt += 1


    return outputs



DATA_PATH = '../../../Data/BSP/'
sppData = load_data()
IMGDATA_PATH = ''

outputs = run_opensource_BSP(sppData, imgdir=IMGDATA_PATH)


RESULT_PATH = ''

with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")
