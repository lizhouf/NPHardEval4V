import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
# from models import *
from prompts import sppPrompts
from check.check_p_SPP import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
from openai import OpenAI
import re
import ast
api_key =  ""


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
        try:
            r = response.json()
        except:
            r = ''


        return r


def parse_to_dict(xml_string):
    xml_string = str(xml_string)
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


def load_data(DATA_PATH):
    data_path = DATA_PATH
    with open(data_path + "spp_instances.json", 'r') as f:
        all_data = json.load(f)

    return all_data

def runSPP(MODEL,q, p=sppPrompts):
    start_node = q['nodes'][0]
    end_node = q['nodes'][-1]
    edges = q['edges']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                  p['Output_content'].format(start_node=start_node, end_node=end_node) + '\n' + \
                  p['Output_format'] + \
                  "\n The graph's edges and weights are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
        prompt_text += this_line + '\n'
    prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
    prompt_text += 'Answer:\n'
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
        prompt_text += this_line + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output

def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])

def run_opensource_SPP(qs, p=sppPrompts, imgdir=None):
    imgcnt = 0
    all_txtprompts = []
    assigned_image_numbers = {}
    image_files = [file for file in os.listdir(imgdir)]
    outputs = []
    i = 0
    sorted_files = sorted(image_files, key=custom_sort)
    for q in tqdm(qs):
        start_node = q['nodes'][0]
        end_node = q['nodes'][-1]
        edges = q['edges']
        prompt_text = "<grounding> " + p['Intro'] + '\n' + \
                      p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                      p['Output_content'].format(start_node=start_node, end_node=end_node) + '\n' + \
                      p['Output_format'] + \
                      "\n The graph's edges and weights are as follows: \n" +\
        '\n The items details are as provided as the figure input.\n'
        prompt_text = prompt_text + "Answer:"
        # for edge in edges:
        #     this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
        #     prompt_text += this_line + '\n'
        prompt_text += 'Answer:\n'
        available_image_numbers = [
            int(file.split("_")[-1].split(".")[0])
            for file in image_files
            if int(file.split("_")[-2]) == q['complexity_level']
        ]

        imgprompts = os.path.join(imgdir, sorted_files[imgcnt])
        output = run_gpt4V(prompt_text, imgprompts)
        output, reasoning = parse_to_dict(output)
        try:
            extracted_content = re.search(r'\{.*?\}', output).group()
            print('aoligei')
            extracted_content = ast.literal_eval(extracted_content)
            print(extracted_content)
        except:
            extracted_content = ''

        output_dict = {}
        a = spp_check(q, extracted_content)
        print(a)

        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]
        imgcnt += 1

    # return outputs
    return outputs



DATA_PATH = '../../../Data/SPP/'
sppData = load_data(DATA_PATH)
IMGDATA_PATH = ''
outputs = run_opensource_SPP(sppData, imgdir=IMGDATA_PATH)
RESULT_PATH = ''

with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")

