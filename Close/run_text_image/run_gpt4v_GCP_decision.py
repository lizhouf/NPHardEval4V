import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
from prompts import gcp_dPrompts
from check.check_cmp_GCP_D import *
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
import ast
import re
from openai import OpenAI
api_key =  ""



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



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def load_data(data_path):

    all_data = []
    for file_num in range(10):
        with open(data_path + "decision_data_GCP_{}.txt".format(file_num)) as f:
            data = f.read()
        all_data += data.split('\n\n')[:-1]
    return all_data


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

def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])

def run_opensource_GCP(qs, p=gcp_dPrompts, imgdir=None): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []
    imgcnt = 0
    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []

    for q in tqdm(qs):
        number_of_colors = q.split('\n')[0].split()[-2]  # last character of the first line
        number_of_vertices = q.split('\n')[1].split(' ')[2]  # third word of the second line
        prompt_text = p['Intro'] + '\n' + \
                      p['Initial_question'].format(total_vertices=number_of_vertices,
                                                   number_of_colors=number_of_colors) + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + '\n' + \
                      '\n The graph is below: \n'
        for line in q.split('\n')[2:]:
            vertex_list = line.split(' ')
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1], vertex_list[2])
            prompt_text += this_line + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."

        imgprompts = os.path.join(imgdir, sorted_files[imgcnt])
        output = run_gpt4V(prompt_text, imgprompts)
        extracted_content, reasoning = parse_to_dict(output)
        try:
            # extracted_content = re.search(r'\{.*?\}', output).group()
            print('aoligei')
            extracted_content = ast.literal_eval(extracted_content)
            print(extracted_content)
        except:
            extracted_content = ''

        num_vertices, adjacency_list = read_dimacs_format(q)

        num, _ = gcp_greedy_solution(adjacency_list)
        a = gcp_decision_check(q, extracted_content, num)

        output_dict = {}
        print(a)

        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]







    return outputs





if __name__ == '__main__':
    DATA_PATH = '../../../Data/GCP_Decision/'
    gcpData = load_data(DATA_PATH)
    IMGDATA_PATH = '../../../Data/GCP_Decision/Images'
    outputs = run_opensource_GCP(gcpData, imgdir=IMGDATA_PATH)
    RESULT_PATH = ''
    with open(RESULT_PATH, 'w') as json_file:
        json.dump(outputs, json_file, indent=4)

    print(f"Outputs saved to {RESULT_PATH}")