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
import ast
import re
import requests
from openai import OpenAI
import PIL.Image
import google.generativeai as genai
import os
import xmltodict
import time


def parse_to_dict(xml_string):
    try:
        xml_string = ast.literal_eval(xml_string)
        print("---------------------------")
        print(xml_string)
        print("---------------------------")

        reasoning = xml_string['root']['reasoning']
        final_answer = xml_string['root']['final_answer']

    except:
        reasoning = ''
        final_answer = " "

    return final_answer, reasoning

def run_gpt4V(prompt, imgPATH):
    retries = 6  # Maximum number of retries
    while retries > 0:
        try:
            genai.configure(api_key='')
            img = PIL.Image.open(imgPATH)
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([prompt, img], stream=False)
            c = str(response.text)
            dict_data = xmltodict.parse(c)
            json_data = json.dumps(dict_data, indent=4)
            time.sleep(3)

            print(json_data)
            return json_data
        except Exception as e:
            print(f"An error occurred: {e}")
            retries -= 1

            if retries == 0:
                print("Maximum retries reached. Returning empty JSON.")
                return json.dumps({})




def load_data(DATA_PATH):
    data_path = DATA_PATH
    with open(data_path + "spp_instances.json", 'r') as f:
        all_data = json.load(f)

    return all_data


def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])

def run_opensource_SPP(qs, p=sppPrompts, imgdir=None):

    outputs = []
    i = 0
    for q in tqdm(qs):
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
        #prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        prompt_text += 'Answer:\n'


        imgprompts = ''
        output = run_gpt4V(prompt_text, imgprompts)
        output, reasoning = parse_to_dict(output)
        print(output)
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

    # return outputs
    return outputs




DATA_PATH = '../../../Data/SPP/'
sppData = load_data(DATA_PATH)
IMGDATA_PATH = ''
outputs = run_opensource_SPP(sppData, imgdir=IMGDATA_PATH)
RESULT_PATH = ''
print(outputs)
with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")
