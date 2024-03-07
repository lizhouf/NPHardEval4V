import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
import PIL.Image
import google.generativeai as genai
import os
import xmltodict




def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def run_gpt4V(prompt, imgPATH):
    retries = 3  # Maximum number of retries
    while retries > 0:
        try:
            genai.configure(api_key='')
            img = PIL.Image.open(imgPATH)
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([prompt, img], stream=False)
            c = str(response.text)
            dict_data = xmltodict.parse(c)
            json_data = json.dumps(dict_data, indent=4)
            time.sleep(1)
            print(json_data)
            return json_data
        except Exception as e:
            print(f"An error occurred: {e}")
            retries -= 1

            if retries == 0:
                print("Maximum retries reached. Returning empty JSON.")
                return json.dumps({})

import ast


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



def load_data():
    data_path = DATA_PATH
    with open(data_path + "bsp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data


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
                      p['Output_format'] + \
                      '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'
        all_prompts.append(prompt_text)

        print(image_files[imgcnt])
        imgprompts = '' + image_files[imgcnt] # imput your image prompts here
        output = run_gpt4V(prompt_text, imgprompts)
        print(output)

        extracted_content, reasoning = parse_to_dict(output)
        print(output)
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


DATA_PATH = '../../Data/BSP/'
sppData = load_data()
IMGDATA_PATH = ''

outputs = run_opensource_BSP(sppData, imgdir=IMGDATA_PATH)
RESULT_PATH = ''
print(outputs)
with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")