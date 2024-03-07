import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
# from models import *
from prompts import edpPrompts
from tqdm import tqdm
from check.check_p_EDP import *
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
import re
import ast
from openai import OpenAI
import PIL.Image
import google.generativeai as genai
import os
import xmltodict
import time


def load_data():
    data_path = DATA_PATH
    with open(data_path + "edp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data



def parse_to_dict(xml_string):
    try:
        xml_dict = ast.literal_eval(xml_string)

        # Now you can access 'reasoning' and 'final_answer' from xml_dict
        reasoning = xml_dict['root']['reasoning']
        final_answer = xml_dict['root']['final_answer']

    except:
         final_answer, reasoning = '',''

    return final_answer, reasoning



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def run_gpt4V(prompt, imgPATH):
    retries = 5  # Maximum number of retries
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

def extract_number(filename):
    # This regular expression will match any number in the file name, even if it's not at the end
    parts = re.findall(r'\d+', filename)
    return int(parts[-1]) if parts else 0

def run_opensource_EDP(qs, p=edpPrompts, imgdir=None):
    imgcnt = 0
    all_prompts = []
    assigned_image_numbers = {}
    image_files_unsorted = [file for file in os.listdir(imgdir) if file.endswith('.png')]
    image_files = sorted(image_files_unsorted, key=extract_number)
    print(image_files)
    outputs = []

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        string_a = q['string_a']
        string_b = q['string_b']
        prompt_text = p['Intro'] + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + '\n'
        # "The details are shown in figure:"
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        prompt_text += 'Answer:\n'

        print(image_files[imgcnt])
        imgprompts = '' + image_files[imgcnt]
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

with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")
