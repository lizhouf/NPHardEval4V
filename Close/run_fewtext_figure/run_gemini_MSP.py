import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
# from models import *
from prompts import mspPrompts
from check.check_hard_MSP import *
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



def load_data():
    data_path = DATA_PATH
    with open(data_path + "msp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data


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



def extract_number(filename):
    # This regular expression will match any number in the file name, even if it's not at the end
    parts = re.findall(r'\d+', filename)
    return int(parts[-1]) if parts else 0



def run_opensource_MSP(qs, p=mspPrompts, imgdir=None):
    imgcnt = 0
    all_prompts = []
    assigned_image_numbers = {}
    image_files_unsorted = [file for file in os.listdir(imgdir) if file.endswith('.png')]
    image_files = sorted(image_files_unsorted, key=extract_number)
    print(image_files)
    outputs = []

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        total_participants = q['participants']
        total_timeslots = q['time_slots']
        prompt_text = p['Intro'] + '\n' \
                      + p['Initial_question'].format(total_participants=total_participants,
                                                     total_timeslots=total_timeslots) + '\n' \
                      + p['Output_content'] + '\n' \
                      + p['Output_format'] + '\n' + \
        '\n The items details are as provided as the figure input.\n'
        prompt_text = prompt_text + "Answer:"

        all_prompts.append(prompt_text)

        print(image_files[imgcnt])
        imgprompts = '/Users/jin666/Desktop/Alien/NPHardEval4V-main/Data/MSP/Images/' + image_files[imgcnt]
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
        a = mspCheck(q, extracted_content)
        print(a)

        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]
        imgcnt += 1



    return outputs



DATA_PATH = '../../Data/MSP/'
sppData = load_data()
IMGDATA_PATH = ''

outputs = run_opensource_MSP(sppData, imgdir=IMGDATA_PATH)
RESULT_PATH = ''
print(outputs)
with open(RESULT_PATH, 'w') as json_file:
    json.dump(outputs, json_file, indent=4)

print(f"Outputs saved to {RESULT_PATH}")
