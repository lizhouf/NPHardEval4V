import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
from prompts import tspPrompts
import pandas as pd
import numpy as np
from check.check_hard_TSP import *
import json
import argparse
import base64
import requests
import re
import ast
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


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def load_data(data_path):
    all_data = []
    if 'test_1' in data_path:
        n = 21
    elif 'test_2' in data_path:
        n = 31
    else:
        n = 10
    start = n - 10
    for level in range(start, n):
        for file_num in range(10):
            # read np arrary
            df = pd.read_csv(data_path + "synthesized_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1),
                             header=None,
                             index_col=False)
            # transform df to
            all_data.append(df)
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


def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])

def run_opensource_TSP(qs, p=tspPrompts, imgdir=None): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []
    imgcnt = 0
    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []

    for q in tqdm(qs):
        total_cities = q.shape[0]
        prompt_text = p['Intro'] + '\n' \
                      + p['Initial_question'].format(total_cities=total_cities) + '\n' \
                      + p['Output_content'] + '\n' \
                      + p['Output_format'] + '\n' \
                                             '\n The distances between cities are below: \n'
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The path between City {} and City {} is with distance {}.".format(i, j, q.iloc[i, j])
                    prompt_text += this_line + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."

        all_prompts.append(prompt_text)

        imgprompts = os.path.join(imgdir, sorted_files[imgcnt])
        output = run_gpt4V(prompt_text, imgprompts)

        extracted_content, reasoning = parse_to_dict(output)
        print(extracted_content)
        try:
            # extracted_content = re.search(r'\{.*?\}', output).group()
            print('aoligei')
            extracted_content = ast.literal_eval(extracted_content)
            print(extracted_content)
        except:
            extracted_content = ''


        output_dict = {}
        a = tspCheck(q, extracted_content, reasoning)
        print(a)

        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]


    return outputs





if __name__ == '__main__':
    DATA_PATH = '../../../Data/TSP/'

    gcpData = load_data(DATA_PATH)

    IMGDATA_PATH = '../../../Data/TSP/Images'
    outputs = run_opensource_TSP(gcpData, imgdir=IMGDATA_PATH)
    RESULT_PATH = ''
    with open(RESULT_PATH, 'w') as json_file:
        json.dump(outputs, json_file, indent=4)

    print(f"Outputs saved to {RESULT_PATH}")



