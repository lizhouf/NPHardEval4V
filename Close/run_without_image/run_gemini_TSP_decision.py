import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tqdm import tqdm
from prompts import tsp_dPrompts
import pandas as pd
import numpy as np
import json
import argparse
import base64
import requests
from check.check_cmp_TSP_D import *
import re
import ast
from openai import OpenAI
api_key =  ""
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
            df = pd.read_csv(data_path + "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1),
                             header=None,
                             index_col=False)
            # transform df to
            all_data.append(df)
    return all_data

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

def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])

def run_opensource_TSP(qs, p=tsp_dPrompts, imgdir=None): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []

    outputs = []

    for q in tqdm(qs):
        threshold = q.iloc[-1, 0]  # therashold is the last row
        adj_matrix = q.iloc[:-1].values  # distance matrix is the rest of the rows
        total_cities = adj_matrix.shape[0]  # exclude the last row
        prompt_text = p['Intro'] + '\n' + \
                      p['Initial_question'].format(total_cities=total_cities, distance_limit=threshold) + '\n' + \
                      p['Output_content'] + '\n' + \
                      p['Output_format'] + '\n' + \
                      'The distances between cities are below: \n'

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if i < j:  # only use the upper triangle
                    this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
                    prompt_text += this_line + '\n'
        #prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."

        imgprompts = ''
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
        a = tsp_decision_check(q, extracted_content, reasoning)
        print(a)

        output_dict['output'] = extracted_content
        output_dict['correctness'] = a
        output_dict['reasoning'] = reasoning

        outputs += [output_dict]


    return outputs



if __name__ == '__main__':
    DATA_PATH = '../../../Data/TSP_Decision/'

    gcpData = load_data(DATA_PATH)

    IMGDATA_PATH = '../../Data/TSP_Decision/Images'
    outputs = run_opensource_TSP(gcpData, imgdir=IMGDATA_PATH)
    RESULT_PATH = ''
    with open(RESULT_PATH, 'w') as json_file:
        json.dump(outputs, json_file, indent=4)

    print(f"Outputs saved to {RESULT_PATH}")



