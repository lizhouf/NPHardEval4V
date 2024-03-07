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
import re
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
    for file_num in range(10):
        with open(data_path + "decision_data_GCP_{}.txt".format(file_num)) as f:
            data = f.read()
        all_data += data.split('\n\n')[:-1]
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

def run_opensource_GCP(qs, p=gcp_dPrompts, imgdir=None): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []
    imgcnt = 0

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
        #prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."

        imgprompts = 'white.png'
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
    IMGDATA_PATH = '../../Data/GCP_Decision/Images'
    outputs = run_opensource_GCP(gcpData, imgdir=IMGDATA_PATH)
    RESULT_PATH = ''
    with open(RESULT_PATH, 'w') as json_file:
        json.dump(outputs, json_file, indent=4)

    print(f"Outputs saved to {RESULT_PATH}")