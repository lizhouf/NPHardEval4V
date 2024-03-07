import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models import *
from prompts import gcpPrompts
from check.check_hard_GCP import *

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import re
import argparse
from utils import run_opensource_models

def load_data():
    data_path = DATA_PATH
    all_data = []
    if 'test_1' in data_path:
        n = 21
    elif 'test_2' in data_path:
        n = 31 
    else:
        n = 10
    start = n - 10
    for file_num in range(start, n):
        with open(data_path+"synthesized_data_GCP_{}.txt".format(file_num)) as f:
            data = f.read()
        all_data += data.split('\n\n')[:-1]
    return all_data

def runGCP(MODEL,q, p=gcpPrompts): # q is the data for the HP-hard question, p is the prompt
    # print(q)
    chromatic_number = q.split('\n')[0][-1] # last character of the first line
    number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
    prompt_text = p['Intro'] + '\n' \
        + p['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
        + p['Output_content'] + '\n' \
        + p['Output_format'] + \
        '\n The graph is below: \n'
    for line in q.split('\n')[2:]:
        vertex_list = line.split(' ')
        this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
        prompt_text += this_line + '\n'
    
    # get output
    if 'gpt' in MODEL:
        output = run_gpt(prompt_text,model = MODEL)
    elif 'claude' in MODEL:
        output = run_claude(text_prompt=prompt_text,model = MODEL)
    else:
        raise NotImplementedError
    return output

def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])


def run_opensource_GCP(qs, p=gcpPrompts, imgdir=None): # q is the data for the HP-hard question, p is the prompt
    all_prompts = []
    imgcnt = 0

    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []
    outputs_tmp = []
    for q in tqdm(qs):
        chromatic_number = q.split('\n')[0][-1] # last character of the first line
        number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
        prompt_text = p['Intro'] + '\n' \
            + p['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
            + p['Output_content'] + '\n' \
            + p['Output_format'] + \
            '\n The graph is below: \n'
        for line in q.split('\n')[2:]:
            vertex_list = line.split(' ')
            this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
            prompt_text += this_line + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        
        #===================limiedtext
        # chromatic_number = q.split('\n')[0][-1] # last character of the first line
        # number_of_vertices = q.split('\n')[1].split(' ')[2] # third word of the second line
        # prompt_text = p['Intro'] + '\n' \
        #     + p['Initial_question'].format(max_vertices=number_of_vertices,max_colors=chromatic_number) + '\n' \
        #     + p['Output_content'] + '\n' \
        #     + p['Output_format'] + \
        #     '\n The graph is shown in figure: \n'
        # for line in q.split('\n')[2:]:
        #     vertex_list = line.split(' ')
        #     this_line = "Vertex {} is connected to vertex {}.".format(vertex_list[1],vertex_list[2])
        #     prompt_text += this_line + '\n'
        # all_prompts.append(prompt_text)

        imgprompts = os.path.join(imgdir, sorted_files[imgcnt])
        print("=============check_prompt==================")
        print("check prompt")
        print("prompt_text:", prompt_text)
        print("prompt_img:", imgprompts)
        print("=============check_prompt==================")
        output = run_opensource_models(args, MODEL, prompt_text, imgprompts)
        result = {
            'promts_num':imgcnt,
            'data':output
        }
        outputs_tmp.append(result)
        outputs.append(output)


    # logging.info(output_str)
        # print('image_count:', imgcnt)
        if imgcnt >= 99:
            import pickle
            with open('{}_{}_Results_fulltext_figure.pkl'.format(DATA_PATH.split('/')[-2], MODEL), 'wb') as json_file:
                pickle.dump(outputs_tmp, json_file)
                
            break
        imgcnt += 1


    return outputs


def llava_out_process(res):
    generated_text = res[0]['generated_text']

    # 使用正则表达式提取 "ASSISTANT: <root>" 之后的部分
    assistant_text_match = re.search(r'ASSISTANT: <root>(.*)', generated_text, re.DOTALL)

    # 如果找到了匹配的内容
    if assistant_text_match:
        # 提取 "ASSISTANT: <root>" 之后的部分
        assistant_text = assistant_text_match.group(1).strip()
        # 打印提取的内容
        print(assistant_text)
        return assistant_text
    else:
        return generated_text

def qwen_out_process(res):
    content = res
    end_tag_index = content.rfind("</root>")
    start_tag_index = content.rfind("<root>", 0, end_tag_index)
    if start_tag_index != -1 and end_tag_index != -1:
        return content[start_tag_index + len("<root>"):end_tag_index]
    else:
        return res

def blip2_out_process(res):
    corrected_text = res.replace("root>", "<root>").replace("/root>", "</root>").replace("reasoning>", "<reasoning>").replace("/reasoning>", "</reasoning>").replace("final_answer>", "<final_answer>").replace("/final_answer>", "</final_answer>")
    # corrected_text = res.replace("> /", "></").replace("/root>", "</root>").replace("/reasoning>", "</reasoning>").replace("/final_answer>", "</final_answer>")
    corrected_text = corrected_text.replace("/<", "</")
    return corrected_text


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../Data/GCP/', help='../Data/finetune_data/test_1/GCP/')
    parser.add_argument('--img_dir', type=str, default='../../Data/GCP/Images', help='../image_data/SPP/')

    # Parse the argument
    args = parser.parse_args()

    # Your script's logic here, using args.model as the model name
    MODEL = str(args.model)

    # MODEL = 'gpt-4-1106-preview'
    # # models: gpt-4-1106-preview, gpt-3.5-turbo-1106, claude-2, claude-instant, palm-2

    DATA_PATH = args.data_dir
    # striiii = '{}_{}_Results_benchmarks.json'.format(DATA_PATH.split('/')[-2], MODEL)
    IMGDATA_PATH = args.img_dir

    if args.tuned_model_dir:
        RESULT_PATH = '../Results/finetuned/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'
        else:
            RESULT_PATH += 'original/'
    else:
        RESULT_PATH = '../../Results/'
        if 'test_1' in args.data_dir:
            RESULT_PATH += 'test_1/'
        elif 'test_2' in args.data_dir:
            RESULT_PATH += 'test_2/'


    # load data
    gcpData = load_data()
    # gcpData = gcpData[:10]
    print('number of datapoints: ', len(gcpData))
    gcpResults = []

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_GCP(gcpData, imgdir=IMGDATA_PATH)
    # sys.exit()
    gcpResults = []
    for q, output in zip(gcpData, outputs):
        output_dict = {}
        # if MODEL == 'llava':
        #     print("==========================")
        #     print("output_reasoning_list:", llava_out_process(output))
        #     print("***************************")
        #     output, reasoning = parse_xml_to_dict(llava_out_process(output))
        #     # print("==========================")
        #     # print("output_reasoning_list:", output)
        #     # print("***************************")
        #     output_dict['output'] = output
        #     output_dict['correctness'] = gcpCheck(q, output)
        #     output_dict['reasoning'] = reasoning
        #     gcpResults.append(output_dict)
        #     # print("results:", gcpResults)
        # elif MODEL == 'qwen_vl':
        #     print("+++++++++++++++++++++++++")
        #     print("output_reasoning_list:", output)
        #     print("-------------------------")
        #     output, reasoning = parse_xml_to_dict(qwen_out_process(output))
        #     output_dict['output'] = output
        #     output_dict['correctness'] = gcpCheck(q, output)
        #     output_dict['reasoning'] = reasoning
        #     gcpResults.append(output_dict)
        # elif MODEL == 'blip2':
        #     print("+++++++++++++++++++++++++")
        #     print("output:", output)
        #     print("-------------------------")
        #     print("+++++++++++++++++++++++++")
        #     print("blip2_out_process:", blip2_out_process(output))
        #     print("-------------------------")
        #     output, reasoning = parse_xml_to_dict(blip2_out_process(output))
        #     print("+++++++++++++++++++++++++")
        #     print("final_answer_element:", output)
        #     print("-------------------------")
        #     output_dict['output'] = output
        #     output_dict['correctness'] = gcpCheck(q, output)
        #     output_dict['reasoning'] = reasoning
        #     gcpResults.append(output_dict)
        # else:
        #     output, reasoning = parse_xml_to_dict(output)
        #     output_dict['output'] = output
        #     output_dict['correctness'] = gcpCheck(q, output)
        #     output_dict['reasoning'] = reasoning
        #     gcpResults.append(output_dict)
        output_dict['output'] = output
        correctness = gcpCheck(q,output)
        output_dict['correctness'] = correctness
        gcpResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'gcpResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(gcpResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'gcpResults.json', 'w') as f:
            f.write(json.dumps(gcpResults) + '\n')

    # MAX_TRY = 10 # updated MAX_TRY
    # for q in gcpData:
    #     output_dict = {}
    #     num_try = 0
    #     while num_try < MAX_TRY:
    #         try:
    #             output = runGCP(q)
    #             output_dict['output'] = output
    #             output_dict['correctness'] = gcpCheck(q, output)
    #             break
    #         except Exception as e:
    #             print(f"Attempt {num_try+1} failed: {e}")
    #             num_try += 1
    #     if output_dict:
    #         gcpResults.append(output_dict)
    #     else:
    #         print(f"Failed to run {q}")
    #         gcpResults.append({'output': '', 'correctness': False})
    # # save the results
    # with open(RESULT_PATH+MODEL+'_'+'gcpResults.json', 'a') as f:
    #     f.write(json.dumps(gcpResults) + '\n')
