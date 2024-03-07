import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models import *
from prompts import kspPrompts
from check.check_cmp_KSP import *

import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm
from utils import run_opensource_models

def load_data():
    data_path = DATA_PATH
    with open(data_path + "ksp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runKSP(MODEL,q, p=kspPrompts):
    knapsack_capacity = q['knapsack_capacity']
    items = q['items']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  '\n The items details are as below: \n'
    for item in items:
        this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
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


def run_opensource_KSP(qs, p=kspPrompts, imgdir=None):
    all_prompts = []
    imgcnt = 0

    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []
    outputs_tmp = []
    for q in tqdm(qs):
        knapsack_capacity = q['knapsack_capacity']
        items = q['items']
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + \
                    '\n The items details are as below: \n'
        for item in items:
            this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
            prompt_text += this_line + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        #==================limitedtext======
        # knapsack_capacity = q['knapsack_capacity']
        # items = q['items']
        # prompt_text = p['Intro'] + '\n' + \
        #             p['Initial_question'].format(knapsack_capacity=knapsack_capacity) + '\n' + \
        #             p['Output_content'] + '\n' + \
        #             p['Output_format'] + \
        #             '\n The items details are shown in figure: \n'
        
        
        # for item in items:
        #     this_line = f"Item {item['id']} has weight {item['weight']} and value {item['value']}."
        #     prompt_text += this_line + '\n'
        prompt_text += 'Answer:\n'
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
    generated_text = res['generated_text']

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
        print("未找到 ASSISTANT: <root> 之后的内容")

def qwen_out_process(res):
    text_content = res
    start_index = text_content.rfind("<root>")
    end_index = text_content.rfind("</root>")+len("</root>")
    desired_text = text_content[start_index:end_index]
    return desired_text

    # start_index = content.rfind("<root>")
    # end_index = content.rfind("</root>") + len("</root>")
    # root_content = content[start_index:end_index]


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run KSP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../Data/KSP/', help='../Data/finetune_data/test_1/KSP/')
    parser.add_argument('--img_dir', type=str, default='../../Data/KSP/Images', help='../image_data/SPP/')

    # Parse the argument
    args = parser.parse_args()

    # Script logic using args.model as the model name
    MODEL = str(args.model)

    DATA_PATH = args.data_dir
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
    kspData = load_data()
    # kspData = kspData[:2]
    kspdResults = []
    print('number of datapoints: ', len(kspData))

    print("Using model: {}".format(MODEL))

    outputs = run_opensource_KSP(kspData, imgdir=IMGDATA_PATH)
    # sys.exit()
    for q, output in zip(kspData, outputs):
        output_dict = {}
        # if MODEL == 'llava':
        #     print("==========================")
        #     print("output_reasoning_list:", output)
        #     print("***************************")
        #     output, reasoning = parse_xml_to_dict(llava_out_process(output))
        #     output_dict['output'] = output
        #     output_dict['correctness'] = ksp_check_single(q, output)
        #     output_dict['reasoning'] = reasoning
        #     kspdResults.append(output_dict)
        # elif MODEL == 'qwen_vl':
        #     print("+++++++++++++++++++++++++")
        #     print("output_reasoning_list:", output)
        #     print("-------------------------")
        #     print("==========================")
        #     print("output_reasoning_list:", qwen_out_process(output))
        #     print("***************************")
        #     output, reasoning = parse_xml_to_dict(qwen_out_process(output))
        #     output_dict['output'] = output
        #     output_dict['correctness'] = ksp_check_single(q, output)
        #     output_dict['reasoning'] = reasoning
        #     kspdResults.append(output_dict)
        # else:
        output, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = output
        output_dict['correctness'] = ksp_check_single(q, output)
        output_dict['reasoning'] = reasoning
        kspdResults.append(output_dict)
        # output_reasoning_list = parse_xml_to_matched_list(output)
        # output_dict['output'] = output_reasoning_list
        # output_dict['correctness'] = ksp_check(q, output_reasoning_list)
        # output_dict['reasoning'] = ""
        # kspdResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'kspResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(kspdResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'kspResults.json', 'w') as f:
            f.write(json.dumps(kspdResults) + '\n')


# if __name__ == '__main__':
#     kspData = load_data()
#     kspResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 1
#     for q in kspData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runKSP(q)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = kspCheck(q, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             kspResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             kspResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'kspResults.json', 'a') as f:
#         f.write(json.dumps(kspResults) + '\n')