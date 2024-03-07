import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models import *
from prompts import tsp_dPrompts
from check.check_cmp_TSP_D import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
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
    for level in range(start, n):
        for file_num in range(10):
            df = pd.read_csv(data_path + "decision_data_TSP_level_{}_instance_{}.csv".format(level, file_num + 1),
                             header=None, 
                             index_col=False)
            all_data.append(df)
    return all_data

def runTSP_D(adj_matrix, distance_limit, p=tsp_dPrompts):
    total_cities = adj_matrix.shape[0] # exclude the last row
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(total_cities=total_cities, distance_limit=distance_limit) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + '\n' + \
                  'The distances between cities are below: \n'
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i < j:  # only use the upper triangle
                this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
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
    parts = filename[:-4].split('_')
    return int(parts[-2]), int(parts[-1])


def run_opensource_TSP_D(qs, p=tsp_dPrompts, imgdir=None):
    all_prompts = []
    imgcnt = 0

    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []
    outputs_tmp = []
    for q in tqdm(qs):
        threshold = q.iloc[-1, 0] # therashold is the last row
        adj_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
        total_cities = adj_matrix.shape[0] # exclude the last row
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
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        
        #=====================limitedtext
        # threshold = q.iloc[-1, 0] # therashold is the last row
        # adj_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
        # total_cities = adj_matrix.shape[0] # exclude the last row
        # prompt_text = p['Intro'] + '\n' + \
        #             p['Initial_question'].format(total_cities=total_cities, distance_limit=threshold) + '\n' + \
        #             p['Output_content'] + '\n' + \
        #             p['Output_format'] + '\n' + \
        #             'The distances between cities are shown in figure: \n'
        
        # for i in range(adj_matrix.shape[0]):
        #     for j in range(adj_matrix.shape[1]):
        #         if i < j:  # only use the upper triangle
        #             this_line = "The distance between City {} and City {} is {}.".format(i, j, adj_matrix[i, j])
        #             prompt_text += this_line + '\n'
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
        print("未找到 ASSISTANT: <root> 之后的内容")


def qwen_out_process(res):
    content = res
    end_tag_index = content.rfind("</root>")
    start_tag_index = content.rfind("<root>", 0, end_tag_index)
    if start_tag_index != -1 and end_tag_index != -1:
        return content[start_tag_index + len("<root>"):end_tag_index]
    else:
        return res


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run TSP-D model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../Data/TSP_Decision/', help='../Data/finetune_data/test_1/TSP_Decision/')
    parser.add_argument('--img_dir', type=str, default='../../Data/TSP_Decision/Images', help='../image_data/SPP/')

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

    tsp_d_Data = load_data()
    # tsp_d_Data = tsp_d_Data[:2]
    print(len(tsp_d_Data))
    tsp_d_Results = []

    outputs = run_opensource_TSP_D(tsp_d_Data, imgdir=IMGDATA_PATH)
    # sys.exit()
    for result, instance in zip(outputs, tsp_d_Data):
        output_dict = {}
        threshold = instance.iloc[-1, 0] # therashold is the last row
        distance_matrix = instance.iloc[:-1].values # distance matrix is the rest of the rows
        # if MODEL == 'llava':
        #     print("==========================")
        #     print("output_reasoning_list:", llava_out_process(result))
        #     print("***************************")
        #     output, reasoning = parse_xml_to_dict(llava_out_process(result))
        #     output_dict['output'] = output
        #     output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
        #     output_dict['reasoning'] = reasoning
        #     tsp_d_Results.append(output_dict)
        #     # print("results:", gcpResults)
        # elif MODEL == 'qwen_vl':
        #     print("+++++++++++++++++++++++++")
        #     print("output_reasoning_list:", result)
        #     print("-------------------------")
        #     print("************************")
        #     print("output_reasoning_list:", qwen_out_process(result))
        #     print("==========================")
        #     output, reasoning = parse_xml_to_dict(qwen_out_process(result))
        #     output_dict['output'] = output
        #     output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
        #     output_dict['reasoning'] = reasoning
        #     tsp_d_Results.append(output_dict)
        # else:
        #     output, reasoning = parse_xml_to_dict(output)
        #     output_dict['output'] = output
        #     output_dict['correctness'] = tsp_decision_check(q, output)
        #     output_dict['reasoning'] = reasoning
        #     tsp_d_Results.append(output_dict)
        output, reasoning = parse_xml_to_dict(result)
        output_dict['output'] = output
        output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
        output_dict['reasoning'] = reasoning
        tsp_d_Results.append(output_dict)

    # Save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'tsp_d_Results_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(tsp_d_Results) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'tsp_d_Results.json', 'w') as f:
            f.write(json.dumps(tsp_d_Results) + '\n')


# if __name__ == '__main__':
#     # Create the parser
#     parser = argparse.ArgumentParser(description='Run TSP-D model script')

#     # Add an argument for the model name
#     parser.add_argument('model', type=str, help='The name of the model to run')

#     # Parse the argument
#     args = parser.parse_args()

#     # Script logic using args.model as the model name
#     MODEL = str(args.model)

#     DATA_PATH = '../Data/TSP_Decision/'
#     RESULT_PATH = '../Results/'


#     tsp_d_Data = load_data()
#     print(len(tsp_d_Data))
#     tsp_d_Results = []

#     MAX_TRY = 10
#     for q in tsp_d_Data:
#         threshold = q.iloc[-1, 0] # therashold is the last row
#         distance_matrix = q.iloc[:-1].values # distance matrix is the rest of the rows
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runTSP_D(distance_matrix, threshold)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = tsp_decision_check(distance_matrix, threshold, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             tsp_d_Results.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             tsp_d_Results.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'tsp_d_Results.json', 'a') as f:
#         f.write(json.dumps(tsp_d_Results) + '\n')