import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models import *
from prompts import bspPrompts
from check.check_p_BSP import *
#from utils import parse_xml_to_dict
import time
from utils import run_opensource_models
import re
import pandas as pd
import numpy as np
import json
import argparse
from tqdm import tqdm

def load_data():
    data_path = DATA_PATH
    with open(data_path + "bsp_instances.json", 'r') as f:
        all_data = json.load(f)
    return all_data

def runBSP(MODEL, q, p=bspPrompts):
    target_value = q['target']
    # TO-DO: fix data not being sorted
    array = sorted(q['array'])
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(target_value=target_value) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'

    if 'gpt' in MODEL:
        output = run_gpt(prompt_text, model=MODEL)
    elif 'claude' in MODEL:
        output = run_claude(prompt_text, model=MODEL)
    else:
        print('Model not found')
        return None

    return output

def construct_few_shot_examples(examples):
    few_shot_examples = '\n\nBelow are {} examples including pairs of questions and answers:\n\n'.format(len(examples))
    for i, example in enumerate(examples):
        question = 'The sorted array elements are: ' + ', '.join(str(a) for a in example['question']['array'])
        few_shot_examples += '<example{}>\nQuestion:\n'.format(i+1)+question + '\nAnswer:\n' + str(example['output']) + '\n</example{}>\n\n'.format(str(i+1))

    return few_shot_examples

def custom_sort(filename):
    parts = filename[:-4].split('_')  # 去掉后缀名，并按 '_' 分割
    return int(parts[-2]), int(parts[-1])


def run_opensource_BSP(args, qs, p=bspPrompts, imgdir=None):
    all_prompts = []
    imgcnt = 0
    all_txtprompts = []
    assigned_image_numbers = {}
    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []
    outputs_tmp = []
    for i, q in enumerate(tqdm(qs)):
        target_value = q['target']
        # TO-DO: fix data not being sorted
        array = sorted(q['array'])
        prompt_text = p['Intro'] + '\n' + \
                    p['Initial_question'].format(target_value=target_value) + '\n' + \
                    p['Output_content'] + '\n' + \
                    p['Output_format'] + \
                    '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'
        prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
        
        #==================limitedtext================
        # target_value = q['target']
        # # TO-DO: fix data not being sorted
        # array = sorted(q['array'])
        # prompt_text = p['Intro'] + '\n' + \
        #             p['Initial_question'].format(target_value=target_value) + '\n' + \
        #             p['Output_content'] + '\n' + \
        #             p['Output_format'] 
                    # '\n The sorted array elements are: ' + ', '.join(map(str, array)) + '\n'
        # all_prompts.append(prompt_text)
        # print("prompts:",all_prompts)
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


def run_opensource_fewshot_BSP(args, qs, p=bspPrompts):
    FEWSHOT_DATA_PATH = '../Data/{}/few_shot/{}_few_shots.json'.format(args.prompt_question_type, args.prompt_question_type)
    with open(FEWSHOT_DATA_PATH, 'r') as f:
        fewshot_data = json.load(f)

    all_prompts = []
    for i, q in enumerate(tqdm(qs)):
        dif_level = (i//10) + args.difficulty_level
        if dif_level < 0:
            continue
        examples = [d for d in fewshot_data if d['complexity_level'] == dif_level+1][:args.example_number]
        few_shot_examples = construct_few_shot_examples(examples)
        target_value = q['target']
        # TO-DO: fix data not being sorted
        array = sorted(q['array'])
        prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(target_value=target_value) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  few_shot_examples + \
                  'Again, ' + p['Initial_question'].format(target_value=target_value) + '\n' + \
                  'Follow the format in the above examples to write your answer.\n' +\
                  '\nThis is the new question you need to solve:\n\nQuestion:\nThe sorted array elements are: ' + ', '.join(map(str, array)) + '\nAnswer:\n'
        all_prompts.append(prompt_text)

    output = run_opensource_models(args, MODEL, all_prompts)
    return output

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


def blip_out_process(res):
    text_content = res
    start_index = text_content.rfind("<root>")
    end_index = text_content.rfind("</root>")+len("</root>")
    desired_text = text_content[start_index:end_index]
    return desired_text

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Run BSP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--data_dir', type=str, default='../../Data/BSP/', help='../Data/finetune_data/test_1/BSP/')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--fewshot', type=bool, default=False)
    parser.add_argument('--difficulty_level', type=int, default=0, help="-5, -4, -3, ...")
    parser.add_argument('--example_number', type=int, default=5, help="2,3,4,5")
    parser.add_argument('--img_dir', type=str, default='../../Data/BSP/Images', help='../image_data/SPP/')

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
    bspData = load_data()
    # bspData = bspData[:2]
    bspResults = []
    print('number of datapoints: ', len(bspData))

    print("Using model: {}".format(MODEL))

    if args.fewshot:
        outputs = run_opensource_fewshot_BSP(args, bspData)
    else:
        outputs = run_opensource_BSP(args, bspData, imgdir=IMGDATA_PATH)
        # sys.exit()
    for q, output in zip(bspData, outputs):
        output_dict = {}
        if MODEL == 'llava':
            print("==========================")
            print("output_reasoning_list:", output)
            print("***************************")
            output, reasoning = parse_xml_to_dict(llava_out_process(output))
            output_dict['output'] = output
            output_dict['correctness'] = bsp_check(q, output)
            output_dict['reasoning'] = reasoning
            bspResults.append(output_dict)
        elif MODEL == 'qwen_vl':
            print("+++++++++++++++++++++++++")
            print("output_reasoning_list:", output)
            print("-------------------------")
            print("==========================")
            print("output_reasoning_list:", qwen_out_process(output))
            print("***************************")
            output, reasoning = parse_xml_to_dict(qwen_out_process(output))
            output_dict['output'] = output
            output_dict['correctness'] = bsp_check(q, output)
            output_dict['reasoning'] = reasoning
            bspResults.append(output_dict)
        elif MODEL == 'blip2':
            print("+++++++++++++++++++++++++")
            print("output_reasoning_list:", output)
            print("-------------------------")
            print("==========================")
            print("output_reasoning_list:", blip_out_process(output))
            print("***************************")
            output, reasoning = parse_xml_to_dict(blip_out_process(output))
            output_dict['output'] = output
            output_dict['correctness'] = bsp_check(q, output)
            output_dict['reasoning'] = reasoning
            bspResults.append(output_dict)
        else:
            output, reasoning = parse_xml_to_dict(output)
            output_dict['output'] = output
            output_dict['correctness'] = bsp_check(q, output)
            output_dict['reasoning'] = reasoning
            bspResults.append(output_dict)
        # output_dict['output'] = output
        # correctness = bsp_check(q,output)
        # output_dict['correctness'] = correctness
        # bspResults.append(output_dict)

    # save the results
    if args.fewshot:
        if args.tuned_model_dir:
            number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
            with open(RESULT_PATH+MODEL+'_'+'bspResults_{}_benchmarks{}.json'.format(args.difficulty_level, number_of_benchmarks), 'w') as f:
                f.write(json.dumps(bspResults) + '\n')
        else:
            with open(RESULT_PATH+MODEL+'_'+'bspResults_{}.json'.format(args.difficulty_level), 'w') as f:
                f.write(json.dumps(bspResults) + '\n')
    else:
        if args.tuned_model_dir:
            number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
            with open(RESULT_PATH+MODEL+'_'+'bspResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
                f.write(json.dumps(bspResults) + '\n')
        else:
            with open(RESULT_PATH+MODEL+'_'+'bspResults.json', 'w') as f:
                f.write(json.dumps(bspResults) + '\n')



# if __name__ == '__main__':
#     bspData = load_data()
#     bspResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10
#     for q in bspData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runBSP(q)
#                 output, reasoning = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = bsp_check(q, output)
#                 output_dict['reasoning'] = reasoning
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             bspResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             bspResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'bspResults.json', 'a') as f:
#         f.write(json.dumps(bspResults) + '\n')