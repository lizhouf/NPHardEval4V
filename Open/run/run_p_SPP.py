import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
# from models import *
from prompts import sppPrompts
from check.check_p_SPP import *
from tqdm import tqdm
# import pandas as pd
import numpy as np
import json
import argparse
from utils import run_opensource_models

import logging



def load_data(DATA_PATH):
    data_path = DATA_PATH
    with open(data_path + "spp_instances.json", 'r') as f:
        all_data = json.load(f)

    return all_data

def runSPP(MODEL,q, p=sppPrompts):
    # start_node = q['start_node']
    # end_node = q['end_node']
    # TO-DO: fix later
    start_node = q['nodes'][0]
    end_node = q['nodes'][-1]
    edges = q['edges']
    prompt_text = p['Intro'] + '\n' + \
                  p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                  p['Output_content'] + '\n' + \
                  p['Output_format'] + \
                  "\n The graph's edges and weights are as follows: \n"
    for edge in edges:
        this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
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


def run_opensource_SPP(qs, p=sppPrompts, imgdir=None):
    imgcnt = 0
    all_txtprompts = []
    assigned_image_numbers = {}
    image_files = [file for file in os.listdir(imgdir)]
    sorted_files = sorted(image_files, key=custom_sort)
    outputs = []
    outputs_tmp = []
    for q in tqdm(qs):
        if MODEL == 'kosmos2':
            start_node = q['nodes'][0]
            end_node = q['nodes'][-1]
            edges = q['edges']
            prompt_text = "<grounding> " + p['Intro'] + '\n' + \
                        p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                        p['Output_content'].format(start_node=start_node, end_node=end_node) + '\n' + \
                        p['Output_format'] + \
                        "\n The graph's edges and weights are as follows: \n"
            for edge in edges:
                this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
                prompt_text += this_line + '\n'
            prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
            prompt_text += 'Answer:\n'
        # all_txtprompts.append(prompt_text)
        elif (MODEL == 'fuyu_8b') | (MODEL == 'qwen_vl') | (MODEL == 'blip2') | (MODEL == 'otter') | \
            (MODEL == 'flamingo') | (MODEL == 'cogvlm') | (MODEL == 'llava'):
            start_node = q['nodes'][0]
            end_node = q['nodes'][-1]
            edges = q['edges']
            prompt_text = p['Intro'] + '\n' + \
                          p['Initial_question'].format(start_node=start_node, end_node=end_node) + '\n' + \
                          p['Output_content'].format(start_node=start_node, end_node=end_node) + '\n' + \
                          p['Output_format']  + \
                          "\n The graph's edges and weights are as follows: \n"
            for edge in edges:
                this_line = f"Edge from {edge['from']} to {edge['to']} has a weight of {edge['weight']}."
                prompt_text += this_line + '\n'
            prompt_text += "The above information contains the specific content of the question. Text provide instruction; both the text and the picture provide data."
            prompt_text += 'Answer:\n'
            # print("+++++++++++++++++++++++++")
            # print("prompt_text:", prompt_text)
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
        else:
            assert("MODEL None!")



        imgprompts = os.path.join(imgdir, sorted_files[imgcnt])
        print("=============check_prompt==================")
        print("check prompt")
        print("prompt_text:", prompt_text)
        print("prompt_img:", imgprompts)
        print("=============check_prompt==================")
        output = run_opensource_models(args, MODEL, prompt_text, imgprompts)
        # print("==================================")
        # print("output:", output)
        # print("==================================")
        result = {
            'promts_num':imgcnt,
            'data':output
        }
        outputs_tmp.append(result)
        outputs.append(output)

        # import logging
        # logging.basicConfig(filename=f'{MODEL}.log', level=logging.INFO,
        #                     format='%(asctime)s - %(levelname)s - %(message)s')
        # output_str = output
        # logging.info(output_str)
        # print('image_count:',imgcnt)
        if imgcnt >= 99:
            import pickle
            with open('{}_{}_Results_fulltext_figure.pkl'.format(DATA_PATH.split('/')[-2], MODEL), 'wb') as json_file:
                pickle.dump(outputs_tmp, json_file)
                # json_file.write('\n')
            break
        imgcnt += 1
        # print("字符串已成功写入日志文件 'output.log'")


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
    text_content = res
    start_index = text_content.rfind("<reasoning>")
    end_index = text_content.rfind("</final_answer>")
    desired_text = text_content[start_index:end_index+len("</final_answer>")]
    return desired_text

def blip_out_process(res):
    corrected_text = res.replace("root>", "<root>").replace("/root>", "</root>").replace("reasoning>", "<reasoning>").replace("/reasoning>", "</reasoning>").replace("final_answer>", "<final_answer>").replace("/final_answer>", "</final_answer>")

    return corrected_text

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run SPP model script')

    # Add an argument for the model name
    parser.add_argument('model', type=str, help='The name of the model to run')
    parser.add_argument('--tuned_model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../Data/SPP/', help='../Data/finetune_data/test_1/SPP/')
    parser.add_argument('--img_dir', type=str, default='../../Data/SPP/Images', help='../image_data/SPP/')

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
    sppData = load_data(DATA_PATH)
    # sppData = sppData[:2]
    sppResults = []
    print('number of datapoints: ', len(sppData))

    print("Using model: {}".format(MODEL))
    logging.basicConfig(filename=f'{MODEL}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    outputs = run_opensource_SPP(sppData, imgdir=IMGDATA_PATH)
    # sys.exit()
    for q, output in zip(sppData, outputs):
        output_dict = {}
    #     if MODEL == 'llava':
    #         # print("==========================")
    #         # print("output_reasoning_list:", output[0]['generated_text'])
    #         # print("***************************")
    #         output, reasoning = parse_xml_to_dict(llava_out_process(output))
            
    #         output_dict['output'] = output
    #         output_dict['correctness'] = spp_check_single(q, output)
    #         output_dict['reasoning'] = reasoning
    #         sppResults.append(output_dict)
    #     elif MODEL == 'qwen_vl':
    #         print("==========================")
    #         print("output_reasoning_list:", output)
    #         print("***************************")
    #         output, reasoning = parse_xml_to_dict(qwen_out_process(output))
    #         output_dict['output'] = output
    #         output_dict['correctness'] = spp_check_single(q, output)
    #         output_dict['reasoning'] = reasoning
    #         sppResults.append(output_dict)
    #     elif MODEL == 'blip2':
    #         print("==========================")
    #         print("output_reasoning_list:", output)
    #         print("***************************")
    #         print("++++++++++++++++++++++++++++")
    #         print("output_reasoning_list:", blip_out_process(output))
    #         print("----------------------------")
    #         output, reasoning = parse_xml_to_dict(blip_out_process(output))
    #         output_dict['output'] = output
    #         output_dict['correctness'] = spp_check_single(q, output)
    #         output_dict['reasoning'] = reasoning
    #         sppResults.append(output_dict)
    #     else:
            # print("==========================")
            # print("output:", output)
            # print("***************************")
        
        output, reasoning = parse_xml_to_dict(output)
        output_dict['output'] = output
        output_dict['correctness'] = spp_check_single(q, output)
        output_dict['reasoning'] = reasoning
        sppResults.append(output_dict)
    # save the results
    if args.tuned_model_dir:
        number_of_benchmarks = args.tuned_model_dir.split('_')[-1]
        with open(RESULT_PATH+MODEL+'_'+'sppResults_benchmarks{}.json'.format(number_of_benchmarks), 'w') as f:
            f.write(json.dumps(sppResults) + '\n')
    else:
        with open(RESULT_PATH+MODEL+'_'+'sppResults.json', 'w') as f:
            f.write(json.dumps(sppResults) + '\n')


# if __name__ == '__main__':
#     sppData = load_data()
#     sppResults = []

#     print("Using model: {}".format(MODEL))

#     MAX_TRY = 10
#     for q in sppData:
#         output_dict = {}
#         num_try = 0
#         while num_try < MAX_TRY:
#             try:
#                 llm_string = runSPP(q)
#                 output = parse_xml_to_dict(llm_string)
#                 output_dict['output'] = output
#                 output_dict['correctness'] = spp_check(q, output)
#                 break
#             except Exception as e:
#                 print(f"Attempt {num_try + 1} failed: {e}")
#                 num_try += 1
#         if output_dict:
#             sppResults.append(output_dict)
#         else:
#             print(f"Failed to run {q}")
#             sppResults.append({'output': '', 'correctness': False})

#     # Save the results
#     with open(RESULT_PATH + MODEL + '_' + 'sppResults.json', 'a') as f:
#         f.write(json.dumps(sppResults) + '\n')
