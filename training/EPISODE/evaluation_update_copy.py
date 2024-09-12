import os, sys
import json, csv
import gspread
import pandas as pd
import openpyxl
from datetime import datetime

current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)
from utils.eval_utils import get_ppl, read_json_file, get_result
from utils.model_utils import (
    get_base_model,
    get_peft_gemma,
    get_peft_llama,
    get_base_llama,
    generate,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json, nltk
import torch
import argparse
from evaluate import load

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument(
        "--model_name", type=str, help="Choose the Model which you want to use"
    )
    parser.add_argument("--task_name", type=str, help="task_name")
    parser.add_argument("--tag_name", type=str, help="Gold or gpt")
    parser.add_argument("--data_path", type=str, help="data_path")
    args = parser.parse_args()
    return args


def extract_data_with_tag(prompt):
    prompt = list(prompt.values())[0]
    prompts = prompt["generate prompt"].rstrip() + "\n"
    ppl_input = prompts + prompt['model response']

    return ppl_input

def evaluation_chat_system(
    name, model, tokenizer, prompt, device, bert_eval, rouge_eval
):
    if name == "w_tag":
        print("This is with tag evaluation")
        print()
        
        # utterance : correct answer
        # response : Model-generated answer
        ppl_input = extract_data_with_tag(prompt)

        utterance = list(prompt.values())[0]['real answer']
        response = list(prompt.values())[0]['model response']
        
        print(f"prediction : {response}")
        print(f"Real answer : {utterance}")

        output_list = [response.strip()]
        last_utter_list = [utterance.strip()]

        reference = [utterance.split()]
        candidate = response.split()

        # evalation
        bert_score = bert_eval.compute(
            predictions=output_list, references=last_utter_list, lang = 'en')
        rouge_score = rouge_eval.compute(
            predictions=output_list, references=last_utter_list
        )

        ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # Trigram BLEU (n=3)
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )

        ### ppl
        ppl = get_ppl(ppl_input, model, tokenizer, device)

        print(f"Bert Score : {bert_score}")
        print(f"Rouge Score : {rouge_score}")
        print(f"bleu 1/2 : {bleu_unigram} {bleu_bigram}")
        print(f"bleu 3/4 : {bleu_trigram} {bleu_fourgram}")
        print(f"ppl : {ppl}")

        return bert_score, rouge_score, bleu_unigram, bleu_bigram, bleu_trigram, bleu_fourgram, response, ppl


def main():

    args = parse_args()
    print("evaluation start")


    print(args.model_name)
    print(args.task_name)

    data = read_json_file(args.data_path)
#    with open(args.data_path, 'r', encoding='utf-8') as file:
#        data = json.load(file)

    all_data = []

    for file in data:
        new_data = []
        accumulate = file[0]
        accumulate_model_select = accumulate[0]
        accumulate_model_response = accumulate[1]
        prompt_ = '''\nTask: Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:\n\n1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.\n2. Everyday language dialogues: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.\n\n**Dialogue History**:'''
        prompt = file[3].replace(prompt_, '')
        
        update = file[1]
        update_model_select = update[0]
        update_model_response = update[1]

        real_answer = file[2]

        utterance = real_answer
        response = accumulate_model_response

        reference = [utterance.split()]
        candidate = response.split()

        ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # Trigram BLEU (n=3)
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )

        accumulate_result = {}
        accumulate_result['bleu_1'] = bleu_unigram
        accumulate_result['bleu_2'] = bleu_bigram
        accumulate_result['bleu_3'] = bleu_trigram
        accumulate_result['bleu_4'] = bleu_fourgram
        accumulate_result['accumulate'] = accumulate
        accumulate_result['model_select'] = accumulate_model_select
        accumulate_result['model_response'] = accumulate_model_response
        accumulate_result['answer'] = utterance

        

        utterance = real_answer
        response = update_model_response


        reference = [utterance.split()]
        candidate = response.split()
        
                ## bleu

        weights_unigram = (1, 0, 0, 0)
        bleu_unigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_unigram,
            smoothing_function=SmoothingFunction().method1,
        )

        weights_bigram = (0.5, 0.5, 0, 0)
        bleu_bigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_bigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # Trigram BLEU (n=3)
        weights_trigram = (0.33, 0.33, 0.33, 0)
        bleu_trigram = sentence_bleu(
            reference,
            candidate,
            weights=weights_trigram,
            smoothing_function=SmoothingFunction().method1,
        )

        # 4-gram BLEU (n=4)
        weights_fourgram = (0.25, 0.25, 0.25, 0.25)
        bleu_fourgram = sentence_bleu(
            reference,
            candidate,
            weights=weights_fourgram,
            smoothing_function=SmoothingFunction().method1,
        )

        update_result = {}
        update_result['bleu_1'] = bleu_unigram
        update_result['bleu_2'] = bleu_bigram
        update_result['bleu_3'] = bleu_trigram
        update_result['bleu_4'] = bleu_fourgram
        update_result['update'] = update
        update_result['model_select'] = update_model_select
        update_result['model_response'] = update_model_response
        update_result['answer'] = utterance



        new_data.append(accumulate_result)
        new_data.append(update_result)
        new_data.append(prompt)
        new_data.append(accumulate_result['bleu_4']- update_result['bleu_4'])

        all_data.append(new_data)

    with open('new_data.json', 'w', encoding='utf-8') as file:
        json.dump(all_data, file, ensure_ascii=False, indent=4)
            

        
        

        



if __name__ == "__main__":  
    main()
   