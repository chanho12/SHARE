import os, sys
import json, csv
import gspread
import pandas as pd
import openpyxl

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
    ppl_input = prompt['model response'].rstrip() + "\n"

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

        ### ppl
        ppl = get_ppl(ppl_input, model, tokenizer, device)

        print(f"Bert Score : {bert_score}")
        print(f"Rouge Score : {rouge_score}")
        print(f"bleu 1/2 : {bleu_unigram} {bleu_bigram}")
        print(f"ppl : {ppl}")

        return bert_score, rouge_score, bleu_unigram, bleu_bigram, response, ppl


def main():

    args = parse_args()
    print("evaluation start")

    if args.task_name == "wo_tag":
        path = "_without_tag"
        print(path)
    elif args.task_name == "w_tag":
        path = "_with_tag"
        print(path)


    print(args.model_name)
    print(args.task_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_name == "llama":

        print(args.model_name)
        path = f"chano12/{args.model_name}" + path
        model, tokenizer = get_peft_llama(path, device)
        model.eval()

    elif args.model_name == "base_llama":
        path = "meta-llama/Llama-2-7b-chat-hf"
        print(args.model_name)
        model, tokenizer = get_base_llama(path, device)
        model.eval()

    elif args.model_name == "gemma":
        path = f"chano12/{args.model_name}" + path
        print(args.model_name)
        model, tokenizer = get_peft_gemma(path, device)
        model.eval()

    elif args.model_name == "base_gemma":
        path = "google/gemma-2b"
        print(args.model_name)
        model, tokenizer = get_base_model(path, device)
        model.eval()


    data = read_json_file(args.data_path)

    bert_list = []
    rouge_list = []
    bleu_1_list = []
    bleu_2_list = []
    infer_list = []
    ppl_list = []

    bertscore_eval = load("bertscore")
    rouge_eval = load("rouge")

    print()

    for num, prompt in enumerate(data[:2]):

        print(num)
        (
            bert_score,
            rouge_score,
            bleu_1_score,
            bleu_2_score,
            infer_sentence,
            ppl_score,
        ) = evaluation_chat_system(
            "w_tag",
            model,
            tokenizer,
            prompt,
            device,
            bertscore_eval,
            rouge_eval,
        )

        prompt['generate_sentence'] = infer_sentence
        bert_list.append(bert_score)
        rouge_list.append(rouge_score)
        bleu_1_list.append(bleu_1_score)
        bleu_2_list.append(bleu_2_score)
        infer_list.append(infer_sentence)
        ppl_list.append(ppl_score)

    (
        bert_score,
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
        bleu_1_score,
        bleu_2_score,
        distinct_1,
        distinct_2,
        ppl
    ) = get_result(
        bert_list, rouge_list, bleu_1_list, bleu_2_list, infer_list, ppl_list
    )
    json_dict = {
        'ppl': round(ppl, 4),
        'bleu-1/2': f'{round(bleu_1_score, 4)}/{round(bleu_2_score, 4)}',
        'rouge-1/2': f'{round(rouge1, 4)}/{round(rouge2, 4)}',
        'rougeL': round(rougeL, 4),
        'bert_score': round(bert_score, 4),
        'distinct_1/2': f'{round(distinct_1, 4)}/{round(distinct_2, 4)}'
}
    
    def save_as_json(dictionary, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, ensure_ascii=False, indent=4)

    path = 'tag_by_gpt-3.5_eval2_v1.json'

    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    json_file_path = 'result.json'
    save_as_json(json_dict, json_file_path)

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)


    data = [data]

    df = pd.DataFrame(data)
    print(df)

    df.to_excel('result.xlsx')
    

    print(
        f"PPL : {ppl}, \nBertScore : {bert_score} \nrouge1 : {rouge1} \nrouge2 : {rouge2} \nrougeL : {rougeL} \nrougeLsum : {rougeLsum}, \nbleu_1 : {bleu_1_score} \nbleu_2 : {bleu_2_score} "
    )
    print(f"distinct 1/2 {distinct_1, distinct_2}")


if __name__ == "__main__":  
    main()
   