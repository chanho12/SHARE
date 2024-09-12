import os, sys
import json, csv
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
        #ppl_input = extract_data_with_tag(prompt)

        utterance = list(prompt.values())[0]['real answer']
        response = prompt['model response']
        
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
        ppl_input = '12'
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


    #data = read_json_file(args.data_path)


    file_path = args.data_path

    # JSON 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    bert_list = []
    rouge_list = []
    bleu_1_list = []
    bleu_2_list = []
    bleu_3_list = []
    bleu_4_list = []
    infer_list = []
    ppl_list = []

    bertscore_eval = load("bertscore")
    rouge_eval = load("rouge")


    for num, prompt in enumerate(data):

        print(num)
        (
            bert_score,
            rouge_score,
            bleu_1_score,
            bleu_2_score,
            bleu_3_score,
            bleu_4_score,
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
        bert_list.append(bert_score)
        rouge_list.append(rouge_score)
        bleu_1_list.append(bleu_1_score)
        bleu_2_list.append(bleu_2_score)
        bleu_3_list.append(bleu_3_score)
        bleu_4_list.append(bleu_4_score)
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
        bleu_3_score,
        bleu_4_score,
        distinct_1,
        distinct_2,
        ppl
    ) = get_result(
        bert_list, rouge_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list,infer_list, ppl_list
    )
    json_dict = {
        'name': f'{args.task_name, args.data_path, args.model_name, num}',
        'ppl': round(ppl, 4),
        'bleu-1/2': f'{round(bleu_1_score, 4)}/{round(bleu_2_score, 4)}',
        'bleu-3/4': f'{round(bleu_3_score, 4)}/{round(bleu_4_score, 4)}',
        'rouge-1/2': f'{round(rouge1, 4)}/{round(rouge2, 4)}',
        'rougeL': round(rougeL, 4),
        'bert_score': round(bert_score, 4),
        'distinct_1/2': f'{round(distinct_1, 4)}/{round(distinct_2, 4)}'
    }

    
    def save_as_json(dictionary, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, ensure_ascii=False, indent=4)


    json_file_path = 'result.json'
    save_as_json(json_dict, json_file_path)

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{args.data_path}.json"
    
    with open(filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)    

    df = pd.DataFrame([data])

    # Excel 파일로 저장
    excel_file = f'{args.data_path}.xlsx'
    df.to_excel(excel_file, index=False)

    print(f"JSON 파일이 성공적으로 {excel_file}로 변환되었습니다.")



    print(
        f"PPL : {ppl}, \nBertScore : {bert_score} \nrouge1 : {rouge1} \nrouge2 : {rouge2} \nrougeL : {rougeL} \nrougeLsum : {rougeLsum}, \nbleu_1 : {bleu_1_score} \nbleu_2 : {bleu_2_score} "
    )
    print(f"distinct 1/2 {distinct_1, distinct_2}")


if __name__ == "__main__":  
    main()
   