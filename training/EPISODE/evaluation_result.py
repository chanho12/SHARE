import os, sys, json

import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument(
        "--file1", type=str, help="Choose the Model which you want to use"
    )
    parser.add_argument("--file2", type=str, help="task_name")
    args = parser.parse_args()
    return args

def make_value(value1, value2):
    sum_ =  1120* value1 + 618 * value2
    return sum_/(1120 + 618) 


def main():
    args = parse_args()
    with open(args.file1, 'r', encoding='utf-8') as file:
        data1 = json.load(file)

    with open(args.file2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)

    ppl = make_value(data1['ppl'], data2['ppl'])
    bleu1 = make_value(float(data1['bleu-1/2'].split('/')[0]), float(data2['bleu-1/2'].split('/')[0]))
    bleu2 = make_value(float(data1['bleu-1/2'].split('/')[1]), float(data2['bleu-1/2'].split('/')[1]))
    bleu3 = make_value(float(data1['bleu-3/4'].split('/')[0]), float(data2['bleu-3/4'].split('/')[0]))
    bleu4 = make_value(float(data1['bleu-3/4'].split('/')[1]), float(data2['bleu-3/4'].split('/')[1]))

    rouge1 = make_value(float(data1['rouge-1/2'].split('/')[0]), float(data2['rouge-1/2'].split('/')[0]))
    rouge2 = make_value(float(data1['rouge-1/2'].split('/')[1]), float(data2['rouge-1/2'].split('/')[1]))
    rougeL = make_value(data1['rougeL'], data2['rougeL'])
    bert = make_value(data1['bert_score'], data2['bert_score'])
    distinct1 = make_value(float(data1['distinct_1/2'].split('/')[0]), float(data2['distinct_1/2'].split('/')[0]))
    distinct2 = make_value(float(data1['distinct_1/2'].split('/')[1]), float(data2['distinct_1/2'].split('/')[1]))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dict = {
        'ppl': round(ppl, 4),
        'bleu-1/2': f'{round(bleu1, 4)}/{round(bleu2, 4)}',
        'bleu-3/4': f'{round(bleu3, 4)}/{round(bleu4, 4)}',
        'rouge-1/2': f'{round(rouge1, 4)}/{round(rouge2, 4)}',
        'rougeL': round(rougeL, 4),
        'bert_score': round(bert, 4),
        'distinct_1/2': f'{round(distinct1, 4)}/{round(distinct2, 4)}'
    }
    filename = f"per_session_v6_{current_time}.json"
    
    with open(filename, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)

if __name__ == "__main__":  
    main()
   