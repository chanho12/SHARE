import os, sys, json
from openai import OpenAI
current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)

import argparse
from itertools import islice

from utils.eval_utils import get_ppl, read_json_file, get_result
from utils.model_utils import (
    get_base_model,
    get_peft_gemma,
    get_peft_llama,
    get_base_llama,
    generate,
)

import torch

def generate(
    model,
    tokenizer,
    inputs,
    num_beams=3,
    num_beam_groups=1,
    do_sample=True,
    num_return_sequences=1,
    max_new_tokens=100,
):
    generate_ids = model.generate(
        inputs.input_ids,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,

        )
    result = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument("--data_path", type=str, help="data_path")
    parser.add_argument('--data_name', type= str, help = 'gpt4o,gpt3.5 turbo')
    parser.add_argument('--model_path', type = str, help = 'generated_data_name')
    parser.add_argument('--task_name', type = str, help = 'persona, personal, shared')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    path = args.data_path

    data = read_json_file(path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_peft_llama(args.model_path, device)

    json_list1 = []

    for num, update in enumerate(data):
        print(num)
        input__ = update['prompt']
        input_ = tokenizer(input__, return_tensors="pt").to(device)
        output = generate(
                model,
                tokenizer,
                input_,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=300,
            )
        print(output)
        json_list1.append(output)
        if num == 100:
            with open(args.data_name, "w", encoding="utf-8") as file:
                json.dump(json_list1, file, ensure_ascii=False, indent=4)

    with open(args.data_name, "w", encoding="utf-8") as file:
        json.dump(json_list1, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()