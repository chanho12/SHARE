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

def main():

    data = {"prompt": "You are a response generator. You will be provided with Memory and Current Conversation, and based on these, you need to create the following response. \n\nThere are two types of Memory:\n1. **Memory-driven dialogues**: If the cue within parentheses details specific character traits or background context, generate responses that reflect these memory-driven elements, ensuring character consistency and rich context.\n2. **Everyday language dialogues**: If the cue within parentheses is labeled \"Everyday Language,\" generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.\n\nWhen generating responses, follow these instructions:\n1. Responses generated based on Memory should be rich in expression and detailed.\n2. If you think the Memory is not suitable for the current dialogue, generate responses based on the current dialogue.\n3. For Everyday Language, create engaging responses that can spark the reader\u2019s interest. Because Everyday Language responses vary, look at the Dialogue History and create answers that anyone can relate to and find enjoyable.\n**Dialogue**: \nMOZART: (Everyday Language) What happened? Is it over?\nSALIERI: (SALIERI is caring and responsible, SALIERI offers to take MOZART home due to his poor health) I'm taking you home. You're not well.\nMOZART: (MOZART is about to become a father, he needs a job urgently due to financial pressures, and he is working on a project that he believes will become a huge success in six to eight months)"}

            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_peft_llama('chano12/generate_new', device)
    input__ = data['prompt']
    input_ = tokenizer(input__, return_tensors="pt").to(device)
    output = generate(
                model,
                tokenizer,
                input_,
                num_beams=1,
                do_sample = False,
                num_return_sequences=1,
                max_new_tokens=300,
            )
    print(output)


if __name__ == "__main__":
    main()