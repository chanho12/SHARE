import os, sys, json
from openai import OpenAI
current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)
from utils.eval_utils import read_json_file
from itertools import islice

def get_response(prompt1, prompt2, model_name, api_key):
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,  # gpt-3.5-turbo, gpt-4o, gpt-3.5-turbo-0125
        messages=[
            {
                "role": "system",
                "content": f"{prompt1}",
            },
            {
                "role": "user",
                "content": f"{prompt2}",
            },
        ],
        temperature = 0.9,
        max_tokens = 4096,
     )
    return completion.choices[0].message.content

def make_dialogue(text_list):
    new = ''
    for line in text_list:
        speaker = line['speaker']
        text = line['text']
        label = line['label']
        new += f"{speaker}-#-{text}-#-{label}###"
    return new
        

def main():
    path = '/home/chanho/Model/EPISODE-update/results/new_update_v5_num5_llama5_final_4_session.json'
    model_name = 'gpt-4o' # gpt-4o-mini
    data = read_json_file(path)

    API_KEY = "sk-proj-aJ8bqWWKWonI0VDgh0b8T3BlbkFJWiszrlE8bNnx8krAz46h"

    json_list = []

    

    for file in data:
        key = list(file.keys())[0]
        value = list(file.values())[0]

        dialogue = value['generate prompt'].split("**Dialogue History**:")[1].strip()
        prompt1 = value['generate prompt'].split("**Dialogue History**:")[0].strip()

        SYSTEM = f'''Generate the next response in a dialogue by focusing on the contextual cues detailed within parentheses in the dialogue history. Responses should be tailored according to the type of cue provided:'''
        prompt = f'''\n\n1. Memory-driven dialogues: If the cue within parentheses details specific character traits or background context, craft responses that reflect these memory-driven elements, ensuring character consistency and rich context.
        \n2. Everyday language dialogues: If the cue within parentheses is labeled \"Everyday Language,\" generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.
        **Dialogue format** :
speaker1: (Memory) text
speaker2: (Memory) text
speaker1: (Memory) text 
speaker2: (Memory)

<Example>
**Dialogue History**:
NEFF: (Everyday Language) Why are you crying? You won't tell me?
LOLA: (LOLA considers NEFF a confidante, LOLA suspects her boyfriend Nino and Phyllis) Of course I will, Walter. I wouldn't tell anybody else but you. It's about Nino.
NEFF: (Everyday Language) Zachetti? What about him?
LOLA: (LOLA feels betrayed, LOLA suspects her boyfriend Nino and Phyllis) They killed my father together. He and Phyllis. He helped her do it. I know he did.
NEFF: (NEFF might be a more logical thinker) What makes you say that?
LOLA: (LOLA suspects her boyfriend Nino and Phyllis, LOLA's suspicions might all be in her mind) I've been following him. He's at her house, night after night. It was Phyllis and him all the time. Maybe he was going with me just for a blind. And the night of the murder --
NEFF: (NEFF's comments imply he's had previous conversations with LOLA about her father's death) You promised not to talk that way any more.
LOLA: (LOLA is prone to emotional distress and might be studying at U.C.L.A) -- he was supposed to pick me up after a lecture at U.C.L.A. -- but he never showed up. He said he was sick. Sick! He couldn't show up, because the train was leaving with my father on it. Maybe I'm just crazy. Maybe it's all just in my mind.
NEFF: (NEFF's comments imply he's had previous conversations with LOLA about her father's death) Sure, it's all in your mind.
LOLA: (Everyday Language)

output : Of course I will, Walter. I wouldn't tell anybody else but you. It's about Nino.

Now, read the conversation and generate the final response.

**Dialogue History**:
{dialogue}

output'''

        #SYSTEM = 'You are a response generator of last line. You will be provided with Memory and Current Conversation, and based on these, you need to create the following response.'
        prompt12 = f'''There are two types of Memory:
1. **Memory-driven dialogues**: If the cue within parentheses details specific character traits or background context, generate responses that reflect these memory-driven elements, ensuring character consistency and rich context.
2. **Everyday language dialogues**: If the cue within parentheses is labeled "Everyday Language," generate responses that are based on typical day-to-day interactions, free from specific personas or detailed context.
When generating responses, follow these instructions:
1. Responses generated based on Memory should be rich in expression and detailed.
2. If you think the Memory is not suitable for the current dialogue, generate responses based on the current dialogue.
3. For Everyday Language, create engaging responses that can spark the reader’s interest. Because Everyday Language responses vary, look at the Dialogue History and create answers that anyone can relate to and find enjoyable.
4. Based on the Memory within (), generate only the response. 
**Dialogue format** :
speaker1: (Memory) text
speaker2: (Memory) text
speaker1: (Memory) text 
speaker2: (Memory)

<Example>
NEFF: (Everyday Language) Why are you crying? You won't tell me?
LOLA: (LOLA considers NEFF a confidante, LOLA suspects her boyfriend Nino and Phyllis) Of course I will, Walter. I wouldn't tell anybody else but you. It's about Nino.
NEFF: (Everyday Language) Zachetti? What about him?
LOLA: (LOLA feels betrayed, LOLA suspects her boyfriend Nino and Phyllis) They killed my father together. He and Phyllis. He helped her do it. I know he did.
NEFF: (NEFF might be a more logical thinker) What makes you say that?
LOLA: (LOLA suspects her boyfriend Nino and Phyllis, LOLA's suspicions might all be in her mind) I've been following him. He's at her house, night after night. It was Phyllis and him all the time. Maybe he was going with me just for a blind. And the night of the murder --
NEFF: (NEFF's comments imply he's had previous conversations with LOLA about her father's death) You promised not to talk that way any more.
LOLA: (LOLA is prone to emotional distress and might be studying at U.C.L.A) -- he was supposed to pick me up after a lecture at U.C.L.A. -- but he never showed up. He said he was sick. Sick! He couldn't show up, because the train was leaving with my father on it. Maybe I'm just crazy. Maybe it's all just in my mind.
NEFF: (NEFF's comments imply he's had previous conversations with LOLA about her father's death) Sure, it's all in your mind.
LOLA: (Everyday Language)
**Dialogue**: {dialogue}'''
        
        response = get_response(SYSTEM,prompt, model_name, API_KEY)
        print("--------------------------" * 30)
        print(response)
        
        file['model response'] = response
    with open('original_model_response_gpt4o.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()