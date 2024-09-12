from openai import OpenAI
from itertools import combinations
import os, sys
import numpy as np
import json

current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)
from utils.eval_utils import read_json_file


API_KEY = "sk-proj-aJ8bqWWKWonI0VDgh0b8T3BlbkFJWiszrlE8bNnx8krAz46h"
client = OpenAI(api_key=API_KEY)


def get_response(prompt1, prompt2):
    completion = client.chat.completions.create(
        model="gpt-4o",  # gpt-3.5-turbo, gpt-4o
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
    )
    return completion.choices[0].message.content

def json_read(path):
    with open(path, 'r') as file:
        data = file.read()

    json_objects = data.split('\n}\n{')
    json_objects[0] += '}'
    for i in range(1, len(json_objects) - 1):
        json_objects[i] = '{' + json_objects[i] + '}'
    json_objects[-1] = '{' + json_objects[-1]

    # Parse each JSON object separately
    data = []
    for obj in json_objects:
        try:
            data.append(json.loads(obj))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return data


path_a = "/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/share_0613_threeTags.json"
path_b = "/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/noshare_0613.json"
path_c = "/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/notag_0613.json"
path_d = '/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/share_0613.json'
share = read_json_file(path_a)
noshare = read_json_file(path_b)
notag = read_json_file(path_c)
print(1)
one_share = read_json_file(path_d)

print(2)
path_share = '/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/share_accum_0616.json'
path_noshare = '/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/noshare_accum_0616.json'
path_base = '/home/chanho/Model/SHARE/Refactorizing/result/prompt_data/base_50.json'
share_accum = read_json_file(path_share)
noshare_accum = read_json_file(path_noshare)
base_accum = read_json_file(path_base)
system = """You will be presented with two dialogues and a memory of previous dialogues. The memory contains information about the relationship between the two individuals as well as personal information about each individual. Based on each evaluation criterion, you need to determine which dialogue better reflects the relationship memory. You must choose one of the two dialogues as answer. Please provide a definitive answer, not "None" or "Neither."""
Example_Dialogue_A = '''CLARA: We've been down here for hours, Theo. The path splits ahead. Which way?
THEO: Right. Always right when in doubt. The map shows a chamber that way—could be what we're looking for.
CLARA: Right it is, then. I hope you're sure about this. It feels like we've been circling the same area.
THEO: I'm as sure as I can be without actual signs. Just keep an eye out for any markers. The chamber should be just beyond this tunnel.
CLARA: Wait, Theo, look at this! It's some kind of ancient carving. Does it look familiar?
THEO: Let me see... Yes, I think it's a directional marker! We're on the right track. This symbol points towards the chamber.
CLARA: That’s a relief! I was starting to think we were lost. How much further?
THEO: Not far now. According to the map, it should be just past this bend. Stay alert, though; we don’t know what’s waiting for us there.
CLARA: I just hope it's worth all this effort. Lead the way, Theo.'''

Example_Dialogue_B = '''CLARA: We've been down here for hours, Theo. The path splits ahead. Which way?
THEO: Right. Always right when in doubt. The map shows a chamber that way—could be what we're looking for.
CLARA: But what if we miss something important on the left path? We should be thorough, especially given the artifact we just found. Maybe we should double-check the map and tools before deciding.
THEO: You've got a point, Clara. Let’s take a minute. Look here, the left path leads to a narrower section that circles back to the main chamber. We might cover more ground that way without missing anything.
CLARA: Good. I'd hate to rush and overlook a clue. Let's use this chance to ensure we're fully prepared.
THEO: Right, and we've got plenty of time. No need to hurry and make a mistake. Let's make camp here tonight, recharge, and start fresh in the morning. That way, we can explore both paths thoroughly.
CLARA: That sounds like a solid plan, Theo. I'll set up the sensors and make sure we have a secure perimeter for the night. Safety first, after all.
'''

Example_Memory = '''CLARA is fascinated by historical artifacts. CLARA tends to consult maps and tools frequently. CLARA is hesitant to take unnecessary risks. CLARA values teamwork and is cooperative. CLARA is vocal about her concerns and seeks reassurance. THEO uses humor to lighten tense situations. THEO is quick to adapt to changing circumstances. THEO values personal experiences over strict plans. CLARA is worried about missing a scheduled check-in due to the deep exploration. THEO is excited about a potential historic finding based on recent cave markings they discovered. CLARA and THEO have recently discovered an ancient artifact that could be key to understanding the cave's history. CLARA and THEO are deep within an uncharted section of the cave, discussing the best route to further explore without getting lost. CLARA and THEO are setting up camp inside the cave as they prepare for an extended stay to explore deeper.'''

Engagingness_A = 0
Engagingness_B = 0
Engagingness_C = 0
Coherence_A = 0
Coherence_B = 0
Coherence_C = 0
Affinity_A = 0
Affinity_B = 0
Affinity_C = 0

# for dialogue_B, dialogue_C in zip(noshare, notag):

#     dialogue_B = dialogue_B["without_tag_dialogues"]
#     dialogue_C = dialogue_C["without_tag_dialogues"]
#     print(dialogue_B)
#     print(dialogue_C)
#     prompt = f"""Coherence: A metric at the dialogue level, evaluating if the response is pertinent and consistent with the given context.
# Engagingness: A metric at the dialogue level, determining whether the annotator finds the speaker interesting enough to sustain a long-term conversation.
# Affinity: A dialogue-level metric that assesses the quality of the relationship between the two interlocutors. It measures how well the participants connect and how their interactions reflect this familiarity.

# Dialogue C: {dialogue_C}
# Dialogue B: {dialogue_B}

# The results must follow the format below:

# Result format:
# Engagingness: |better_dialogue|
# Coherence: |better_dialogue|
# Affinity: |better_dialogue|
# """

#     result = get_response(system, prompt).split("\n")
#     Engagingness = result[0].split(":")[1]
#     Coherence = result[1].split(":")[1]
#     Affinity = result[2].split(":")[1]

#     print(
#         f"result -> Engagingness : {Engagingness}, Coherence : {Coherence}, Affinity : {Affinity}"
#     )

#     if "B" in Engagingness:
#         Engagingness_B += 1
#     elif "C" in Engagingness:
#         Engagingness_C += 1

#     if "B" in Coherence:
#         Coherence_B += 1
#     elif "C" in Coherence:
#         Coherence_C += 1

#     if "B" in Affinity:
#         Affinity_B += 1
#     elif "C" in Affinity:
#         Affinity_C += 1



# print(
#     f"B Engagingness : {Engagingness_B/50} Coherence : {Coherence_B/50} Affinity : {Affinity_B/50}"
# )
# print(
#     f"C Engagingness : {Engagingness_C/50} Coherence : {Coherence_C/50} Affinity : {Affinity_C/50}"
# )

# Engagingness_A = 0
# Engagingness_B = 0
# Engagingness_C = 0
# Coherence_A = 0
# Coherence_B = 0
# Coherence_C = 0
# Affinity_A = 0
# Affinity_B = 0
# Affinity_C = 0


# for dialogue_A, dialogue_B in zip(share, noshare):
#     dialogue_A = dialogue_A['without_tag_dialogues']
#     dialogue_B = dialogue_B['without_tag_dialogues']
#     prompt = f"""Coherence: A metric at the dialogue level, evaluating if the response is pertinent and consistent with the given context.
# Engagingness: A metric at the dialogue level, determining whether the annotator finds the speaker interesting enough to sustain a long-term conversation.
# Affinity: A dialogue-level metric that assesses the quality of the relationship between the two interlocutors. It measures how well the participants connect and how their interactions reflect this familiarity.

# Dialogue B: {dialogue_B}
# Dialogue A: {dialogue_A}

# The results must follow the format below:

# Result format:
# Engagingness: |better_dialogue|
# Coherence: |better_dialogue|
# Affinity: |better_dialogue|
# """
#     result = get_response(system, prompt).split("\n")
#     Engagingness = result[0].split(":")[1]
#     Coherence = result[1].split(":")[1]
#     Affinity = result[2].split(":")[1]

#     print(f"result Engagingness : {Engagingness}, Coherence : {Coherence}, Affinity : {Affinity}")

#     if 'A' in Engagingness:
#         Engagingness_A += 1
#     elif 'B' in Engagingness:
#         Engagingness_B += 1

#     if 'A' in Coherence:
#         Coherence_A += 1
#     elif 'B' in Coherence:
#         Coherence_B += 1

#     if 'A' in Affinity:
#         Affinity_A += 1
#     elif 'B' in Affinity:
#         Affinity_B += 1

# print(
#     f"A Engagingness : {Engagingness_A/50} Coherence : {Coherence_A/50} Affinity : {Affinity_A/50}"
# )
# print(
#     f"B Engagingness : {Engagingness_B/50} Coherence : {Coherence_B/50} Affinity : {Affinity_B/50}"
# )


Engagingness_A = 0
Engagingness_B = 0
Engagingness_C = 0
Coherence_A = 0
Coherence_B = 0
Coherence_C = 0
Reflectiveness_A = 0
Reflectiveness_B = 0
Reflectiveness_C = 0


for dialogue_A, dialogue_C in zip(noshare_accum, base_accum):
    Memory = ' '.join(dialogue_A['p1_persona'] + dialogue_A['p2_persona'] + dialogue_A['p1_temp'] + dialogue_A['p2_temp'] + dialogue_A['share'])
    dialogue_A = dialogue_A["without_tag_dialogues"]
    dialogue_C = dialogue_C["without_tag_dialogues"]
    prompt = f"""    
Coherence: A dialogue-level metric assessing the consistency and logical connection of responses within a session.
Engagingness: A dialogue-level metric determining the speaker's ability to maintain the annotator's interest for a long-term conversation.
Reflectiveness: A dialogue-level metric evaluating how well the dialogue reflects the relationship indicated in the Memory.

Memory: {Memory}
Dialogue_A: {dialogue_C}
Dialogue_B: {dialogue_A}


The results must follow the format below.

Result format:
Engagingness: |better_dialogue|
Coherence: |better_dialogue|
Reflectiveness: |better_dialogue|
"""
    
# Example :
# Dialogue_A : {Example_Dialogue_A}
# Dialogue_B : {Example_Dialogue_B}
# Memory : {Example_Memory}

# Result format:
# Engagingness: |B|
# Coherence: |A|
# Reflectiveness: |B|    
    

    result = get_response(system, prompt).split("\n")

    Engagingness = result[0].split(":")[1]
    Coherence = result[1].split(":")[1]
    Reflectiveness = result[2].split(":")[1]

    print(
        f"result Engagingness : {Engagingness}, Coherence : {Coherence}, Reflectiveness : {Reflectiveness}"
    )

    if "A" in Engagingness:
         Engagingness_A += 1
    elif "B" in Engagingness:
         Engagingness_C += 1

    if "A" in Coherence:
         Coherence_A += 1
    elif "B" in Coherence:
         Coherence_C += 1

    if "A" in Reflectiveness:
         Reflectiveness_A += 1
    elif "B" in Reflectiveness:
         Reflectiveness_C += 1

print(
    f"A Engagingness : {Engagingness_A/50} Coherence : {Coherence_A/50} Reflectiveness : {Reflectiveness_A/50}"
)
print(
    f"C Engagingness : {Engagingness_C/50} Coherence : {Coherence_C/50} Reflectiveness : {Reflectiveness_C/50}"
)
