from openai import OpenAI
import os, random, re, json
import argparse

from check_memory import load_session,load_data, save_json, make_dialogues, get_random_items_from_dict


def parse_args():
    parser = argparse.ArgumentParser(description="take Model")
    parser.add_argument("--api_key", type=str, help="GPT api_key")
    parser.add_argument("--accumulate", type=str, help="accumulate")
	parser.add_argument("--update", type=str, help="update")
    parser.add_argument('--model_name', type= str, help = 'gpt4o,gpt3.5 turbo')
    parser.add_argument("--result", type=str, help = 'result')
	parser.add_arguemtn("--criterion", type=str, help='criterion')
    args = parser.parse_args()
    return args


def read_json_file(file_path):

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in line: {line.strip()}")
                    print(f"Error message: {e}")
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
)

def get_gpt_response(prompt):
	try:
		response = client.chat.completions.create(
			model="gpt-4o",
			messages=[
				{"role": "user", "content": f"{prompt}"},
			],
			temperature=1,
			max_tokens=1000,
			top_p=1,
			frequency_penalty=0,
			presence_penalty=0,
		)

	except Exception as e:
		error_str = str(e)
		print("error:", error_str)
		return None

	# 모델의 텍스트 응답 추출
	if response.choices:
		model_response = response.choices[0].message.content
		return model_response
	else:
		return None
    

def return_prompt(criterion ,dialogue, response1, response2):
    if criterion == 'Memorability':
		prompt = f"""Your task is to choose the better response based on one metric with a brief explanation. 

You will be given a list of past memories and a current conversation between two individuals. You will then be given two response options for the next turn in the current conversation. Based on the past memories, choose one response under the following criteria. 

Evaluation Criteria: 
Memorability - The response should properly recall past memories when needed. Higher reflection of past memories indicates higher Memorability. 
The output format should be as follows: 
- Explanation: (a brief explanation with evidence or reasons) 
- Better Response: [Response #] Now choose the response that has better Memorability given the past memories and current conversation. 
- Past Memories: {past_memories} 
- Current Conversation: {current_conversation} 
- Response Options: 
[Response 1] : {response1} 
[Response 2] : {response2} 
- Explanation:"""
		
    elif criterion == 'Specificity':
		prompt = f"""Your task is to choose the better response based on one metric with a brief explanation. 

You will be given a conversation between two individuals. You will then be given two response options for the next turn in the conversation. 

Choose a better response under the following criteria. 
Evaluation Criteria: 
Specificity - The response should be detailed and precise, providing clear and specific information relevant to the conversation. 

The output format should be as follows: 
- Explanation: (a brief explanation with evidence or reasons) 
- Better Response: [Response #] Now choose the response that is more specific given the current conversation. 
- Conversation: {conversation} 
- Response Options: 
[Response 1] {speaker}: {response1} 
[Response 2] {speaker}: {response2} 
- Explanation:"""
		
    elif criterion == 'Consistency':
		
		prompt = f'''Your task is to choose the better response based on one metric with a brief explanation. 

You will be given a list of past memories and a current conversation between two individuals. You will then be given two response options for the next turn in the current conversation. 

Based on the past memories, choose one response under the following criteria. 

Evaluation Criteria: 
Consistency - The response should not contain information that is contradictory to the past memory. 
The output format should be as follows: 
- Explanation: (a brief explanation with evidence or reasons) 
- Better Response: [Response #] Now choose the response that has better Consistency given the past memories and current conversation. 
- Past Memories: {past_memories} 
- Current Conversation: {current_conversation} - Response Options: 
[Response 1] {speaker}: {response1} 
[Response 2] {speaker}: {response2} 

- Explanation:'''
		
    else:
		print("You write down wrong criterion.")
	
    return prompt


def main():
	
    args = parse_args()
    data_path = args.data_path
    
    API_KEY = args.api_key
    client = OpenAI(api_key=API_KEY)
	accumulate = args.accumulate 
	update = args.update
	answer_file = "/home/chanho/Model/SHARE/Refactorizing/training/EPISODE/test_list_v3.json"
	save_data = read_json_file(accumulate)
	save_data2 = read_json_file(update)
	answer_data = load_data(answer_file)

	save_data = get_random_items_from_dict(save_data, 100)
	chosen_responses = {}

	for key, value in save_data.items():
		
		_, dialogues = make_dialogues(answer_data[key]['dialogue'][1])
		response1 = value['model response']
		response2 = save_data2[key]['model response']

		responses = [
			{'response': response1, 'source': 'update'},
			{'response': response2, 'source': 'accumulate'}
    	]
		random.shuffle(responses)
		for i, resp in enumerate(responses, 1):
			print(f"{i}: {resp['response']}")
		
		prompt = return_prompt(dialogues, responses[0]['response'], responses[1]['response'])
		print('-'*100)
		print(prompt)
		print("*"*100)
		gpt_answer = get_gpt_response(prompt)
		print(gpt_answer)

		while True:
			choice = parse_response(gpt_answer)
			print(choice)
			if choice == 'A':
				num = 0
				break
			elif choice == 'B':
				num = 1
				break
			elif choice == 'tie':
				responses.append({'source': 'tie'})
				num = 2
				break
			else:
				print("잘못된 선택입니다. 다시 입력하세요.")
				gpt_answer = get_gpt_response(prompt)
		
		chosen_responses[key] = responses[num]
		print(chosen_responses[key])

	
	save_json(chosen_responses, 'chosen_responses_session4.json')
	print("선택된 응답이 'chosen_responses.json' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
