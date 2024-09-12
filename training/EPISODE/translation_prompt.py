import os, sys, json
from openai import OpenAI
current_dir = os.getcwd()
episode_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(episode_dir)

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
    path = '/home/chanho/Model/SHARE/Refactorizing/training/EPISODE/final_list_dataset_v3.json'
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    model_name = 'gpt-4o' # gpt-4o-mini

    API_KEY = 

    json_list = []
    text_list = []
    memory_list = []

###    data = {key:val for key,val in data.items() if key == "('BEN', 'MRS. ROBINSON')"}


    for key,val in data.items():
        for session in val['dialogue']:
            text_list2 = []
            dialogue = session['dialogues']
            text = make_dialogue(dialogue)
            text_list2.append(key)
            text_list2.append(text)
            text_list.append(text_list2)
            keys = list(session.keys())
            memory = ''
            for key_ in keys[2:]:
                memory += f'{session[key_]}###'
            memory_list2 = []
            memory_list2.append(key)
            memory_list2.append(memory)
            memory_list.append(memory_list2)
    try:
        for text_, memory_ in zip(text_list, memory_list):
            text = text_[1]
            memory = memory_[1]
            name = text_[0]

            SYSTEM = "You will be provided with two characters' English movie lines and the characteristics of each line. Your task is to translate the English dialogue into Korean and also translate the characteristics of each line into Korean. When translating the dialogue, make sure to preserve the unique characteristics of the Korean language. Below are the guidelines (instructions) for translation."
            prompt = f"""
The entire conversation should be translated so that it flows naturally and is easy to read, in a way that two Koreans would speak to each other.
<Instruction>
1. Convert to Colloquial Language
When translating English dialogue into Korean, avoid formal or written language. Translate using colloquial speech.
<Example>
EN: I will always be there for you.
KO: 내가 항상 네 곁에 있을게.
2. Use Appropriate Honorifics
In Korean, depending on the situation, you need to choose between using honorifics or informal speech. Read the situation carefully and choose the appropriate style.
<Example>
EN: What do you think you're doing?
Informal: 너 뭐하는 거야?
Honorific: 지금 뭐하시는 겁니까?
3. Choose Words That Match Korean Sensibilities
Korean language tends to be very expressive when it comes to emotions. Use rich, emotionally expressive words commonly used by Koreans. Also, be mindful of 'Konglish' (Korean-style English); use Konglish when it’s frequently used, and replace unnecessary English with Korean expressions. It would be great if you could use appropriate Korean expressions based on the situation. Also, if possible, avoid omitting particles and try to write with specific expressions.
<Example>
EN: I’m fine.
KO: 괜찮아... 진짜 괜찮아.

EN: You didn’t load it in, did you?
KO: 너 이거 안 넣었지?
4. Swear Words and Slang
In Korean films, characters sometimes use mild swear words or slang, especially in relationships between gangsters or close friends. Avoid using extreme swear words and I’d appreciate it if you could avoid using language that might make Koreans feel uncomfortable as much as possible..

<Example>
EN: You are so annoying!
KO: 너 진짜 짱나게 한다. Or 너 정말 성가시다.

Characteristics fall under one of six categories:
1. speaker1's persona
2. speaker2's persona
3. speaker1's temporary event
4. speaker2's temporary event
5. Shared memory
6. Mutual event

You just need to provide a clean and accurate translation for the characteristics. Each line may have one or more characteristics, and if there are multiple, separate them with commas.
You will be provided with data in the following format. If the characteristic "Everyday Language" appears in English, translate it as '일상언어'.If there is no characteristic, return [] in that place. The distinction between characteristics must be separated by '||' unconditionally. It must strictly follow the structure outlined below.
Characteristics: ['speaker1's persona1', 'speaker1's persona2']###['speaker2's persona1','speaker2's persona2']###['speaker1's temporary event1','speaker1's temporary event2']###['speaker2's temporary event1','speaker2's temporary event2']###['Shared memory1','Shared memory2']###['Mutual event1','Mutual event2']
English dialogue: speaker1-#-'text1'-#-['characteristics1','characteristics2']###speaker2-#-'text2'-#-['characteristics1','characteristics2','characteristics3']###speaker1-#-'text3'-#-['characteristics1','characteristics2']###speaker2-#-’text4’-#-['characteristics1','characteristics2']

The output format should be as follows. In the case of receiving multiple features, the features should all be separated by ||. Also, Translate the English name to match the Korean pronunciation.
-특징: ['화자1의 페르소나1'||'화자1의 페르소나2']###['화자2의 페르소나1'||'화자2의 페르소나2']###['화자1의 일시적 특성1'||'화자1의 일시적 특성2']###['화자2의 일시적 특성1'||'화자2의 일시적 특성2']###['공유된 기억1'||'공유된 기억2']###['상호 기억1'||'상호 기억2']
-한국어 대화: 발화자1-#-'대사1'-#-['특성1'||'특성2']###발화자2-#-'대사2'-#-['특성1'||’특성2’]###발화자1-#-'대사3'-#-['특성1'||'특성2']###발화자2-#-'대사4'-#-['특성1'||'특성2']

<Example>
Characteristics: ['KIT is a diligent student with a 4.0 GPA.','KIT is unsure about going to law school as expected by her family.','KIT is scared about her future.','KIT has strong feelings for ROB.']###['ROB is less academically inclined as he finds KIT's investment in him to be crazy.','ROB is aware of their differences.','ROB is concerned about the future of their relationship.']###['KIT is dealing with the pressure of expectations from her family and her uncertainty about going to law school.']###['ROB is worried about the future of his relationship with KIT and if her feelings will remain the same in the coming years.']###[]###['The mutual event between KIT and ROB is their discussion about their future and the state of their relationship.']
English dialogue:  KIT-#-"Rob? What's wrong?"-#-['Everyday Language']###ROB-#-"I swear it wasn't this cold yesterday."-#-['Everyday Language']###KIT-#-"Are you giving up?"-#-['Everyday Language']###ROB-#-"What about you? What about your future? You're the one with the four point. ...maybe it's crazy--you investing so much energy in me."-#-['ROB is less academically inclined as he finds KIT's investment in him to be crazy']###KIT-#-"I don't know. I'm scared. I don't know if I want what Mom and everybody else expects of me. I don't want to go to law school."-#-['KIT is scared about her future','KIT is unsure about going to law school as expected by her family']###ROB-#-"You've got a great mind."-#-['Everyday Language']###KIT-#-"...only it's not made up."-#-['KIT is unsure about her future']###ROB-#-"Great... Do you realize how perfectly unmatched we are?"-#-['ROB is aware of their differences']###KIT-#-"It's made up about one thing though."-#-['Everyday Language']###ROB-#-"Yeah? What's that? Will you feel the same about me a year from now? Two years, five years from now?"-#-['ROB is concerned about the future of their relationship','ROB is worried about the future of his relationship with KIT and if her feelings will remain the same in the coming years']

-특징: ['킷은 4.0 학점을 가진 부지런한 학생이다'||'킷은 가족이 기대하는 대로 로스쿨에 가는 것에 대해 확신이 없다'||'킷은 자신의 미래가 두렵다'||'킷은 롭에게 강한 감정을 가지고 있다']###['롭은 학문적으로 덜 성향이 있으며 킷이 자신에게 너무 많은 에너지를 쏟는 것을 미쳤다고 생각한다'||'롭은 그들의 차이를 인식하고 있다'||'롭은 그들의 관계의 미래에 대해 걱정하고 있다']###['킷은 가족의 기대와 로스쿨에 대한 불확실성으로 압박을 받고 있다']###['롭은 킷과의 관계가 앞으로도 유지될지 걱정하고 있다']###[]###['킷과 롭 간의 상호 사건은 그들의 미래와 관계 상태에 대한 논의이다']
-한국어 대화: 킷-#-"롭? 왜 그래?"-#-['일상언어']###롭-#-"어제 이렇게 춥지 않았던 것 같은데."-#-['일상언어']###킷-#-"너 포기하는 거야?"-#-['일상언어']###롭-#-"넌 어때? 네 미래는? 넌 학점이 4.0이잖아. ...솔직히 말해서, 네가 나한테 이렇게까지 신경 쓰는 게 좀 미친 거 같아."-#-['롭은 학문적으로 덜 성향이 있으며 킷이 자신에게 너무 많은 에너지를 쏟는 것을 미쳤다고 생각한다']###킷-#-"나도 모르겠어. 나 무서워. 엄마랑 다른 사람들이 기대하는 게 내가 원하는 건지 모르겠어. 로스쿨 가기 싫어."-#-['킷은 자신의 미래가 두렵다'||'킷은 가족이 기대하는 대로 로스쿨에 가는 것에 대해 확신이 없다']###롭-#-"너 머리 진짜 좋잖아."-#-['일상언어']###킷-#-"...근데 확실하지가 않아."-#-['킷은 자신의 미래에 대해 확신이 없다']###롭-#-"대단하다... 우리 진짜 안 맞는다는 거 느껴지지 않아?"-#-['롭은 그들의 차이를 인식하고 있다']###킷-#-"근데 한 가지는 확실해."-#-['일상언어']###롭-#-"뭔데? 앞으로 1년, 2년, 5년 뒤에도 나한테 똑같이 느낄 수 있을까?"-#-['롭은 그들의 관계의 미래에 대해 걱정하고 있다'||'롭은 킷과의 관계가 앞으로도 유지될지 걱정하고 있다']    

Now initiate the translation based the current conversation. When reading a conversation, it shouldn't feel awkward, and especially for a Korean reader, it should feel natural and enjoyable.

Characteristics: {memory}
English dialogue: {text}
"""        
            response = get_response(SYSTEM, prompt, model_name, API_KEY)
            print("--------------------------" * 30)
            print(response)
            file = dict()
            file[name] = response
            json_list.append(file)
    except:
        with open('translation_v1.json', "w", encoding="utf-8") as file:
            json.dump(json_list, file, ensure_ascii=False, indent=4)
    with open('translation_final_v2.json', "w", encoding="utf-8") as file:
        json.dump(json_list, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()