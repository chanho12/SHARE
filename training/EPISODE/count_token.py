import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")

previous = '''    [
        [
            "Sue is an Asian woman, likely Hmong, dating a man named Trey.",
            "Sue is part of the Hmong community involved in the Vietnam war and moved to the Midwest USA, brought over by Lutherans.",
            "Sue has a brother named Tao who is struggling to find his direction.",
            "Sue has a keen sense of humor and sharp wit.",
            "Sue does not back down when Walt displays his bigotry."
        ]
'''
current = '''        [
            "SUE is social.",
            "SUE likes to host barbecues.",
            "SUE has a sense of humor."
        ],'''

SYSTEM = f'You will be provided with two memories related to a persona. Please read, understand, and remember them well. Then, proceed to update them for memory management.'        
prompt = f'''When updating, try to retain as much of the original expression as possible.
Persona memory typically describes personal characteristics, preferences, age, favorite foods, music, hobbies, family, daily life, health conditions, and other related information about the individual. The instruction can be selected multiple times.
    <Instruction>
    1. Updating Similar Personas: If there are expressions in the persona from the previous session and the current session that reveal similar characteristics, merge them. Example: Previous memory: ["John is a diligent worker who always completes tasks on time."]\nCurrent memory: ["John is known for his punctuality and dedication to meeting deadlines."]\nUpdated memory: ["John is a diligent worker known for his punctuality and dedication to completing tasks on time."]
    2. Updating Opposite Personas: If there are expressions in the persona from the previous session and the current session that reveal opposing characteristics, express the change. Example: Previous memory: ["Susan used to be shy and avoided speaking in public."]\nCurrent memory: ["Susan has become more confident and now enjoys public speaking."]\nUpdated memory: ["Susan has transitioned from being shy and avoiding public speaking to becoming more confident and enjoying it."]
    3. Removing Unnecessary Persona-Related Information: Remove any statements related to the persona. 
    4. Accumulating Unrelated Persona: Accumulate any personas that do not fall under instructions 1 to 2.

    You must follow the updated format below for the update and provide the instruction number you followed.
    Updated format: [memories]
    Instruction number: [instruction number]
    Previous memory: {previous}
    Current memory: {current}
    Updated memory:
    Instruction number:'''

text = SYSTEM + prompt
tokens = encoding.encode(text)
print(f"Token 개수: {len(tokens)}")