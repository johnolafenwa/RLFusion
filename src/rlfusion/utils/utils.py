import re

def get_boxed_answer(text: str):
    
    boxed_answers = re.findall(r"\[(.*?)\]", text)

    if len(boxed_answers) == 0:
        return None
    else:
        return boxed_answers[-1]
    