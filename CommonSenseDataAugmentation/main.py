# Your main script

import re
import pandas as pd
# from ctransformers import AutoModelForCausalLM, AutoConfig, Config
#
# conf = AutoConfig(Config(temperature=0.7, repetition_penalty=1.1, batch_size=52,
#                          max_new_tokens=1024, context_length=2048))
# llm = AutoModelForCausalLM.from_pretrained("zephyr-7b-alpha.Q4_K_M.gguf",
#                                            model_type="mistral", config=conf)


def read_prompt_template(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    prompt_template = {}
    for section in re.split(r'\[\w+\]', content)[1:]:
        key, value = section.split('\n', 1)
        prompt_template[key.strip()] = value.strip()

    return prompt_template


prompt_template = read_prompt_template("prompt_template.txt")
#
# while 1:
#     df = pd.read_csv('prevents.csv')
#     random_selection = df.sample(n=5)
#     print('len df', len(df))
#
#     selected_sentences = [f"- {row['Sentences']}" for _, row in random_selection.iterrows()]
#     examples = "\n".join(selected_sentences)
#
#     template = f'''generate examples with prevent relations following the same described scheme. sentences should describe common sense situations
#
#     {prompt_template["Prompt"] + examples}</s>
#
#     '''
#
#     result = llm(template)
#     print(result)
#     sentences = re.split(r'\n\s?\d+\.\s*', result)[1:]
#     new_sentences_df = pd.DataFrame({'Sentences': sentences})
#     df = pd.concat([df, new_sentences_df], ignore_index=True)
#
#     df.to_csv('prevents.csv', index=False)
