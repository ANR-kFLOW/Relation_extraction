import os
import pandas as pd
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"  # Specify the GPU id
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface/"
# device = "cuda"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B")
model = AutoModelForCausalLM.from_pretrained("yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B", device_map='sequential' ,load_in_8bit=True, use_safetensors=True)

prompt = ''''I am seeking data augmentation for sentences that contains one condition and one event with enable relationship between them, where each sentence comprises the condition and the event in which one is enabling the other to happen.
  Below are the definitions for the tags in the sentence:
 condition: a condition is The fact of having certain qualities, which may trigger events , to be enclosed between <ARG0> and </ARG0>.
 Event2: The event that is enabled to happen, to be enclosed between <ARG1> and </ARG1>.
 Signal: Words that transition the condition  "condition" to the enabled event event2, to be enclosed between <SIG0> and </SIG0>. The signal is not always present.
generate examples with enable relations following the same scheme as the following examples. sentences should describe common sense. do not generate more than the requested sentences, no definition needed'''

prompt = ''''
I am seeking data augmentation for causal sentences, where each sentence comprises two events in which one is preventing the other from happening.
  Below are the definitions for the tags in the sentence:
 event1: the event that prevent the next event from happening, , to be enclosed between <ARG0> and </ARG0>.
 Event2: The event that is prevented from happening, to be enclosed between <ARG1> and </ARG1>.
 Signal: Words that transition the preventing event event1 to the prevented event event2, to be enclosed between <SIG0> and </SIG0>.
generate examples with prevent relations following the same scheme as the following examples. sentences should describe common sense. do not generate more than the requested sentences, no definition needed'''


request = '''generate examples with prevent relations follwing the same scheme as the previous examples. sentences shoudl describe common sense.'''
df_explicit = pd.read_csv('clean.csv')
#df_implit= pd.read_csv('V_V_others.csv')
#df_explicit['Combined']=df_explicit['Combined_Column']
df=pd.DataFrame()
while 1:
    random_examples_explicit = df_explicit.sample(n=2)
#    random_examples_implit = df_implit.sample(n=2)
#    combined_df = pd.concat([random_examples_explicit, random_examples_implit])

    # random_selection = df.sample(n=5)
    # print('len df', len(df))

    # Storing selected sentences in a list
    combined_text = ""
    for i, text in enumerate(random_examples_explicit['text'], start=1):
        combined_text += f"{i}. {text}\n"



    examples =combined_text
    # Printing selected sentences as one text block with bullet points

    template = f'''<|system|> generate examples with enable relations following the same described scheme . sentences should describe common sense situations

    {prompt + examples }</s>
    <|assistant|>
    '''
    # print(llm(template))
    # inputs = tokenizer(template, return_tensors="pt")
    inputs = tokenizer([template], return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    result=tokenizer.decode(outputs[0], skip_special_tokens=True)


    print(result)
    sentences = re.split(r'\n\s?\d+\.\s*', result)[1:]
    new_sentences_df = pd.DataFrame({'Sentences': sentences})
    df = pd.concat([df, new_sentences_df], ignore_index=True)

    # df = df.append(pd.DataFrame({'Sentences': sentences}), ignore_index=True)

    df.to_csv('prevention_new.csv', index=False)
