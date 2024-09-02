import streamlit as st
from annotated_text import annotated_text
import subprocess
import os
import re
import yaml
import pandas as pd
from call_pipeline import main_call
import ast
import difflib
def app(model_dict):
    # Title of the app
    st.title("Streamlit Relation Extraction Demo")
    
    
    preset_choices = ['', 
        'here are some test sentences. I need to see if this works. hello hello 3.5. help me see if this works. The responses should be a mess. I need to figure out the right amount of sentences in order to let the program run. help me with the process. these are words and multiple words form a sentence. sentences form paragraphs. I was distracted so I fell. when I hit the ground I hurt myself. when you water a plant in the future it will grow. I was sad so I listened to music. I need more samples so I create more samples. Dropping the cup made me frustrated. Pushing the domino made it fall. The wind blowing made the house of cards fall. After dieting I lost weight. I was late because the strikes happend. seeing this caused me to be sad. watching food videos made me hungry. I was thristy so I drank water.']
    
    st1_st3 = model_dict['st1'] + model_dict['st3']
    st2_st3 = model_dict['st2'] + model_dict['st3']
    
    st0_models = []
    for entry in model_dict['st0']:
        name = entry.split('/')
        type = name[-2]
        st0_models.append(type + '/' + name[-1])
    st1_models = []
    for entry in model_dict['st1']:
        name = entry.split('/')
        type = name[-2]
        st1_models.append(type + '/' + name[-1])
    st2_models = []
    for entry in model_dict['st2']:
        name = entry.split('/')
        type = name[-2]
        st2_models.append(type + '/' + name[-1])
    for entry in model_dict['st3']:
        name = entry.split('/')
        type = name[-2]
        st1_models.append(type + '/' + name[-1])
        st2_models.append(type + '/' + name[-1])
    
    # Dropdown menu
    #data_options = model_dict['data']
    #selected_data = st.selectbox("Choose a dataset:", data_names)
    selected_st0 = st.selectbox("Choose the model used for the filter:", st0_models)
    selected_st1 = st.selectbox("Choose the model used for subtask 1:", st1_models)
    selected_st2= st.selectbox("Choose the model used for subtask 2:", st2_models)
    
    preset_text = st.selectbox("Choose a preset choice:", preset_choices)
    
    
    # Textbox
    user_text = st.text_input("Enter causal sentences:")
    api = st.text_input("Enter your OpenAI api key if you are using gpt-4:")
    
   

    # Button to trigger the output
    if st.button("Submit"):
        
        fail_flag = False
        
        filter_path = ''.join([s for s in model_dict['st0'] if selected_st0 in s])
        st1_path = ''.join([s for s in st1_st3 if selected_st1 in s])
        st2_path = ''.join([s for s in st2_st3 if selected_st2 in s])
        llm_api = api
        text = user_text
        
        call_dict = {}
        #note you need to check if the user leaves the text field balnk
        
        call_dict['filter'] = filter_path
        call_dict['st1'] = st1_path
        call_dict['st2'] = st2_path
        
        if not preset_text:
            if not text:
                st.write('You have to either submit a sentence yourself or select a preset text')
                fail_flag = True
            else:
                call_dict['q'] = text
        else:
            call_dict['q'] = preset_text
            
        if not api:
            call_dict['api'] = 'None'
            if 'gpt4' in st1_path or 'gpt4' in st2_path:
                st.write('You have to put in an OpenAI key if you are going to use gpt-4')
                fail_flag = True
        else:
            call_dict['api'] = llm_api
            
        if not fail_flag:
            st.write('done')
            result = main_call(call_dict, flag=True)
            st.write(result)
            flag, inf = annotate_inf(result)
            print(inf)
            if flag:
                for i in inf:
                    annotated_text(i)
                annotated_text(
        "This ",
        ("is", "verb"),
        " some ",
        ("annotated", "adj"),
        ("text", "noun"),
        " for those of ",
        ("you", "pronoun"),
        " who ",
        ("like", "verb"),
        " this sort of ",
        ("thing", "noun"),
        "."
    )
            else:
                for i in inf:
                    st.text(i)

def annotate_inf(dict):
    annotated_list = []
    label_dict = {'cause':'#fea', 'enable':'#8ef', 'prevent':'#afa', 'intend':'#faf', 'invalid':'#faa'}
    t_list = ['subj', 'obj']
    
    use_annotate = False
    
    if 'roberta' in dict['st2_model'][0]:
        print('use case not implemented')
        df = pd.DataFrame(dict)
        for index, row in df.iterrows():
            
            sub_obj = ast.literal_eval(row['span_pred'])
            rel = row['label']
            
            if not rel in label_dict.keys():
                label = 'invalid'
            else:
                label = rel
            color = label_dict[label]
            for entry in sub_obj:
                text = entry
                
                subj_tag = '](' + label + '-subj)'
                obj_tag = '](' + label + '-obj)'
                
                subj_a = entry.find('<ARG0>')
                subj_b = entry.find('</ARG0>')
                
                if subj_a < subj_b:
                    text = re.sub('<ARG0>', '[', text)
                    text = re.sub('</ARG0>', subj_tag, text)
                else:
                    text = re.sub('</ARG0>', '[', text)
                    text = re.sub('<ARG0>', subj_tag, text)
                
                
                obj_a = entry.find('<ARG1>')
                obj_b = entry.find('</ARG1>')
                
                if obj_a < obj_b:
                    text = re.sub('<ARG1>', '[', text)
                    text = re.sub('</ARG1>', obj_tag, text)
                else:
                    text = re.sub('</ARG1>', '[', text)
                    text = re.sub('<ARG1>', obj_tag, text)
                    
                text = re.sub(r"</?[^>]+>", "", text)
                print(text)
                annotated_list.append(text)
            
            #span_list = extract_spans(sub_obj)
            #print(span_list)
            '''
            for span in span_list:
                text = row['text']
                for i, s in enumerate(span):
                    part = re.sub(s, '[' + s + ']' + label + '-' + t_list[i], text)
                    text = merge_strings(text, part)
                annotated_list.append(text)
            '''
            
    else:
        df = pd.DataFrame(dict)
        for index, row in df.iterrows():
            #you can potentially have it pull up the column name that has span_pred and use that as a variable
            sub_obj = row['span_pred']
            rel = row['label']
            if not rel in label_dict.keys():
                label = 'invalid'
            else:
                label = rel
            color = label_dict[label]
            sentence = row['text']
            loop_list = [sentence]
            
            for i in range(2):
                temp_list = []
                obj = sub_obj[i]
                for entry in loop_list:
                    if isinstance(entry,str):
                        x = annotate_sub(entry, obj, label + '-' + t_list[i], color)
                        temp_list.extend(x)
                    else:
                        temp_list.append(entry)
                loop_list = temp_list
            annotated_list.append(loop_list)
            use_annotate = True
    return use_annotate, annotated_list
                
                    
                
            


def get_choices(directory_path):
    model_dict = {}
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        model_list = []
        if filename != '.DS_Store':
            for typename in os.listdir(filepath):
                typepath = os.path.join(filepath, typename)
                if typename != '.DS_Store':
                    for modelname in os.listdir(typepath):
                        if modelname != '.DS_Store':
                            modelpath = os.path.join(typepath, modelname)
                            model_list.append(modelpath)
            model_dict[filename] = model_list
    return model_dict


def annotate_sub(sent, obj, label, color):
    annotated_list = []
    if obj.lower() in sent.lower():
        parts = re.split(re.escape(obj), sent, flags=re.IGNORECASE, maxsplit=1)
        t = (obj, label, color)
        annotated_list.append(parts[0])
        annotated_list.append(t)
        annotated_list.append(parts[1])
        return annotated_list
    else:
        annotated_list.append(sent)
        return annotated_list


def extract_substring(text, start_tag, end_tag):
    #start_tag = "<ARG1>"
    #end_tag = "</ARG1>"

    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    if start_index == -1 or end_index == -1:
        return None  # Tags not found

    return text[start_index:end_index]


def remove_tags(text):
    
    if text is None:
        return ''
    
    # Pattern to match anything between < and >
    cleaned_text = re.sub(r"</?[^>]+>", "", text)
    cleaned_text = re.sub('  ', ' ', cleaned_text)
    return cleaned_text


def merge_strings(str_a, str_b):
    # Create a SequenceMatcher object to compare the strings
    s = difflib.SequenceMatcher(None, str_a, str_b)
    
    # Initialize an empty result list
    result = []
    
    # Iterate over the matching blocks and differences
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            result.append(str_a[i1:i2])
        elif tag == 'replace':
            result.append(f"({str_b[j1:j2]})[{str_a[i1:i2]}]")
        elif tag == 'insert':
            result.append(f"({str_b[j1:j2]})")
        elif tag == 'delete':
            result.append(f"[{str_a[i1:i2]}]")
    
    # Join the list into a single string
    return ''.join(result)


def extract_spans(sents):
    span_list = []
    print(sents)
    
    if not sents:
        return 0
    for sent in sents:
        loop_list = []
        subj = extract_substring(sent, '<ARG0>', '</ARG0>')
        
        if not subj:
            subj = extract_substring(sent, '</ARG0>', '<ARG0>')
            
        obj = extract_substring(sent, '<ARG1>', '</ARG1>')
        
        if not obj:
            obj = extract_substring(sent, '</ARG1>', '<ARG1>')
        
        #print(subj)
        #print(obj)
        
        subj = remove_tags(subj)
        obj = remove_tags(obj)
        
        loop_list.append(subj)
        loop_list.append(obj)
        
        span_list.append(loop_list)
    print(len(sents))
    return span_list

if __name__ == "__main__":
    #print('hello world')
    available_llms = [
    "llm/zephyr",
    "llm/dpo",
    "llm/una",
    "llm/solar",
    "llm/gpt4"]
    directory_path = 'pretrained_models/'
    data_path = 'new_data/'
    model_dict = get_choices(directory_path)
    #model_dict['data'] =  get_tf(data_path)
    model_dict['st3'] += available_llms
    app(model_dict)
    
