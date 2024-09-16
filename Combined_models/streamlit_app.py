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
        'He was hungry which made him angry. He worked hard with the intention of  getting a good grade. She brought water in order for her to not be dehydrated on her hike. I will give you the keys to the studio so that you can record', 
        
        'Bad inferences. Lalu , Rabri upbeat after success of shutdown 29th January 2010 01:40 PM An RJD activist wears a garland and crown made of vegetables and shouts slogans along with others during a protest against inflation in Patna . Protests were held across Andhra Pradesh criticising police action on Naidu and his supporters . Denied Aid , Dalit Boy tries to End Life. So we are asking people to come out because it may be the last time that we are going to have a peaceful and lawful protest in Hong Kong , ” said one of the organisers of the rally.  Mining for trouble Sino Gold Mining , which only last week announced a joint venture to expand exploration near its White Mountain Mine in Jilin province , had to halt operations yesterday as protesting farmers blocked the main access road .',
        
        'Good inferences. At Balagangamanahalli panchayat in Dharmapuri , residents of Eechampatti village laid siege to the Nallampalli BDO s office in protest against non supply of water . 15th September 2015 05:49 AM THIRUVANANTHAPURAM : With the government appearing to be in no mood to meet the demand of the doctors of the health service , the Kerala Government Medical Officers Association spearheading the hunger strike in front of the state secretariat has called for intensifying the agitation in the coming days . Subcontractors  will  be offered a settlement and a swift transition to new management  is expected  to avert an exodus of skilled workers from Waertsilae Marines two big shipyards, government officials said. As part of the year-long partnership, Oral-B and Scientific American Custom Media are releasing a series of content, including educational resources from leading medical and dental researchers that will help readers better understand the connections between oral health and whole body health. NE Youth s Death Sparks Protest 01st February 2014 09:23 AM Nido Taniam , son of Arunachal Pradesh Congress legislator Nido Pavitra died on Thursday allegedly after being beaten up at a market area in Lajpar Nagar here .'
        ]
    
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
    selected_st0 = st.selectbox("Choose the model used to filter out sentences that have no relationships between their words:", st0_models)
    selected_st1 = st.selectbox("Choose the model used to classify the relation in each sentence:", st1_models)
    selected_st2= st.selectbox("Choose the model used to extract the spans of words that have a relationship to each other:", st2_models)
    
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
            result = main_call(call_dict, flag=True)
            if isinstance(result, str):
                st.write(result)
            else:
                flag, inf = annotate_inf(result)
                print(inf)
                if flag:
                    for i in inf:
                        annotated_text(i)
                else:
                    for i in inf:
                        annotated_text(i)
                        #st.text(i)

def annotate_inf(dict):
    annotated_list = []
    test_list = []
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
                rel = 'other(' + rel + ')'
            else:
                label = rel
            color = label_dict[label]
            for entry in sub_obj:
                text = entry
                
                #subj_tag = '](' + label + '-subj)'
                #obj_tag = '](' + label + '-obj)'
                subj_tag = ']'
                obj_tag = ']'
                
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
                
                subj_index = text.find(subj_tag)
                obj_index = text.find(obj_tag)
                
                if subj_index > obj_index:
                    tag_order = ['obj', 'subj']
                else:
                    tag_order = ['subj', 'obj']
                
                #print(text)
                
                subj_tag2 = rel + '-subj'
                obj_tag2 = rel + '-obj'
                
                tag_list = {'subj':subj_tag2, 'obj':obj_tag2}
                h = highlight_sent(text, tag_order, tag_list)
                test_list.append(h)
                #annotated_list.append(text)
                annotated_list.append(h)
            #print(test_list)
            
            #span_list = extract_spans(sub_obj)
            #print(span_list)
            
            
    else:
        df = pd.DataFrame(dict)
        for index, row in df.iterrows():
            #you can potentially have it pull up the column name that has span_pred and use that as a variable
            sub_obj = row['span_pred']
            rel = row['label']
            if not rel in label_dict.keys():
                label = 'invalid'
                rel = 'other(' + rel + ')'
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
                        x = annotate_sub(entry, obj, rel + '-' + t_list[i], color)
                        #x = annotate_sub(entry, obj, label + '-' + t_list[i], color)
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



def highlight_sent(sent, tag_order, tag_list):
    color_list = {'subj':'#fea', 'obj':'#8ef'}
    
    tag = tag_list[tag_order[1]]
    
    bracket_list = ['[', ']']
    i = 0
    #print('this is before the highlight test')
    #first_split = re.split(r'[\[\]]', sent, 1)
    split1 = sent.split('[', 1)
    split2 = split1[1].split(']', 1)
    #split2 = split1[1].split(']', 1)
    first_split = [split1[0], split2[0], split2[1]]
    #print(sent)
    #print(first_split)
    first_split_list = [first_split[0], ('[', '', color_list[tag_order[0]]), first_split[1], (']', tag_list[tag_order[0]], color_list[tag_order[0]]), first_split[2]]
    split_list = first_split_list
    #print(first_split_list)
    
    for bracket in bracket_list:
        loop_list = []
        for entry in split_list:
            if isinstance(entry, str):
                loop_part = subsplit_text(entry, color_list[tag_order[1]], bracket, tag)
                loop_list.extend(loop_part)
            else:
                #print(entry)
                loop_list.append(entry)
        split_list = loop_list
        
    #print(split_list)
    #print('highlight_sent ends here')
    return(split_list)

def subsplit_text(part, color, s, tag):
    sub_list = []
    note = ''
    if s == ']':
        note = tag
    if s in part:
        #print('called')
        parts = part.split(s)
        sub_list.append(parts[0])
        x = (s, note, color)
        sub_list.append(x)
        sub_list.append(parts[1])
    else:
        sub_list.append(part)
    return sub_list


def extract_spans(sents):
    span_list = []
    #print(sents)
    
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
    #print(len(sents))
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
    
