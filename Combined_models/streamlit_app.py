import streamlit as st
import subprocess
import os
import re
import yaml
from call_pipeline import main_call
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
            #st.write(result)

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
    
