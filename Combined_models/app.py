from flask import Flask, render_template, jsonify, request
from two_step_model import run_pipeline
from dotenv import load_dotenv
import subprocess
import os
import re
import yaml
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint

import configparser


def set_choices(yaml_file):
    with open(yaml_file, 'r') as file:
    # Read the entire file content
        file_content = file.read()
    s = file_content
    #print(s)
    x = s.split('name:')
    #print(x)
    final_string = x[0]
    final_string = final_string + 'name:' + x[1]
    #print(x)
    flag = False
    for sent in x:
        
        if flag == True:
            final_string = final_string + 'name:' + sent
            
            
            
        if sent[:4] == ' st0':
            options = []
            directory_path = 'pretrained_models/st0'
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isdir(file_path):
                    if filename != '.DS_Store':
                        prefix = filename + '_'
                        for modelname in os.listdir(file_path):
                            if os.path.isdir(file_path):
                                if modelname != '.DS_Store':
                                    option = prefix + modelname
                                    options.append(option)
            
            parts = re.split(r'enum:|description:', sent)
            st0_str = 'enum:\n'
            #print(parts)
            for i in options:
                st0_str = st0_str + '              - ' + i + '\n'
            #st0_str = st0_str + '              - off' + '\n'
            st0_str = st0_str + '          description:'
            final_st0 = 'name:' + parts[0] + st0_str + parts[2]
            final_string = final_string + final_st0
            
                        
            
        if sent[:4] == ' st1':
            
            st1_str = 'enum:\n'
            directory_list = ['pretrained_models/st1','pretrained_models/st3']
            directory_path = 'pretrained_models/st1'
            for directory_path in directory_list:
                options = []
                for filename in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, filename)
                    if os.path.isdir(file_path):
                        if filename != '.DS_Store':
                            prefix = filename + '_'
                            for modelname in os.listdir(file_path):
                                if os.path.isdir(file_path):
                                    if modelname != '.DS_Store':
                                        option = prefix + modelname
                                        options.append(option)
                
                parts = re.split(r'enum:|description:', sent)
                
                #print(parts)
                for i in options:
                    st1_str = st1_str + '              - ' + i + '\n'
            
            st1_str = st1_str + '              - zephyr' + '\n'
            st1_str = st1_str + '              - dpo' + '\n'
            st1_str = st1_str + '              - una' + '\n'
            st1_str = st1_str + '              - solar' + '\n'
            st1_str = st1_str + '              - gpt4' + '\n'
            st1_str = st1_str + '              - off' + '\n'
            st1_str = st1_str + '          description:'
            final_st1 = 'name:' + parts[0] + st1_str + parts[2]
            final_string = final_string + final_st1
        if sent[:4] == ' st2':
            options = []
            directory_list = ['pretrained_models/st2','pretrained_models/st3']
            #directory_path = 'pretrained_models/st2'
            st2_str = 'enum:\n'
            for directory_path in directory_list:
                options = []
                #print('here')
                #print(directory_path)
                #print('here')
                for filename in os.listdir(directory_path):
                    file_path = os.path.join(directory_path, filename)
                    if os.path.isdir(file_path):
                        if filename != '.DS_Store':
                            prefix = filename + '_'
                            for modelname in os.listdir(file_path):
                                if os.path.isdir(file_path):
                                    
                                    if modelname != '.DS_Store':
                                        option = prefix + modelname
                                        options.append(option)
                parts = re.split(r'enum:|description:', sent)
                #print(parts)
                
                #print(options)
                for i in options:
                    st2_str = st2_str + '              - ' + i + '\n'
                #print(st2_str)
            
            st2_str = st2_str + '              - zephyr' + '\n'
            st2_str = st2_str + '              - dpo' + '\n'
            st2_str = st2_str + '              - una' + '\n'
            st2_str = st2_str + '              - solar' + '\n'
            st2_str = st2_str + '              - gpt4' + '\n'
            st2_str = st2_str + '              - off' + '\n'
            st2_str = st2_str + '          description:'
            final_st2 = 'name:' + parts[0] + st2_str + parts[2]
            final_string = final_string + final_st2
            print(final_string)
        '''
        if sent[:4] == ' st3':
            options = []
            directory_path = 'pretrained_models/st3'
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isdir(file_path):
                    if filename != '.DS_Store':
                        prefix = filename + '_'
                        for modelname in os.listdir(file_path):
                            if os.path.isdir(file_path):
                                
                                if modelname != '.DS_Store':
                                    option = prefix + modelname
                                    options.append(option)
            parts = re.split(r'enum:|description:', sent)
            st3_str = 'enum:\n'
            for i in options:
                st3_str = st3_str + '              - ' + i + '\n'
            st3_str = st3_str + '              - off' + '\n'
            st3_str = st3_str + '              - zephyr' + '\n'
            st3_str = st3_str + '              - dpo' + '\n'
            st3_str = st3_str + '              - una' + '\n'
            st3_str = st3_str + '              - solar' + '\n'
            st3_str = st3_str + '              - gpt4' + '\n'
            st3_str = st3_str + '          description:'
            final_st3 = 'name:' + parts[0] + st3_str + parts[2]
            final_string = final_string + final_st3
            #flag = True
            
         '''   
        if sent[:4] == ' t_f':
            options = []
            directory_path = 'new_data/'
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    if filename != '.DS_Store':
                        option = filename
                        #print(option)
                        options.append(option)
            parts = re.split(r'enum:|description:', sent)
            #print(parts)
            tf_str = 'enum:\n'
            for i in options:
                tf_str = tf_str + '              - ' + i + '\n'
            tf_str = tf_str + '          description:'
            final_tf = 'name:' + parts[0] + tf_str + parts[2]
            final_string = final_string + final_tf
            flag = True
    #print(final_string)

    #print(final_string)
    if file_content == final_string:
        return
    else:
        print('changes have been made')
        print(final_string)
        fs = final_string
        print(fs)
        #data = yaml.load(fs,Loader=yaml.BaseLoader)
        #with open('static/swagger.yaml', 'w') as f:
        with open('static/swagger.yaml', 'w') as file:
            file.write(fs)
        #with open('output5.yaml', 'w') as f:
            #yaml.dump(data, f)
        return


def get_params():
    params = request.args
    response = {}
    
    for key in params.keys():
        value = params.get(key)
        response[key] = value
    print(response)
    print('above is the param list')
    return response


def set_config(flags):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    '''
    if flags['Do subtask 1'] == False:
        config['DEFAULT']['subtask1_flag'] = 'True'
    if flags['Do subtask 2'] == False:
        config['DEFAULT']['subtask2_flag'] = 'True'
        
    if flags['Do subtask 3'] == False:
        config['DEFAULT']['subtask3_flag'] = 'True'
    else:
        if flags['Use REBEL'] == True:
            config['DEFAULT']['rebel_flag'] = 'True'
        if flags['Use LLM'] == True:
            config['DEFAULT']['LLM_flag'] = 'True'
    if flags['User sent'] != None:
        config['DEFAULT']['text_from_user'] = flags['User sent']
    '''
    
    if 't_f' not in flags and 'q' not in flags:
        print('either choose a dataset or give your own sentences')
        return False
    if 'q' not in flags:
        config['DEFAULT']['test_file'] = 'new_data/' + flags['t_f']
    
    
    filter_s = flags['st0']
    filter_parts = filter_s.split('_')
    filter_prefix = filter_parts[0] + '_' + filter_parts[1]
    filter_name = filter_parts[2]
    for i in range(3,len(filter_parts)):
        filter_name = filter_name + '_' + filter_parts[i]
    filter_path = 'pretrained_models/st0/' + filter_prefix + '/' + filter_name
    config['DEFAULT']['filter_model_path'] = filter_path
    config['DEFAULT']['rebel_flag'] = 'False'
    config['DEFAULT']['split_st3_flag'] = 'True'
    
    if flags['st1'] != 'off':
        s = flags['st1']
        if s == 'zephyr':
            config['DEFAULT']['LLMS_llm'] = 'zephyr'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'dpo':
            config['DEFAULT']['LLMS_llm'] = 'dpo'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'una':
            config['DEFAULT']['LLMS_llm'] = 'una'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'solar':
            config['DEFAULT']['LLMS_llm'] = 'solar'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'gpt4':
            config['DEFAULT']['LLMS_llm'] = 'gpt4'
            config['DEFAULT']['LLM_flag'] = 'True'
        else:
            parts = s.split('_')
            prefix = parts[0] + '_' + parts[1]
            model_name = parts[2]
            if parts[1] != 'st3':
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st1/' + prefix + '/' + model_name
                config['DEFAULT']['st1_model_name_or_path'] = path
                config['DEFAULT']['subtask1_flag'] = 'False'
                config['DEFAULT']['st1_flag'] = 'True'
            else:
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st3/' + prefix + '/' + model_name
                config['DEFAULT']['rebel_inf_model_name_or_path'] = path
                config['DEFAULT']['rebel_st1_flag'] = 'True'
                config['DEFAULT']['rebel_flag'] = 'True'
                config['DEFAULT']['subtask1_flag'] = 'True'
                config['DEFAULT']['subtask3_flag'] = 'False'
                config['DEFAULT']['st1_flag'] = 'False'
    else:
        config['DEFAULT']['subtask1_flag'] = 'True'
    
    
    if flags['st2'] != 'off':
        s = flags['st2']
        if s == 'zephyr':
            config['DEFAULT']['LLMS_llm'] = 'zephyr'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'dpo':
            config['DEFAULT']['LLMS_llm'] = 'dpo'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'una':
            config['DEFAULT']['LLMS_llm'] = 'una'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'solar':
            config['DEFAULT']['LLMS_llm'] = 'solar'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'gpt4':
            config['DEFAULT']['LLMS_llm'] = 'gpt4'
            config['DEFAULT']['LLM_flag'] = 'True'
        else:
            parts = s.split('_')
            prefix = parts[0] + '_' + parts[1]
            model_name = parts[2]
            if parts[1] != 'st3':
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st2/' + prefix + '/' + model_name
                config['DEFAULT']['st2_pretrained_path'] = path
                config['DEFAULT']['st2_load_checkpoint_for_test'] = path + '/pytorch_model.bin'
                config['DEFAULT']['subtask2_flag'] = 'False'
                config['DEFAULT']['st2_flag'] = 'True'
            else:
                for i in range(3,len(parts)):
                    model_name = model_name + '_' + parts[i]
                path = 'pretrained_models/st3/' + prefix + '/' + model_name
                config['DEFAULT']['rebel_inf_model_name_or_path'] = path
                config['DEFAULT']['rebel_flag'] = 'True'
                config['DEFAULT']['rebel_st2_flag'] = 'True'
                config['DEFAULT']['subtask2_flag'] = 'True'
                config['DEFAULT']['subtask3_flag'] = 'False'
    else:
        config['DEFAULT']['subtask2_flag'] = 'True'
    
    if 'q' in flags:
        config['DEFAULT']['text_from_user'] = flags['q']
    if 'api' in flags:
        config['DEFAULT']['LLMS_api_key'] = flags['api']
    #if flags['t_f'] != 'None':
        #config['DEFAULT']['test_file'] = 'new_data/' + flags['t_f']
    with open('config_swagger.ini', 'w') as configfile:
        config.write(configfile)
    return True
'''   
    if flags['st3'] != 'off':
        s = flags['st3']
        config['DEFAULT']['rebel_flag'] = 'False'
        if s == 'zephyr':
            config['DEFAULT']['LLMS_llm'] = 'zephyr'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'dpo':
            config['DEFAULT']['LLMS_llm'] = 'dpo'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'una':
            config['DEFAULT']['LLMS_llm'] = 'una'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'solar':
            config['DEFAULT']['LLMS_llm'] = 'solar'
            config['DEFAULT']['LLM_flag'] = 'True'
        elif s == 'gpt4':
            config['DEFAULT']['LLMS_llm'] = 'gpt4'
            config['DEFAULT']['LLM_flag'] = 'True'
        else:
            parts = s.split('_')
            prefix = parts[0] + '_' + parts[1]
            model_name = parts[2]
            for i in range(3,len(parts)):
                model_name = model_name + '_' + parts[i]
            path = 'pretrained_models/st3/' + prefix + '/' + model_name
            config['DEFAULT']['rebel_inf_model_name_or_path'] = path
            config['DEFAULT']['rebel_flag'] = 'True'
    else:
        config['DEFAULT']['subtask3_flag'] = 'True'
'''
    


app = Flask(__name__)
api = Api(app)

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'  # URL for exposing Swagger file
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI endpoint
    API_URL,  # Swagger file URL
    config={  # Swagger UI config overrides
        'app_name': "Relation Detection API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

class RunPipeline(Resource):
    def get(self):
        # Extract query parameters with default values
        flags = get_params()
        '''
        do_subtask_1 = request.args.get('Do subtask 1', default='true') == 'true'
        do_subtask_2 = request.args.get('Do subtask 2', default='true') == 'true'
        do_subtask_3 = request.args.get('Do subtask 3', default='true') == 'true'
        use_rebel = request.args.get('Use REBEL', default='true') == 'true'
        use_llm = request.args.get('Use LLM', default='false') == 'true'
        user_sent = request.args.get('User Submitted Text', default=None)
        # Collect the flags into a dictionary
        flags = {
            'Do subtask 1': do_subtask_1,
            'Do subtask 2': do_subtask_2,
            'Do subtask 3': do_subtask_3,
            'Use REBEL': use_rebel,
            'Use LLM': use_llm,
            'User sent': user_sent
        }
        '''
        # Example response
        response = {
            'selected_flags': flags,
            'items': ['item1', 'item2', 'item3']  # Example response data
        }
        #print(flags)
        f = set_config(flags)
        if f == False:
            return jsonify('either choose a dataset or give your own sentences')
        
        json = run_pipeline('config_swagger.ini')
        
        return jsonify(json)
        #return json

api.add_resource(RunPipeline, '/RunPipeline')







args_script1 = ['python', 'two_step_model.py']
@app.route('/')
def index():
    return render_template('index.html')
    #return subprocess.run(args_script1, capture_output=True, text=True)
@app.route('/test')
def test_pipe():
    #print("Current working directory:", os.getcwd())
    #print("Cache directory:", os.getenv('HF_HOME', '~/.cache/huggingface'))
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='data/huggingface/')
    json = run_pipeline('config_swagger.ini')
    return json
if __name__ == "__main__":
    set_choices('static/swagger.yaml')
    print('done')
    app.run(port=5004, host='0.0.0.0', debug=True)
