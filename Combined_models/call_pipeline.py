import argparse
import configparser
import os
from two_step_model import run_pipeline
import pandas as pd
from datetime import datetime

available_llms = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4"  # Added GPT-4
}

def build_preset_label(t_f, filter, st1, st2, preset_labels):
    presets_dict = {}
    
    tf = t_f.split('/')[-1]
    filter_name = filter.split('/')[-1]
    
    filter_preset = 'out/tf-' + tf + '-filter-' + preset_labels['filter'] + filter_name + '.csv'
    filter_label = 'tf-' + tf + '-filter-' + preset_labels['filter'] + filter_name
    
    presets_dict['st0_preset'] = filter_preset
    
    if st1 != 'None':
        st1_name = st1.split('/')[-1]
        st1_preset = 'out/' + filter_label + '-st1-' + preset_labels['st1'] + st1_name + '.csv'
        presets_dict['st1_preset'] = st1_preset
        
    if st2 != 'None':
        st2_name = st2.split('/')[-1]
        st2_preset = 'out/' + filter_label + '-st2-' + preset_labels['st2'] + st2_name + '.csv'
        presets_dict['st2_preset'] = st2_preset
        
    return presets_dict

def create_config(build_dict):
    llms = ["zephyr", "dpo", "una", "solar", "gpt4"]
    config = configparser.ConfigParser()
    config['TEMP'] = {}
    preset_labels = {}
    ut = False
    if '.csv' in build_dict['text']:
        config['TEMP']['test_file'] = build_dict['text']
    else:
        ut = True
        config['TEMP']['text_from_user'] = build_dict['text']
    
    config['TEMP']['preset_cache_dir'] = 'out/'
    config['TEMP']['filter_threshold'] = str(build_dict['filter_threshold'])
    config['TEMP']['filter_model_path'] = build_dict['filter_mod']
    config['TEMP']['llms_api_key'] = str(build_dict['llms_api_key'])
    preset_labels['filter'] = 'roberta-'
    if build_dict['skip_st1'] == 'False':
        
        if 'roberta' in build_dict['st1_mod']:
            config['TEMP']['subtask1_flag'] = 'True'
            config['TEMP']['st1_model_name_or_path'] = build_dict['st1_mod']
            config['TEMP']['st1_roberta_flag'] = 'True'
            preset_labels['st1'] = 'roberta-'
            
        if 'rebel' in build_dict['st1_mod']:
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['rebel_st1_mod'] = build_dict['st1_mod']
            config['TEMP']['rebel_flag'] = 'True'
            config['TEMP']['rebel_st1_flag'] = 'True'
            preset_labels['st1'] = 'rebel-'
        
        if any(mod in build_dict['st1_mod'] for mod in llms):
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['llm_st1_mod'] = build_dict['st1_mod']
            config['TEMP']['llm_flag'] = 'True'
            config['TEMP']['llm_st1_flag'] = 'True'
    
    
    if build_dict['skip_st2'] == 'False':
            
        if 'roberta' in build_dict['st2_mod']:
            config['TEMP']['subtask2_flag'] = 'True'
            config['TEMP']['st2_pretrained_path'] = build_dict['st2_mod']
            config['TEMP']['st2_load_checkpoint_for_test'] = build_dict['st2_mod'] + '/pytorch_model.bin'
            config['TEMP']['st2_roberta_flag'] = 'True'
            preset_labels['st2'] = 'roberta-'
            
        if 'rebel' in build_dict['st2_mod']:
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['rebel_st2_mod'] = build_dict['st2_mod']
            config['TEMP']['rebel_flag'] = 'True'
            config['TEMP']['rebel_st2_flag'] = 'True'
            preset_labels['st2'] = 'rebel-'
        
        if any(mod in build_dict['st2_mod'] for mod in llms):
            config['TEMP']['subtask3_flag'] = 'True'
            config['TEMP']['llm_st2_mod'] = build_dict['st2_mod']
            config['TEMP']['llm_flag'] = 'True'
            config['TEMP']['llm_st2_flag'] = 'True'
    if not ut:
        presets_dict = build_preset_label(build_dict['text'], build_dict['filter_mod'], build_dict['st1_mod'], build_dict['st2_mod'], preset_labels)
        config['TEMP'].update(presets_dict)
    
    return config
    



def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default='None', 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument('--filter_mod', type=str, default='None', help='Path to model')
    parser.add_argument('--filter_threshold', type=float, default=0.8, help='Threshold for classification')
    parser.add_argument(
        "--st1_mod",
        type=str,
        default='None',
        help="model used for st1"
    )
    parser.add_argument(
        "--st2_mod",
        type=str,
        default='None',
        help="model used for st2",
    )
    
    parser.add_argument('--llms_api_key', help='API key for GPT-4', required=False)
    parser.add_argument(
        "--config_path",
        default='None',
        help="Tells the pipeline to use st2 from llm",
    )
    parser.add_argument('--text_from_user', default='None', required=False)
    
    parser.add_argument('--skip_st1', default='False', required=False)
    parser.add_argument('--skip_st2', default='False', required=False)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    default_used = False
    user_dict = {}
    arg_list = [args.test_file, args.filter_mod, args.st1_mod, args.st2_mod, args.text_from_user]
    
    
    if all(x == arg_list[0] for x in arg_list):
        user_submitted = False
    else:
        user_submitted = True
        
    if args.skip_st1 != 'False' or args.skip_st2 != 'False':
        user_submitted = True
    
    if args.config_path != 'None':
        config = configparser.ConfigParser()
        config.read(args.config_path)
        for key in config['TEMP']:
                value = config['TEMP'].get(key)
                if hasattr(args, key):
                    attr_type = type(value)
                    setattr(args, key, attr_type(config['TEMP'][key]))
                    
        user_dict['text'] = args.test_file
        user_dict['filter_mod'] = args.filter_mod
        user_dict['st1_mod'] = args.st1_mod
        user_dict['st2_mod'] = args.st2_mod
        
        user_dict['filter_threshold'] = args.filter_threshold
        user_dict['llms_api_key'] = args.llms_api_key
        
        user_dict['skip_st1'] = args.skip_st1
        user_dict['skip_st2'] = args.skip_st2
    else:
        if user_submitted:
            user_dict['text'] = args.test_file
            user_dict['filter_mod'] = args.filter_mod
            user_dict['st1_mod'] = args.st1_mod
            user_dict['st2_mod'] = args.st2_mod
            
            user_dict['filter_threshold'] = args.filter_threshold
            user_dict['llms_api_key'] = args.llms_api_key
            
            user_dict['skip_st1'] = args.skip_st1
            user_dict['skip_st2'] = args.skip_st2
            
            
        else:
            print('default used')
            
            default_config = configparser.ConfigParser()
            default_config.read('config_default.cfg')
            default_dict = {section: dict(default_config.items(section)) for section in default_config.sections()}
            
            build_dict = default_dict['TEMP']
            
            build_dict['text'] = default_dict['TEMP']['test_file']
            build_dict['filter_threshold'] = args.filter_threshold
            build_dict['llms_api_key'] = args.llms_api_key
            
            build_dict['skip_st1'] = args.skip_st1
            build_dict['skip_st2'] = args.skip_st2
            
            
            config = create_config(build_dict)
            with open('config_temp.cfg', 'w') as configfile:
                config.write(configfile)
                
            json_dict = run_pipeline('config_temp.cfg')
            #print(json_dict)
            df = pd.DataFrame(json_dict)
            if df == 'There are no causal sentences':
                print('There are no causal sentences')
            else:
                df.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
            default_used = True
    
    
    if not default_used:    
        
        default_config = configparser.ConfigParser()
        default_config.read('config_default.cfg')
        default_dict = {section: dict(default_config.items(section)) for section in default_config.sections()}
        build_dict = user_dict
        
        if build_dict['skip_st1'] != 'False':
            build_dict['st1_mod'] = 'None'
            
        if build_dict['skip_st2'] != 'False':
            build_dict['st2_mod'] = 'None'
        
        if args.text_from_user == 'None':
            if build_dict['text'] == 'None':
                build_dict['text'] = default_dict['TEMP']['test_file']
        else:
            build_dict['text'] = args.text_from_user
            
        if build_dict['filter_mod'] == 'None':
            build_dict['filter_mod'] = default_dict['TEMP']['filter_mod']
            
        if build_dict['st1_mod'] == 'None' and build_dict['skip_st1'] == 'False':
            build_dict['st1_mod'] = default_dict['TEMP']['st1_mod']
            
        if build_dict['st2_mod'] == 'None' and build_dict['skip_st2'] == 'False':
            build_dict['st2_mod'] = default_dict['TEMP']['st2_mod']
            
            
        config = create_config(build_dict)
        with open('config_temp.cfg', 'w') as configfile:
            config.write(configfile)
            
        json_dict = run_pipeline('config_temp.cfg')
        #print(json_dict)
        df = pd.DataFrame(json_dict)
        df.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
        