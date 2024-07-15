import subprocess
import argparse
import numpy as np
import pandas as pd
from accelerate import Accelerator
from accelerate.logging import get_logger
import json
from datetime import datetime
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker validation, truncate the number of "
            "validation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker test, truncate the number of "
            "test examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id", 
        type=str, 
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    # Custom Arguments
    parser.add_argument(
        "--add_signal_bias",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--signal_bias_on_top_of_lm",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--postprocessing_position_selector",
        action="store_true",
        help="Whether or not to use postprocessing position selector to control overlap problem.",
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Whether or not to add MLP layer on top of the pretrained LM.",
    )
    parser.add_argument(
        "--signal_classification",
        action="store_true",
        help="Conduct signal classification to verify whether we need to detect signal span.",
    )
    parser.add_argument(
        "--pretrained_signal_detector",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument( #"outs_test/signal_cls"
        "--signal_model_and_tokenizer_path",
        type=str,
        help="Path to pretrained signal detector model.",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="Whether to do bean search selection.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help="Activate to use model with Highest Overall F1 score, else defaults to Last model.",
    )
    parser.add_argument(
        "--load_checkpoint_for_test",
        type=str,
        default=None,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to use model to predict on test set.",
    )
    parser.add_argument(
        "--augmentation_file",
        type=str,
        default=None,
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for classification')
    parser.add_argument('--use_cpu', action="store_true", help='To tell that the model should only use cpu')
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None  and args.test_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
def main():
    args = parse_args()
    

    args_script_filter = [
        'python', 'binary_filter.py',
        '--train_file', args.train_file,
        '--val_file', args.validation_file,
        '--test_file', args.test_file,
        '--threshold', args.threshold
    ]
    args_script_filter = [str(x) for x in args_script_filter]
    if args.do_train:
        args_script_st2 = [
        'python', 'st2_combine.py',
        '--train_file', args.train_file,
        '--validation_file', args.validation_file,
        '--test_file', args.test_file, 
        '--dropout', args.dropout, 
        '--learning_rate', args.learning_rate, 
        '--model_name_or_path', args.model_name_or_path, 
        '--num_train_epochs', args.num_train_epochs,
        '--num_warmup_steps', args.num_warmup_steps,
        '--output_dir', args.output_dir,
        '--per_device_train_batch_size', args.per_device_train_batch_size, 
        '--per_device_eval_batch_size', args.per_device_eval_batch_size, 
        '--per_device_test_batch_size', args.per_device_test_batch_size, 
        '--report_to wandb', 
        '--task_name', args.task_name,
        '--do_test', 
        '--do_train', 
        '--weight_decay', args.weight_decay,
        '--use_best_model'
        ]
        args_script_st2 = [str(y) for y in args_script_st2]
    else:
        args_script_st2 = [
            'python', 'st2_combine.py',
            '--test_file', 'only_causal.csv',
            '--dropout', args.dropout, 
            '--learning_rate', args.learning_rate, 
            '--model_name_or_path', args.model_name_or_path, 
            '--num_train_epochs', args.num_train_epochs,
            '--num_warmup_steps', args.num_warmup_steps,
            '--output_dir', args.output_dir,
            '--per_device_train_batch_size', args.per_device_train_batch_size, 
            '--per_device_eval_batch_size', args.per_device_eval_batch_size, 
            '--per_device_test_batch_size', args.per_device_test_batch_size, 
            '--report_to', 'wandb',
            '--task_name', args.task_name,
            '--do_test', 
            '--weight_decay', args.weight_decay,
            '--load_checkpoint_for_test', args.load_checkpoint_for_test
        ]
        args_script_st2 = [str(z) for z in args_script_st2]
        args_script_st1 = ['python', 'st1_combine.py',
                           '--task_name', 'cola',
                           '--do_predict',
                           '--model_name_or_path', 'best_model',
                           '--output_dir', 'outs/2sft_st1_base_new',
                           '--test_file', 'only_causal.csv',
                           '--use_cpu', 'True'
                           ]
        args_script_rebel = ['python' , '--test_file', '', '--model_name_or_path', '', '--model_type', '--config'
                             ]
    #result1 = subprocess.run(args_script1, capture_output=True, text=True)
    result1 = subprocess.run(args_script_filter, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Output from binary_filter.py:")
    print(result1.stdout)
    print("Errors from binary_filter.py (if any):")
    print(result1.stderr)
    
    result2 = subprocess.run(args_script_st2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Output from st2_combine.py:")
    print(result2.stdout)
    print("Errors from st2_combine.py (if any):")
    print(result2.stderr)
    
    result3 = subprocess.run(args_script_st1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Output from st1_combine.py:")
    print(result3.stdout)
    print("Errors from st1_combine.py (if any):")
    print(result3.stderr)
    
    json_file_path = "outs/baseline/test-submission-temp.json"
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    df_combined = pd.read_csv('predicted_as_causal.csv')
    df_st1 = pd.read_csv('outs/2sft_st1_base_new/predict_results_cola.txt', delimiter='\t')
    i_m2 = 0
    for i in range(len(df_combined)):
        d = df_combined.iloc[i]['num_rs']
        if d > 0:
            #df_combined.iat[i]['num_rs'] = len(df.iloc[i_m2]['prediction'])
            #df_combined.iat[i]['causal_text_w_pairs'] = df.iloc[i_m2]['prediction']
            df_combined.iat[i, 7] = len(df.iloc[i_m2]['prediction'])
            df_combined.iat[i, 6] = df.iloc[i_m2]['prediction']
            i_m2 = i_m2 + 1
            
            
    df_only_causal = df_combined.loc[df_combined['num_rs'] > 0]
    #df_only_causal['label'] = df_st1['prediction']
    df_only_causal.reset_index(drop=True, inplace=True)
    
    df_only_causal.loc[:, 'label'] = df_st1['prediction'].values
    df_only_causal.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
    #df_combined.to_csv(f'final-combined_pred-{datetime.now()}.csv')
    

if __name__ == "__main__":
    main()