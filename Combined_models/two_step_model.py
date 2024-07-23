import subprocess
import argparse
import numpy as np
import pandas as pd
from accelerate import Accelerator
from accelerate.logging import get_logger
import json
from inf_rebel import test_model
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

from binary_filter import run_filter
from st2_combine import main_st2
from st1_combine import main_st1
from LLM_run import run_LLM

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#print('--------')
available_llms = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4"  # Added GPT-4
}
def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    ''''''
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
    #parser.add_argument('--filter_threshold', type=float, required=True, help='Threshold for classification')
    parser.add_argument('--use_cpu', action="store_true", help='To tell that the model should only use cpu')
    parser.add_argument(
        "--rebel_inf_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    
    
    
    
    #parser.add_argument('--filter_model_path', type=str, help='Path to model')
    
    
    #st1
    
    
    
    
    '''
    parser.add_argument(
        "--st1_get_process_log_level",
        type=str,
        default=None,
        help="output for csv file",
    )
    
    parser.add_argument(
        "--st1_do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    
    parser.add_argument(
        "--st1_output_dir",
        type=str,
        default=None,
        help="output for csv file",
    )
    
    
    parser.add_argument(
        "--st1_do_predict",
        action="store_true",
        help="sets the model to predict",
    )
    parser.add_argument('--st1_use_cpu', action="store_true", help='To tell that the model should only use cpu')
    parser.add_argument('--st1_main_process_first', action="store_true", help='To tell that the model should only use cpu')
    
    '''
    parser.add_argument(
        "--st1_do_predict",
        action="store_true",
        help="sets the model to predict",
    )
    parser.add_argument('--st1_use_cpu', action="store_true", help='To tell that the model should only use cpu')
    
    
    parser.add_argument(
        "--st1_output_dir",
        type=str,
        default=None,
        help="output for csv file",
    )
    parser.add_argument(
        "--st1_task_name",
        type=str,
        default=None,
        help="The name of the task to train on: "
    )
    parser.add_argument(
        "--st1_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--st1_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--st1_max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--st1_overwrite_cache",
        action='store_true',
        help="Overwrite the cached preprocessed datasets or not."
    )
    parser.add_argument(
        "--st1_pad_to_max_length",
        action='store_true',
        default=True,
        help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."
    )
    parser.add_argument(
        "--st1_max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set."
    )
    parser.add_argument(
        "--st1_max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
    )
    parser.add_argument(
        "--st1_max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this value if set."
    )
    parser.add_argument(
        "--st1_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--st1_config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--st1_tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--st1_cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--st1_use_fast_tokenizer",
        action='store_true',
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
    )
    parser.add_argument(
        "--st1_model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id)."
    )
    parser.add_argument(
        "--st1_use_auth_token",
        action='store_true',
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."
    )
    parser.add_argument(
        "--st1_train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--st1_validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--st1_test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--st1_is_regression",
        action='store_true',
        default=False,
        help="If the model to use with predictions is a regression model."
    )
    
    parser.add_argument(
        "--st1_seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    
    
    
    
    
    
    
    
    
    #st1
    
    
    
    
    #st2
    
    
    
    
    
    parser.add_argument(
        "--st2_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--st2_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--st2_train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--st2_validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--st2_test_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--st2_max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of "
            "training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker validation, truncate the number of "
            "validation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_test_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker test, truncate the number of "
            "test examples to this value if set."
        ),
    )
    parser.add_argument(
        "--st2_max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--st2_pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--st2_pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--st2_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--st2_config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--st2_tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--st2_per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--st2_per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--st2_per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--st2_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--st2_weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--st2_num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--st2_max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--st2_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--st2_lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--st2_num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--st2_output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--st2_seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--st2_model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--st2_label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--st2_return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--st2_task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--st2_debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--st2_push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--st2_hub_model_id", 
        type=str, 
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--st2_hub_token", 
        type=str, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--st2_checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--st2_resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--st2_with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--st2_report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--st2_with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--st2_ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    # Custom Arguments
    parser.add_argument(
        "--st2_add_signal_bias",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--st2_signal_bias_on_top_of_lm",
        action="store_true",
        help="Whether or not to add signal bias",
    )
    parser.add_argument(
        "--st2_postprocessing_position_selector",
        action="store_true",
        help="Whether or not to use postprocessing position selector to control overlap problem.",
    )
    parser.add_argument(
        "--st2_mlp",
        action="store_true",
        help="Whether or not to add MLP layer on top of the pretrained LM.",
    )
    parser.add_argument(
        "--st2_signal_classification",
        action="store_true",
        help="Conduct signal classification to verify whether we need to detect signal span.",
    )
    parser.add_argument(
        "--st2_pretrained_signal_detector",
        action="store_true",
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument( #"outs_test/signal_cls"
        "--st2_signal_model_and_tokenizer_path",
        type=str,
        help="Path to pretrained signal detector model.",
    )
    parser.add_argument(
        "--st2_beam_search",
        action="store_true",
        help="Whether to do bean search selection.",
    )
    parser.add_argument(
        "--st2_dropout",
        type=float,
        default=0.1,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--st2_use_best_model",
        action="store_true",
        help="Activate to use model with Highest Overall F1 score, else defaults to Last model.",
    )
    parser.add_argument(
        "--st2_load_checkpoint_for_test",
        type=str,
        default=None,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--st2_do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    parser.add_argument(
        "--st2_do_test",
        action="store_true",
        help="Whether to use model to predict on test set.",
    )
    parser.add_argument(
        "--st2_augmentation_file",
        type=str,
        default=None,
        help="Whether to use pretrained signal detector",
    )
    parser.add_argument(
        "--st2_topk",
        type=int,
        default=5,
        help="Whether to use pretrained signal detector",
    )
    
    
    #st2
    
    
    
    
    #filter
    
    
    
    
    
    
    
    
    
    
    parser.add_argument('--filter_train_file', type=str, help='Path to the training data file')
    parser.add_argument('--filter_val_file', type=str, help='Path to the validation data file')
    parser.add_argument('--filter_test_file', type=str, help='Path to the test data file')
    parser.add_argument('--filter_threshold', type=float, required=True, help='Threshold for classification')
    parser.add_argument('--filter_model_path', type=str, help='Path to model')
    
    
    
    
    
    
    
    
    
    
    
    
    #filter
    
    
    
    #LLMS
    
    
    
    parser.add_argument('LLMS_task', help='Task to perform', choices=['test'])
    parser.add_argument('--LLMS_news_dataset', help='Path to the news dataset CSV file', required=True)
    parser.add_argument('--LLMS_test_dataset', help='Path to the test dataset CSV file', required=True)
    parser.add_argument('--LLMS_num_examples', type=int, help='Number of examples per relation', required=True)
    parser.add_argument('--LLMS_llm', help='LLM to use', default='zephyr', choices=available_llms)
    parser.add_argument('--LLMS_template', help='Path to the prompt template YAML file', required=True)
    parser.add_argument('--LLMS_output', help='Path to save the output predictions CSV file', required=True)
    parser.add_argument('--LLMS_api_key', help='API key for GPT-4', required=False)
    parser.add_argument('--LLMS_verbose', help='Print the full prompt', default=False, action='store_true')
    parser.add_argument("--LLMS_log", type=int, choices=[10, 20, 30, 40, 50], action="store", default=20,
                        help="Verbosity (default: INFO) : DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50")
    
    
    #LLMS
    
    
    #flags
    parser.add_argument(
        "--st2_flag",
        action="store_true",
        help="Tells the pipeline not to use this model",
    )
    parser.add_argument(
        "--st1_flag",
        action="store_true",
        help="Tells the pipeline not to use this model",
    )
    parser.add_argument(
        "--rebel_flag",
        action="store_true",
        help="Tells the pipeline not to use this model",
    )
    parser.add_argument(
        "--LLM_flag",
        action="store_true",
        help="Tells the pipeline not to use this model",
    )
    
    #flags
    #text from user
    parser.add_argument('--text_from_user', type=str, help='this is user submitted text to be evaluated')
    #text from user
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




def para_into_df(s):
    parts = s.split('. ')
    parts = [part + '. ' for part in parts if part != '']
    df = pd.DataFrame()
    df['text'] = parts
    return df
    
def main():
    args = parse_args()
    
    if args.text_from_user != None:
        base_df = para_into_df(args.text_from_user)
    else:
        base_df = pd.read_csv(args.test_file)
        base_df = base_df.drop(columns=['causal_text_w_pairs'])
        base_df = base_df.drop(columns=['num_rs'])
    args.st1_test_file = args.test_file
    args.base_df = base_df
    #print('---------------------------------')
    only_causal_df = run_filter(args)
    #only_causal_df = only_causal_df.drop(columns=['causal_text_w_pairs'])
    args.only_causal = only_causal_df
    df_final = only_causal_df.copy()
    
    if not args.st2_flag:
        st2_df = main_st2(args)
        st2_indexes = []
        st2_pred = []
        for i in range(len(st2_df)):
            st2_indexes.append(len(st2_df[i]))
            st2_pred.append(str(st2_df[i]))
        
        df_final['num_rs'] = st2_indexes
        df_final['pred_st2'] = st2_pred
    print('st1')    
    if not args.st1_flag:
        st1_df = main_st1(args)
        df_final['label_st1'] = st1_df
        df_final = df_final.drop(columns=['label'])
    #print(len(st2_indexes))
    #print(len(st2_pred))
    #print(len(args.only_causal))
    print('rebel')
    if not args.rebel_flag:
        rebel_df = test_model(args.only_causal, args.rebel_inf_model_name_or_path)
        #print(len(args.only_causal))
        #print(len(args.base_df))
        #print(len(rebel_df))
        df_final['triplet-rebel'] = rebel_df['prediction']
        
    #df_final.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
    #return 0
    print('LLM')
    if not args.LLM_flag:    
        args.LLMS_output = 'LLM_pred/' + args.LLMS_llm +'/' + args.LLMS_llm + f'_pred-{datetime.now()}.csv'
        
        llm_df = run_LLM(args)
        llm_df['subj-obj-rel-LLM-' + args.LLMS_llm] = llm_df.apply(lambda row: [row['subject'], row['object'], row['relation']], axis=1)
        df_final['subj-obj-rel-LLM-' + args.LLMS_llm] = llm_df['subj-obj-rel-LLM-' + args.LLMS_llm]
        df_final = df_final.drop(columns=['triplets'])
        
    df_final.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
    
    '''
    args_script_filter = [
        'python', 'binary_filter.py',
        '--train_file', args.train_file,
        '--val_file', args.validation_file,
        '--test_file', args.test_file,
        '--threshold', args.threshold,
        '--filter_model_path', args.filter_model_path
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
        args_script_rebel = ['python', 'train.py' , '--test_file', 'only_causal.csv', '--model_name_or_path', 'model_our_data_gpt_augmented.pth', '--model_type', 'Babelscape/rebel-large', '--config', 'rebel_config/'
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
    
    result4 = subprocess.run(args_script_rebel)
    print("Output from rebel:")
    print(result4.stdout)
    print("Errors from rebel (if any):")
    print(result4.stderr)
    
    json_file_path = "outs/baseline/test-submission-temp.json"
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    '''
    df_combined = pd.read_csv('predicted_as_causal.csv')
    
    '''
            
    df = pd.DataFrame(data)
    
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
            
    '''
    
    #df_rebel = pd.read_csv('rebel_prediction/pred_rebel.csv')
    #df_rebel['subj-rel-obj'] = df_rebel.apply(lambda row: [row['subject'], row['relation'], row['object']], axis=1)
    
    #df_only_causal = df_combined.loc[df_combined['num_rs'] > 0]
    #df_only_causal['label'] = df_st1['prediction']
    #df_only_causal.reset_index(drop=True, inplace=True)
    
    #df_only_causal.loc[:, 'label'] = df_st1['prediction'].values
    #df_only_causal.drop(columns=['triplets'], inplace=True)
    #df_only_causal['subj-rel-obj'] = df_rebel['subj-rel-obj']
    #df_only_causal['subj-rel-obj'] = df_rebel['prediction']
    #df_only_causal.to_csv('combined_outs/'f'final-combined_pred-{datetime.now()}.csv')
    #df_combined.to_csv(f'final-combined_pred-{datetime.now()}.csv')
    

if __name__ == "__main__":
    main()