# Event_Relation_Extraction

## Overview

This repository contains code and data for various components of the project. Below is a brief overview of each subfolder:
###  [Relation Detection](./Relation-Detection/)
The script can be run from the command line with the following arguments:

- `--train_file`: Path to the training data file (CSV).
- `--val_file`: Path to the validation data file (CSV).
- `--test_file`: Path to the test data file (CSV).
- `--threshold`: Threshold for classification (float).

```bash
python train_thre.py --train_file /path/to/train_file.csv --val_file /path/to/val_file.csv --test_file /path/to/test_file.csv --threshold 0.8

```

### [CNC](./CNC/)

The `CNC` folder contains code for event extraction, adapted from the [CausalNewsCorpus](https://github.com/tanfiona/CausalNewsCorpus/tree/master). This code takes our data that are transformed into the required format and performs event extraction.

To execute the event extraction code, you can use the following command:

```bash
python run_st2.py \
  --dropout 0.3 \
  --learning_rate 2e-05 \
  --model_name_or_path albert-xxlarge-v2 \
  --num_train_epochs 10 \
  --num_warmup_steps 200 \
  --output_dir "outs/baseline" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 8 \
  --report_to wandb \
  --task_name ner \
  --do_train --do_test \
  --train_file <train.csv> \
  --validation_file <val.csv> \
  --test_file <test.csv> \
  --weight_decay 0.005 \
  --use_best_model
```
### [Joint Relation Classification and Event Extraction with Sequence-to-Sequence Language Model-REBEL](./REBEL_Joined_RelationClassification_and_EventExtraction/)
In Thins folder you find [data](./REBEL_Joined_RelationClassification_and_EventExtraction/data/) and [scripts](./REBEL_Joined_RelationClassification_and_EventExtraction/src/) for trainig the seq2seq model REBEL on event relation extraction.

### [LLMs as Event Extractors](./LLMs_as_Relation_Classifiors_and_Event_Extractors/) 

#### Inference Script

```bash
python main.py test --llm LLM-name --template prompt_template.yml --verbose --num_examples 2 --news_dataset examples-dataset --test_dataset ground-truth-dataset  --output prediction-dataset

```
#### Evaluation script 
```bash
python eval_bio.py --gt /path/to/ground_truth_file.csv --pred /path/to/prediction_file.csv
(x is the number of examples)
``` 


###  [Data](./data/)

The `Data` folder stores datasets used in the project. These datasets are either constructed by LLMs or obtained previously.
#### CommonSenseDataAugmentation

The `CommonSenseDataAugmentation` folder hosts code for leveraging Language Models (LLMs) to generate common-sense data for event relations such as enable and prevent.

To run the program, execute the follwing commannd 
```bash
CUDA_VISIBLE_DEVICES='3' python main.py --verbose -x 1
```






