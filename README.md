# Relation_Extraction
This repository contains data (data) and methods for Event relation extraction(CNC), Data augmnetation using LLMs(CommonSenseDataAugmentation)

## Overview

This repository contains code and data for various components of the project. Below is a brief overview of each subfolder:

### 1. CNC

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
### 2. CommonSenseDataAugmentation

The `CommonSenseDataAugmentation` folder hosts code for leveraging Language Models (LLMs) to generate common-sense data for event relations such as enable and prevent.

To run the program, execute the follwing commannd 
```bash
CUDA_VISIBLE_DEVICES='3' python main.py --verbose -x 1
```
x is the number of example 

### 3. Relation Detection
The script can be run from the command line with the following arguments:

- `--train_file`: Path to the training data file (CSV).
- `--val_file`: Path to the validation data file (CSV).
- `--test_file`: Path to the test data file (CSV).
- `--threshold`: Threshold for classification (float).

### 4. LLMs as Event Extractors 
## Training Script



```bash
python main.py test --llm dpo --template prompt_template.yml --verbose --num_examples 2 --news_dataset data/CS.csv --test_dataset data/test.csv --output 2_shot_cs_dpo.csv


### 5. Data

The `Data` folder stores datasets used in the project. These datasets are either constructed by LLMs or obtained previously.

### 6. Refining

The `Refining` folder contains resources for future refinements. 

### 7. Scripts

The `Scripts` folder contains various data processing scripts utilized in the project.

## Aknowledgement 
This work has been partially supported by the French National Research Agency (ANR) within the kFLOW project (Grant n°ANR-21-CE23-0028).

## Contact 
youssra.rebboud@eurecom.fr




