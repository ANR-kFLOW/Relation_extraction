# Relation_Extraction
TBW

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

### 3. Data

The `Data` folder stores datasets used in the project. These datasets are either constructed by LLMs or obtained previously.

### 4. Refining

The `Refining` folder contains resources for future refinements. 

### 5. Scripts

The `Scripts` folder contains various data processing scripts utilized in the project.

## Getting Started

TBW




