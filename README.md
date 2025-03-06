
# Refined Causality Extraction from Text

This repository contains the data and scripts used for refined causality extraction from text.

## Data
The `data` directory contains the datasets used in our experiments.

## End-to-End Models
The `end_to_end_models` directory includes models used for training, specifically:
- **REBEL**
- **RoBERTa_end_to_end**

### Training Instructions
- **REBEL**:  
  - Train the REBEL model using:  
    ```bash
    python end_to_end_models/REBEL/train.py
    ```
  - Transform data into the format required for REBEL-based event relation extraction (ERE) using:  
    ```bash
    python end_to_end_models/REBEL/Data_transform.py
    ```
    *Ensure to adjust the script as needed for your dataset.*

- **RoBERTa_end_to_end**:  
  - Train the RoBERTa model for end-to-end event relation extraction using:  
    ```bash
    python end_to_end_models/RoBERTa_end_to_end/end_to_end_train.py
    ```

## Separate Task Training
For training each subtask in refined causality extraction, use the following scripts:

- **Relation Detection**:  
  ```bash
  python separate_tasks/Relation_detection.py
  ```

- **Relation Classification**:  
  ```bash
  python separate_tasks/relation_classification.py
  ```

- **Event Extraction**:  
  ```bash
  python separate_tasks/EE.py
  ```

