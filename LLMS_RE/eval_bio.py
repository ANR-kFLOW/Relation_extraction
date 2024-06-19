import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from sklearn.metrics import classification_report, precision_recall_fscore_support
import re


def create_bio_tagging(text, subject, object_):
    # Remove commas from the text, subject, and object
    text = str(text).replace(',', '')
    subject = str(subject).replace(',', '')
    object_ = str(object_).replace(',', '')

    words = text.split()
    bio_tags = ['O'] * len(words)

    # Create a helper function to find the index of a substring in a list of words
    def find_sublist(sublist, main_list):
        sublen = len(sublist)
        for i in range(len(main_list) - sublen + 1):
            if main_list[i:i + sublen] == sublist:
                return i
        return -1

    subject_words = subject.split()
    object_words = object_.split()

    # Find and tag the subject
    subject_start_idx = find_sublist(subject_words, words)
    if subject_start_idx != -1:
        bio_tags[subject_start_idx] = 'B-C'
        for i in range(1, len(subject_words)):
            bio_tags[subject_start_idx + i] = 'I-C'

    # Find and tag the object
    object_start_idx = find_sublist(object_words, words)
    if object_start_idx != -1:
        bio_tags[object_start_idx] = 'B-E'
        for i in range(1, len(object_words)):
            bio_tags[object_start_idx + i] = 'I-E'

    return ' '.join(bio_tags)


def main(gt_path, pred_path):
    # Load the CSV files
    ground_truth = pd.read_csv(gt_path)
    predictions = pd.read_csv(pred_path)

    # Add BIO tagging columns to both DataFrames
    ground_truth['BIO_tagging'] = ground_truth.apply(
        lambda row: create_bio_tagging(row['text'], row['subject'], row['object']), axis=1)
    predictions['BIO_tagging'] = predictions.apply(
        lambda row: create_bio_tagging(row['text'], row['subject'], row['object']), axis=1)

    # Calculate precision, recall, and F1 score for BIO tagging
    gt_bio_tags = ground_truth['BIO_tagging'].apply(lambda x: x.split()).tolist()
    pred_bio_tags = predictions['BIO_tagging'].apply(lambda x: x.split()).tolist()

    all_gt_tags = [tag for sublist in gt_bio_tags for tag in sublist]
    all_pred_tags = [tag for sublist in pred_bio_tags for tag in sublist]

    precision = precision_score(all_gt_tags, all_pred_tags, average='weighted',
                                labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])
    recall = recall_score(all_gt_tags, all_pred_tags, average='weighted', labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])
    f1 = f1_score(all_gt_tags, all_pred_tags, average='weighted', labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])

    print(f"BIO Tagging - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Calculate precision, recall, and F1 score for relations
    gt_relations = ground_truth['relation'].tolist()
    pred_relations = predictions['relation'].tolist()

    precision_rel = precision_score(gt_relations, pred_relations, average='weighted')
    recall_rel = recall_score(gt_relations, pred_relations, average='weighted')
    f1_rel = f1_score(gt_relations, pred_relations, average='weighted')


    print(f"Relations - Precision: {precision_rel}, Recall: {recall_rel}, F1 Score: {f1_rel}")

    report = classification_report(all_gt_tags, all_pred_tags)
    print(report)
    # Create mappings for replacements
    replacement_dict = {
        'B-C': 'subject',
        'I-C': 'subject',
        'B-E': 'object',
        'I-E': 'object',
        'O': 'O'  # Assuming 'O' stays as 'O'
    }

    # Function to replace labels
    def replace_labels(labels, mapping):
        return [mapping[label] for label in labels]

    # Replace labels in gt and pred
    gt_replaced = replace_labels(all_gt_tags, replacement_dict)
    pred_replaced = replace_labels(all_pred_tags, replacement_dict)

    # Compute the classification report
    report = classification_report(gt_replaced, pred_replaced, target_names=['subject', 'object', 'O'])
    print(report)
    # Compute micro average
    precision, recall, f1, _ = precision_recall_fscore_support(gt_replaced, pred_replaced,
                                                               average='weighted')
    print(precision,recall,f1)

    # Save the DataFrames with BIO tagging columns
    ground_truth.to_csv('ground_truth_with_bio.csv', index=False)
    predictions.to_csv('predictions_with_bio.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate BIO tagging and relations from ground truth and prediction CSV files.')
    parser.add_argument('--gt', type=str, required=True, help='Path to the ground truth CSV file')
    parser.add_argument('--pred', type=str, required=True, help='Path to the predictions CSV file')

    args = parser.parse_args()
    main(args.gt, args.pred)
