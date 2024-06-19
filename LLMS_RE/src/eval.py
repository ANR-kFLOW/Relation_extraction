import pandas as pd
import numpy as np
from nltk.util import ngrams
from collections import Counter
import argparse

# Read the CSV files
predictions_df = pd.read_csv('/data/Youss/RE/Relation_extraction/LLMS_RE/data/predictions/2_shot.csv')
ground_truths_df = pd.read_csv('/data/Youss/RE/Relation_extraction/LLMS_RE/data/test.csv')

# Convert the DataFrames to lists of lists
predictions = predictions_df[['subject', 'relation', 'object']].values.tolist()
ground_truths = ground_truths_df[['subject', 'relation', 'object']].values.tolist()
def get_ngrams(text, n):
    """Generate n-grams from the text"""
    if isinstance(text, float):
        text = str(text)  # Convert float to string
    words = text.split()
    return list(ngrams(words, n))

def count_ngram_overlap(pred, gt, n):
    """Count the overlapping n-grams between prediction and ground truth"""
    pred_ngrams = Counter(get_ngrams(pred, n))
    gt_ngrams = Counter(get_ngrams(gt, n))
    overlap = sum((pred_ngrams & gt_ngrams).values())
    return overlap

def re_score(predictions, ground_truths, type, n=1):
    """Evaluate RE predictions with n-gram overlap"""
    if type == 'relation':
        vocab = ['cause', 'enable', 'prevent', 'intend']
        predictions = [pred[1] for pred in predictions]
        ground_truths = [gt[1] for gt in ground_truths]

    elif type == 'subject':
        predictions = [pred[0] for pred in predictions]
        ground_truths = [gt[0] for gt in ground_truths]
        vocab = np.unique(ground_truths).tolist()

    elif type == 'object':
        predictions = [pred[2] for pred in predictions]
        ground_truths = [gt[2] for gt in ground_truths]
        vocab = np.unique(ground_truths).tolist()

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in vocab + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(ground_truths)
    n_rels = n_sents  # Since every 'sentence' has only 1 relation
    n_found = n_sents

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(predictions, ground_truths):
        for entity in vocab:
            pred_overlap = count_ngram_overlap(pred_sent, entity, n)
            gt_overlap = count_ngram_overlap(gt_sent, entity, n)

            scores[entity]["tp"] += min(pred_overlap, gt_overlap)
            scores[entity]["fp"] += max(pred_overlap - gt_overlap, 0)
            scores[entity]["fn"] += max(gt_overlap - pred_overlap, 0)

    # Compute per relation Precision / Recall / F1
    for entity in scores.keys():
        if scores[entity]["tp"]:
            scores[entity]["p"] = 100 * scores[entity]["tp"] / (scores[entity]["fp"] + scores[entity]["tp"])
            scores[entity]["r"] = 100 * scores[entity]["tp"] / (scores[entity]["fn"] + scores[entity]["tp"])
        else:
            scores[entity]["p"], scores[entity]["r"] = 0, 0

        if not scores[entity]["p"] + scores[entity]["r"] == 0:
            scores[entity]["f1"] = 2 * scores[entity]["p"] * scores[entity]["r"] / (
                    scores[entity]["p"] + scores[entity]["r"])
        else:
            scores[entity]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[entity]["tp"] for entity in vocab])
    fp = sum([scores[entity]["fp"] for entity in vocab])
    fn = sum([scores[entity]["fn"] for entity in vocab])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in vocab])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in vocab])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in vocab])

    # Print evaluation results
    if type == 'relation':
        print(
            "processed {} sentences with {} entities; found: {} relations; correct: {}.".format(n_sents, n_rels,
                                                                                                n_found,
                                                                                                tp))
        print(
            "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
                scores["ALL"]["tp"],
                scores["ALL"]["fp"],
                scores["ALL"]["fn"]))
        print(
            "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
                precision,
                recall,
                f1))
        print(
            "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
                scores["ALL"]["Macro_p"],
                scores["ALL"]["Macro_r"],
                scores["ALL"]["Macro_f1"]))

        for entity in vocab:
            print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
                entity,
                scores[entity]["tp"],
                scores[entity]["fp"],
                scores[entity]["fn"],
                scores[entity]["p"],
                scores[entity]["r"],
                scores[entity]["f1"],
                scores[entity]["tp"] +
                scores[entity][
                    "fp"]))

    else:
        print(f"Macro F1 for {type}: {scores['ALL']['Macro_f1']:.4f}")
        print(f"Micro F1 for {type}: {scores['ALL']['f1']:.4f}")

    return scores, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Relation Extraction predictions.")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions CSV file.")
    parser.add_argument("--ground_truths_file", type=str, help="Path to the ground truths CSV file.")
    args = parser.parse_args()

    # Read the CSV files
    predictions_df = pd.read_csv(args.predictions_file)
    ground_truths_df = pd.read_csv(args.ground_truths_file)

    # Convert the DataFrames to lists of lists
    predictions = predictions_df[['subject', 'relation', 'object']].values.tolist()
    ground_truths = ground_truths_df[['subject', 'relation', 'object']].values.tolist()

    # Evaluate for each type
    scores_relation, precision_relation, recall_relation, f1_relation = re_score(predictions, ground_truths, 'relation')
    scores_subject, precision_subject, recall_subject, f1_subject = re_score(predictions, ground_truths, 'subject')
    scores_object, precision_object, recall_object, f1_object = re_score(predictions, ground_truths, 'object')