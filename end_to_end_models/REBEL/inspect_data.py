import pandas as pd
import re
import argparse

def extract_triplet(triplet_text):
    """
    Extracts subject, object, and relation using regular expressions.
    
    Args:
        triplet_text (str): The triplet annotation containing <triplet>, <subj>, and <obj> tags.

    Returns:
        tuple: (subject, object, relation) extracted from the triplet text.
    """
    pattern = r"<triplet>(.*?)<subj>(.*?)<obj>(.*)"
    match = re.match(pattern, triplet_text.strip()) if isinstance(triplet_text, str) else None

    if match:
        subject = match.group(1).strip()  # Extract text between <triplet> and <subj>
        object_ = match.group(2).strip()  # Extract text between <subj> and <obj>
        relation = match.group(3).strip()  # Extract text after <obj>
        return subject, object_, relation
    else:
        return None, None, "no_relation"  # Default to "no_relation" if extraction fails


def main():
    parser = argparse.ArgumentParser(description='Extract triplets from a CSV file and compute statistics.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--column', type=str, default='triplet', help='Column containing triplet text (default: triplet)')
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.csv_file)
    
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in the dataset.")
        return
    
    # Extract triplets
    df[['subject', 'object', 'relation']] = df[args.column].apply(lambda x: pd.Series(extract_triplet(x)))
    
    # Compute statistics
    total_samples = len(df)
    relation_counts = df['relation'].value_counts()
    
    print(f"Total samples in dataset: {total_samples}")
    print("Relation value counts:")
    print(relation_counts)
    
if __name__ == '__main__':
    main()

