import pandas as pd
import re
import argparse
import os

def extract_label_from_index(index):
    if 'cnc' in index:
        return 'cause'
    label = index.split('_')[0]
    if label.endswith('s'):
        label = label[:-1]
    return label

def convert_to_rebel(data):
    data = data.rename(columns={"text": "context"})

    def create_triplets(row):
        triplets = []
        labels = row['label'].split(',')

        for e1, e2, rel in zip(row['event0'], row['event1'], labels):
            triplet = f'<triplet> {e1} <subj> {e2} <obj> {rel}'
            triplets.append(triplet)
        return triplets

    data['triplets'] = data.apply(create_triplets, axis=1)
    data = data[['context', 'triplets']]
    return data

def extract_events(df):
    events0_all = []
    events1_all = []
    relations_all = []
    texts_all = []

    for index, row in df.iterrows():
        if row['num_rs'] > 0:
            events0 = []
            events1 = []

            for sentence in eval(row['causal_text_w_pairs']):
                event0 = re.search(r'<ARG0>(.*?)</ARG0>', sentence).group(1)
                event1 = re.search(r'<ARG1>(.*?)</ARG1>', sentence).group(1)
                relation = row['label']

                event0 = event0.replace('<SIG0>', '').replace('</SIG0>', '')
                event1 = event1.replace('<SIG0>', '').replace('</SIG0>', '')

                events0.append(event0)
                events1.append(event1)

            events0_all.append(events0)
            events1_all.append(events1)
            relations_all.append(relation)
            texts_all.append(row['text'])

    return events0_all, events1_all, relations_all, texts_all

def main(input_path, output_dir):
    df = pd.read_csv(input_path)
    df['label'] = df['index'].apply(extract_label_from_index)
    events0, events1, relations, texts = extract_events(df)

    new_df = pd.DataFrame({
        'event0': events0,
        'event1': events1,
        'label': relations,
        'text': texts
    })

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    new_df.to_csv('try.csv', index=False)
    train_transformed = convert_to_rebel(new_df)
    train_transformed.to_csv(output_path, index=False)
    print(f"Transformed file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform input CSV file and save the result in a specified directory.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the transformed CSV file.")

    args = parser.parse_args()
    main(args.input, args.output)
