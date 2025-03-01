import pandas as pd
import re
import argparse
import os

#def extract_label_from_index(index):
#    if 'cnc' in index:
#        return 'cause'
#    label = index.split('_')[0]
#    if label.endswith('s'):
#        label = label[:-1]
#    return label

def convert_to_rebel(data):
    data = data.rename(columns={"text": "context"})

   
    triplets = []
       
    labels = data['relation']
    events0=data['event1']
    events1=data['event2']
      

    for e1, e2, rel in zip(events0,events1,labels):
            triplet = f'<triplet> {e1} <subj> {e2} <obj> {rel}'
            triplets.append(triplet)
        

    data['triplets'] = triplets
    data = data[['context', 'triplets']]
    return data


def main(input_path, output_dir):
    df = pd.read_csv(input_path)
    
#    df['label'] = df['index'].apply(extract_label_from_index)
#    events0, events1, relations, texts = extract_events(df)

#    new_df = pd.DataFrame({
#        'event0': df['sub'],
#        'event1': df['obj'],
#        'label': df['relation'],
#        'text': df['text']
#    })

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

#    new_df.to_csv('try.csv', index=False)
    train_transformed = convert_to_rebel(df)
    train_transformed.to_csv(output_path, index=False)
    print(f"Transformed file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform input CSV file and save the result in a specified directory.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the transformed CSV file.")

    args = parser.parse_args()
    main(args.input, args.output)

