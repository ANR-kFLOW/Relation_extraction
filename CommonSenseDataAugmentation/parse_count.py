import re
import csv
import pandas as pd

# Read the text file
with open('zep_q.txt', 'r') as file:
    text = file.read()

# Define the regular expression pattern
pattern = r'\d+\..*?</ARG1>'

# Find all matches using the regular expression
sentences = re.findall(pattern, text, re.DOTALL)

# Write the extracted sentences to a CSV file
with open('extracted_sentences.csv', 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    for sentence in sentences:
        # Clean up the sentence by removing the </ARG1> tag
        # cleaned_sentence = re.sub(r'</ARG1>', '', sentence)
        # Write the cleaned sentence to the CSV file
        csv_writer.writerow([str(sentence).strip()])
file=pd.read_csv('extracted_sentences.csv')
print(file)
