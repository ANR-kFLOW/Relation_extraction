import torch
import sys
from transformers import BartForConditionalGeneration, BartTokenizer


MODEL_PATH = "common_sense_secomd_cahnce.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BartTokenizer.from_pretrained("Babelscape/rebel-large")


print("Loading model...")
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

def extract_relations(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

  
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=1024)

   
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
   
    if len(sys.argv) < 2:
        print("Usage: python inference.py \"Your text here\"")
        sys.exit(1)
    
    input_text = sys.argv[1]
    print("\n🔹 Input Text: ", input_text)

   
    extracted_relations = extract_relations(input_text)

   
    print("\n🔹 Extracted Relations:\n", extracted_relations)

