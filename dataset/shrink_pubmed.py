import json
from tqdm import tqdm

# Function to load a portion of a jsonl file and save it as a new jsonl file
def load_and_save_jsonl(input_file_path, output_file_path, num_docs=None, key='text'):
    print(f"Loading data from {input_file_path}...")
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file), desc="Loading JSONL", unit="lines"):
            if num_docs is not None and i >= num_docs:
                break
            item = json.loads(line)
            data.append(item)  # Save the entire item or adjust based on your JSON structure
    
    print(f"Loaded {len(data)} entries from the JSONL file.")
    
    # Save the loaded data to a new JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')  # Write each JSON object as a separate line
    
    print(f"Saved {len(data)} entries to {output_file_path}.")

# Example usage
input_file  = 'pubmed/pubmed_corpus.jsonl'
output_file = 'pubmed_corpus_500K.jsonl'
load_and_save_jsonl(input_file, output_file, num_docs=500000, key='contents')
