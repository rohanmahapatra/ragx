from datasets import load_dataset
import os

# Specify your CORPUS_DIR
corpus_dir = 'pubmed/'

# Load PubMed corpus data
corpus_name = 'pubmed'
corpus_dataset = load_dataset("jenhsia/ragged", corpus_name, split='train')

# Define the output path for the JSONL file
corpus_jsonl_path = os.path.join(corpus_dir, f"{corpus_name}_corpus.jsonl")

# Save the corpus dataset to a JSONL file using the to_json method
corpus_dataset.to_json(corpus_jsonl_path, orient="records", lines=True)

print(f"Corpus data saved to {corpus_jsonl_path}")