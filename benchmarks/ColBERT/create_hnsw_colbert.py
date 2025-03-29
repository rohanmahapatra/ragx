import os
import json
import numpy as np
import faiss
import torch
from torch.nn import DataParallel
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import time
import h5py

# Limit the script to run on only 3 GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Function to load data from jsonl file with a progress bar
def load_jsonl(file_path, num_docs=None, key='text'):
    print(f"Loading data from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file), desc="Loading JSONL", unit="lines"):
            if num_docs is not None and i >= num_docs:
                break
            item = json.loads(line)
            data.append(item.get(key, ''))  # Adjust based on your JSON structure
    print(f"Loaded {len(data)} entries from the JSONL file.")
    return data

# Function to generate embeddings using BERT model
def generate_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
    return embeddings.cpu().numpy()

# Main function to create FAISS indices or save embeddings in HDF5 and run evaluations
def main(corpus_file, num_docs=100000, k=10, index_save_file=""):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BERT model and tokenizer
    model_name = 'bert-base-uncased'
    model_name_custom = 'custom_bert_model'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name_custom)

    # Move model to the appropriate device before wrapping it in DataParallel
    model = model.to(device)

    # Wrap the model with DataParallel if more than one GPU is available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        #model = DataParallel(model, device_ids=[0, 1, 2])  # Use the 3 GPUs you selected
        model = DataParallel(model)  

    model.eval()

    # Load documents
    corpus_data = load_jsonl(corpus_file, num_docs=num_docs, key='contents')

    # Generate embeddings for documents
    print("Generating embeddings for documents...")
    corpus_embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(corpus_data), batch_size), desc="Embedding Corpus", unit="batch"):
        batch_texts = corpus_data[i:i+batch_size]
        embeddings = generate_embeddings(batch_texts, model, tokenizer, device)
        corpus_embeddings.append(embeddings)
    corpus_embeddings = np.vstack(corpus_embeddings).astype('float32')

    # Create HNSW Flat index
    print("Creating HNSW Flat index")
    d = corpus_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 100
    index.verbose = True
    index.add(corpus_embeddings)

    # Save the index for future use
    faiss.write_index(index, index_save_file)
    print(f"HNSW Flat index saved at '{index_save_file}'.")


if __name__ == "__main__":
    main(
        corpus_file='/app/dataset/pubmed_500K/pubmed_corpus_500K.jsonl',
        num_docs=500000,
        index_save_file="ColBERT_pubmed_500K_HNSW.faiss",  # Specify the file name for the index
    )