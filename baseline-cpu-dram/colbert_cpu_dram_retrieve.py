import os
import time
import json
import torch
import numpy as np
import faiss
import csv
from transformers import BertModel, BertTokenizer
from tqdm import tqdm  # Import the tqdm library

# Function to set thread environment variables
def set_threads(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

# Function to load data from jsonl file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            data.append(item.get('input', ''))  # Adjust based on your JSON structure
    return data

# Function to generate embeddings using BERT model
def encode_text_batch(texts, model, tokenizer, num_threads):
    set_threads(num_threads)  # Set threads for BERT embedding

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens

    return end_time-start_time, embeddings.numpy()

# Function to process a single query in batches and collect timing data
def process_query(query, model, tokenizer, faiss_index, batch_size, iterations, num_threads):
    # Initialize timing data collector
    embedding_times = []
    search_times = []

    for _ in range(iterations):
        # Replicate the query to form a batch
        query_batch = [query] * batch_size
            
        # Embed and time the entire batch
        start_time = time.time()
        embed_time, embeddings = encode_text_batch(query_batch, model, tokenizer, num_threads)
        end_time = time.time()
        total_batch_time = end_time - start_time
        
        # Store the average embedding time for the batch
        avg_embedding_time = total_batch_time / batch_size
        #embedding_times.append(avg_embedding_time)
        embedding_times.append(embed_time)

        # Time FAISS search for the batch
        set_threads(num_threads)
        faiss.omp_set_num_threads(num_threads)  # Set the number of threads for FAISS
        start_time = time.time()
        faiss_index.search(embeddings[-1:], 100)
        end_time = time.time()
        #print(query_batch)
        #print(embeddings[-1:])
        search_times.append((end_time - start_time) / batch_size)

    return embedding_times, search_times

# Function to run BERT retriever and write timing data to CSV
def run_bert_retriever(queries_file, output_csv, index_path, num_queries=10, batch_size=10, iterations=10, num_threads=16):
    # Load queries from JSONL file
    queries = load_jsonl(queries_file)[:num_queries]
    print(queries)

    # Initialize BERT tokenizer and model
    model_name = 'bert-base-uncased'
    model_name_custom = '/app/benchmarks/ColBERT/custom_bert_model'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name_custom)
    model.eval()

    # Load FAISS index
    faiss_index = faiss.read_index(index_path)
    faiss_index.hnsw.efSearch = 1500


    # Open CSV file for writing timing data
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Query', 'Iteration', 'Embedding Time (s)', 'Search Time (s)'])

        # Process each query
        for i, query in enumerate(tqdm(queries, desc="Processing Queries", unit="query")):
            embedding_times, search_times = process_query(
                query, model, tokenizer, faiss_index, batch_size, iterations, num_threads
            )

            # Write results for all iterations to CSV
            for j in range(iterations):
                csvwriter.writerow([f"Query {i+1}", j+1, embedding_times[j], search_times[j]])

            # Print average embedding and search time for each query
            avg_embedding_time = np.mean(embedding_times)
            avg_search_time = np.mean(search_times)
            print(f"Query {i+1} - Average Embedding Time: {avg_embedding_time:.4f}s, Average Search Time: {avg_search_time:.4f}s")

if __name__ == "__main__":
    run_bert_retriever(
        queries_file='/app/dataset/bioasq/bioasq_questions.json',
        output_csv='colbert_cpu_dram_results.csv',
        index_path='/app/benchmarks/ColBERT/ColBERT_pubmed_500K_HNSW.faiss',
        num_queries=100 # Set the number of queries to process for testing
    )