import os
import time
import csv
import faiss
import torch
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# Function to set thread environment variables
def set_threads(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    faiss.omp_set_num_threads(num_threads)  # Set the number of threads for FAISS

def read_jsonl(file_path, max_items=None):
    """
    Read a JSONL file and return a list of documents/queries with progress bar.
    """
    items = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading JSONL", unit='lines')):
            if max_items and i >= max_items:
                break
            items.append(json.loads(line))
    return items

def encode_text_batch(texts, model, num_threads):
    """
    Embed a batch of texts using the T5 model and return the embeddings.
    """
    set_threads(num_threads)
    
    # Use the model's encode function, which handles tokenization and pooling internally
    start_time = time.time()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    elapsed_time = time.time() - start_time
    
    return embeddings, elapsed_time

def search_and_time_hnsw(index, query_embedding, top_k=1000):
    """
    Search the HNSW index and time the search.
    Returns the search results and the time taken.
    """
    start_time = time.time()
    _, _ = index.search(query_embedding, top_k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def run_benchmark(queries_file, output_csv, model, index_file, iterations=1, max_queries=100, batch_size=1, num_threads=1):
    """
    Run the benchmark by replicating a query, embedding in batches, and timing each step.
    Results are saved to a CSV file with progress displayed via tqdm.
    """
    queries = read_jsonl(queries_file, max_queries)

    # Load the HNSW index
    index = faiss.read_index(index_file)
    index.hnsw.efSearch = 1500  # Adjust this for higher recall

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['query_index', 'iteration', 'embedding_time', 'search_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process queries
        for i, single_query in enumerate(tqdm(queries, desc="Processing queries")):
            text = single_query['input']
            
            # Replicate the query to form a batch
            query_batch = [text] * batch_size
            
            # Run the embedding `iterations` times and record the times
            embeddings = None
            for iteration in range(iterations):
                embeddings, embed_time = encode_text_batch(query_batch, model, num_threads)

                # Take the last embedding from the current iteration and search the HNSW index
                search_time = search_and_time_hnsw(index, embeddings[-1:].reshape(1, -1))

                # Save results to CSV, one row per iteration
                writer.writerow({
                    'query_index': i,
                    'iteration': iteration + 1,
                    'embedding_time': embed_time,
                    'search_time': search_time
                })

if __name__ == "__main__":
    # File paths and parameters
    queries_file = '/app/dataset/bioasq/bioasq_questions.json'
    output_index_file = "/app/benchmarks/GTR/GTR_pubmed_500K_HNSW.faiss"
    output_csv = "gtr_cpu_dram_results.csv"
    
    num_threads = 32  # Control the number of threads
    set_threads(num_threads)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5EncoderModel.from_pretrained("t5-base")

    run_benchmark(queries_file, output_csv, model, output_index_file, max_queries=100)

    print(f"Benchmark completed. Results saved to {output_csv}.")