import os
import time
import ujson as json
import numpy as np
import faiss
import csv
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string
import multiprocessing
import torch

def set_threads(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

def load_and_process_jsonl(file_path, content_field, max_docs=None):
    """Load and preprocess data from a JSONL file into a list of TaggedDocuments."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(tqdm(file, desc=f"Loading {os.path.basename(file_path)}", total=max_docs or sum(1 for _ in open(file_path)))):
            if max_docs is not None and i >= max_docs:
                break  # Stop reading if max_docs is reached
            item = json.loads(line)
            content = item.get(content_field, '').strip()
            if content:
                words = preprocess_string(content)
                if words:
                    documents.append(TaggedDocument(words=words, tags=[item['id']]))

    return documents

def train_doc2vec(corpus, vector_size=100, epochs=40, num_threads=1, model_path="doc2vec_model"):
    """Train a Doc2Vec model on the given corpus using all available threads."""
    print(f"Training Doc2Vec model using {num_threads} threads...")
    model = Doc2Vec(vector_size=vector_size, window=2, min_count=1, workers=num_threads)
    
    # Build vocabulary
    model.build_vocab(corpus)

    # Train model
    for epoch in range(epochs):
        model.train(corpus, total_examples=len(corpus), epochs=1)
        model.alpha -= 0.002  # Decrease the learning rate
        model.min_alpha = model.alpha  # Fix the learning rate, no decay
        print(f"Completed epoch {epoch + 1}/{epochs}")

    model.save(model_path)
    return model

def create_faiss_hnsw_index(doc_vectors, hnsw_path, M=32, ef_construction=200, num_threads=4):
    """Create a FAISS HNSW index for the document vectors."""
    print("Creating FAISS HNSW index...")
    dimension = doc_vectors.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_construction
    faiss.omp_set_num_threads(num_threads)  # Set the number of threads for FAISS
    index.add(doc_vectors)
    print("FAISS index created successfully.")

    # Save the index to a file
    faiss.write_index(index, hnsw_path)

    return index

def encode_query(query, model, num_threads=4):
    """Generate an embedding for a query using the Doc2Vec model."""
    model.workers = num_threads  # Restrict the number of threads for inference
    start_time = time.time()
    query_embedding = model.infer_vector(preprocess_string(query))
    embedding_time = time.time() - start_time
    return query_embedding, embedding_time

def faiss_scoring(index, query_embedding, ef_search=325):
    """Perform FAISS scoring using HNSW index."""
    start_time = time.time()
    index.hnsw.efSearch = ef_search
    distances, indices = index.search(np.array([query_embedding]), 100)
    search_time = time.time() - start_time
    return distances, indices, search_time

def select_topk(distances, indices, topK):
    """Select the top-K results."""
    start_time = time.time()
    topk_indices = []
    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])[:topK]
        topk_indices.append(indices[i][sorted_indices])
    top_k_time = time.time() - start_time
    return np.array(topk_indices), top_k_time

def process_query(query, model, faiss_index, topK, iterations, collect, num_threads):
    """Process a query and collect detailed timing data."""
    times = {'embedding': [], 'search': [], 'top_k': []}

    # Determine iteration counts for each stage based on `collect` parameter
    embedding_iterations = iterations if 'embedding' in collect or 'all' in collect else 1
    search_iterations = iterations if 'search' in collect or 'all' in collect else 1
    topk_iterations = iterations if 'topK' in collect or 'all' in collect else 1

    # Embed the query
    for _ in range(embedding_iterations):
        query_embedding, embedding_time = encode_query(query, model, num_threads)
        times['embedding'].append(embedding_time)

    # Search the index
    set_threads(num_threads)
    for _ in range(search_iterations):
        distances, indices, search_time = faiss_scoring(faiss_index, query_embedding, ef_search=375)
        times['search'].append(search_time)

    # Select top-K results
    #for _ in range(topk_iterations):
        #topk_indices, top_k_time = select_topk(distances, indices, topK)
    #    times['top_k'].append(0)

    return times#, topk_indices

def run_experiment(queries_file, output_csv, corpus_file, num_queries=10, topK=5, iterations=10, max_docs=None, collect=['all'], num_threads=1, model_path="doc2vec_model", hnsw_path="doc2vec_hnsw_5M.faiss"):
    
    # Train or load Doc2Vec model
    if os.path.exists(model_path):
        model = Doc2Vec.load(model_path)
        print("Loaded existing Doc2Vec model.")
    else:
        corpus = load_and_process_jsonl(corpus_file, 'contents', max_docs=max_docs)
        print(f"{len(corpus)} documents loaded from the corpus.")
        model = train_doc2vec(corpus, num_threads=num_threads, model_path=model_path)

    # Create or load FAISS HNSW index
    faiss.omp_set_num_threads(num_threads)  # Set the number of threads for FAISS
    if os.path.exists(hnsw_path):
        print("here")
        faiss_index = faiss.read_index(hnsw_path)
        print("Loaded existing FAISS HNSW index.")
    else:
        print(hnsw_path)
        doc_vectors = np.array([model.infer_vector(doc.words) for doc in tqdm(corpus, desc="Encoding corpus vectors")])
        faiss_index = create_faiss_hnsw_index(doc_vectors, hnsw_path, num_threads=num_threads)

    # Load and process the queries
    queries = load_and_process_jsonl(queries_file, 'input', max_docs=num_queries)
    print(f"{len(queries)} queries loaded for processing.")

    # Prepare CSV output
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Query', 'Iteration', 'Embedding Time (s)', 'Search Time (s)'])

        for i, doc in enumerate(tqdm(queries, desc="Processing queries")):
            query = ' '.join(doc.words)
            timing_data = process_query(query, model, faiss_index, topK, iterations, collect, num_threads)

            # Write each iteration's data to the CSV file
            for j in range(iterations):
                csvwriter.writerow([
                    f"Query {i+1}", j+1,
                    timing_data['embedding'][j] if j < len(timing_data['embedding']) else '',
                    timing_data['search'][j] if j < len(timing_data['search']) else ''
                    #timing_data['top_k'][j] if j < len(timing_data['top_k']) else ''
                ])

if __name__ == "__main__":
    run_experiment(
        queries_file='/app/dataset/bioasq/bioasq_questions.json',
        corpus_file = '/app/dataset/pubmed_500K/pubmed_corpus_500K.jsonl',
        output_csv='doc2vec_cpu_dram_results.csv',
        num_queries=100,
        topK=100,
        model_path="/app/benchmarks/Doc2Vec/doc2vec_model",  # Set the path for saving/loading the model
        hnsw_path="/app/benchmarks/Doc2Vec/Doc2Vec_pubmed_500K_HNSW.faiss"  # Set the path for saving/loading the HNSW index
    )