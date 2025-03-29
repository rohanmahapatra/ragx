import os
import jnius_config
import time
import json
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import csv

# Function to load a limited number of documents from JSONL file
def load_jsonl(file_path, max_docs=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
            if max_docs and len(data) >= max_docs:
                break
    return data

# Function to perform BM25 retrieval using Pyserini
def bm25_retrieval(searcher, query, k):
    start_time = time.time()
    hits = searcher.search(query, k=k)
    end_time = time.time()
    search_time = end_time - start_time
    return hits, search_time

# Function to calculate average document length
def calculate_avg_doc_len(index_reader):
    stats = index_reader.stats()
    avg_doc_len = stats['total_terms'] / stats['documents']
    return avg_doc_len

# Function to calculate distinct documents
def calculate_distinct_docs(query_terms, index_reader):
    distinct_doc_ids = set()

    for term in query_terms:
        postings_list = index_reader.get_postings_list(term, analyzer=None)
        if postings_list is not None:
            distinct_doc_ids.update(posting.docid for posting in postings_list)

    return len(distinct_doc_ids)

# Function to score documents using the complete query
def score_documents(hits, query, index_reader):
    scores = []
    scoring_times = []

    for hit in hits:
        docid = hit.docid
        start_time = time.time()

        # Use Pyserini's compute_query_document_score to calculate BM25 score for the entire query
        score = index_reader.compute_query_document_score(docid, query)

        end_time = time.time()
        scoring_times.append(end_time - start_time)
        scores.append((docid, score))

    return scores, scoring_times

# Function to process a single query for timing
def process_query(query_data, searcher, index_reader, k, iterations, count_distinct_docs, dump_postings, query_idx):
    query_processing_times = []
    search_times = []
    scoring_times = []
    num_documents_scored = []

    for _ in range(iterations):
        # Time extraction of a query
        query_time_start = time.time()
        query_text = query_data['input']
        query_time = time.time() - query_time_start
        query_processing_times.append(query_time) 
        query_terms = query_text.split()
            
        # Perform search and get top-K documents
        hits, search_time = bm25_retrieval(searcher, query_text, k=k)
        search_times.append(search_time)

        # Dump unique posting lists to trace.txt if required
        if dump_postings:
            with open('posting_list.txt', 'a') as f:
                f.write(f"QUERY {query_idx + 1}\n")  # Writing QUERY N
                for term in set(query_terms):
                    try:
                        postings_list = index_reader.get_postings_list(term)
                        # Extracting only document IDs from the postings list
                        doc_ids = [str(posting.docid) for posting in postings_list] if postings_list else []
                        f.write(f"Term '{term}': {', '.join(doc_ids)}\n")
                    except Exception as e:
                        # If there is no posting list for the term, just continue without writing anything
                        continue

        # Calculate distinct number of documents using postings lists if required
        if count_distinct_docs:
            distinct_docs_count = calculate_distinct_docs(query_terms, index_reader)
        else:
            distinct_docs_count = 0

        num_documents_scored.append(distinct_docs_count)

        # Score top-K documents and calculate average scoring time
        # scores, score_times = score_documents(hits, query_text, index_reader)
        # scoring_times.extend(score_times)

    return query_processing_times, search_times, scoring_times, num_documents_scored

def run_bm25_retriever(index_path, queries_file, k=100, iterations=1, max_queries=None, num_queries_with_distinct_count=0, max_docs=None, dump_postings=False, output_csv="query_timings.csv"):
    searcher = LuceneSearcher(index_path)
    queries = load_jsonl(queries_file, max_docs)

    if max_queries is not None:
        queries = queries[:max_queries]

    timings = []
    index_reader = IndexReader(index_path)
    avg_doc_len = calculate_avg_doc_len(index_reader)
    N = index_reader.stats()['documents']

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Query Index', 'Average Search Time (s)'])  # Header

        for query_idx, query_data in enumerate(queries):  # Iterate over each query
            count_distinct_docs = query_idx < num_queries_with_distinct_count
            query_processing_times, search_times, scoring_times, num_documents_scored = process_query(
                query_data, searcher, index_reader, k, iterations, count_distinct_docs, dump_postings, query_idx
            )

            # Calculate average statistics for each query
            avg_query_processing_time = np.mean(query_processing_times)
            avg_search_time = np.mean(search_times)
            avg_scoring_time = np.mean(scoring_times)

            # Write to CSV
            csvwriter.writerow([query_idx + 1, f"{avg_search_time:.6f}"])

            # Print average statistics for each query
            print(f"Query {query_idx + 1}: {query_data['input']}")
            print(f"  Average Search Time: {avg_search_time:.6f} seconds")
            print("-" * 40)

if __name__ == "__main__":
    queries_file = '/app/dataset/bioasq/bioasq_questions.json'
    index_path = '/app/benchmarks/BM25/bm25_pubmed_500K'
    output_csv = 'bm25_cpu_dram_results.csv'

    run_bm25_retriever(index_path=index_path,
                       queries_file=queries_file,
                       k=100,
                       iterations=1,
                       max_queries=100,
                       output_csv=output_csv)
