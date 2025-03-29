cd /app/benchmarks/
python3 download_databases.py
mv bm25_pubmed_500K/ BM25/
mv ColBERT_pubmed_500K_HNSW.faiss ColBERT/
mv Doc2Vec_pubmed_500K_HNSW.faiss Doc2Vec/
mv GTR_pubmed_500K_HNSW.faiss GTR/
mv doc2vec_model* Doc2Vec/
cd /app/