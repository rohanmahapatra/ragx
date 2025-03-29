cd /app/benchmarks/BM25
python -m pyserini.index --collection JsonCollection --input /app/dataset/pubmed_500K --index bm25_pubmed_500K --generator DefaultLuceneDocumentGenerator --storePositions --storeDocvectors --storeRaw --storeContents
cd /app/benchmarks/ColBERT
python3 create_hnsw_colbert.py
cd /app/benchmarks/Doc2Vec
python3 create_hnsw_doc2vec.py
cd /app/benchmarks/GTR
python3 create_hnsw_gtr.py
cd /app