cd /app/dataset
python3 download_pubmed.py
python3 download_bioasq.py
python3 shrink_pubmed.py
mkdir pubmed_500K
mv pubmed_corpus_500K.jsonl pubmed_500K/
cd /app