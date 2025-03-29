cd /app/baseline-cpu-dram/
python3 bm25_cpu_dram_retrieve.py
python3 colbert_cpu_dram_retrieve.py
python3 doc2vec_cpu_dram_retrieve.py
python3 gtr_cpu_dram_retrieve.py

mkdir cpu_dram_results
mv *_cpu_dram_results.csv cpu_dram_results/