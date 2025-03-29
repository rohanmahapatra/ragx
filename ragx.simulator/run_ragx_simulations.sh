#!/bin/bash

# Unzip the compiled kernels
if [ ! -d "compiled_kernels" ]; then
    unzip compiled_kernels.zip
else
    echo "compiled_kernels directory already exists. Skipping unzip step."
fi


# Define a list of configuration and corresponding trace file paths
declare -A CONFIG_TRACE_MAP=(
    ["config/bm25-500K.yaml"]="../baseline-cpu-dram/traces/bm25_query1_500K_trace.json"
    ["config/splade-500K.yaml"]="../baseline-cpu-dram/traces/spladev2_query1_500K_trace.json"
    ["config/colbert-500K.yaml"]="../baseline-cpu-dram/traces/colbert_query1_500K_trace.json"
    ["config/doc2vec-500K.yaml"]="../baseline-cpu-dram/traces/doc2vec_query1_500K_trace.json"
    ["config/gtr-500K.yaml"]="../baseline-cpu-dram/traces/gtr_query1_500K_trace.json"
)

# Directory to store the simulation logs
LOG_DIR="simulation_logs"
mkdir -p "$LOG_DIR"

# Loop over each configuration and trace file pair
for CONFIG_PATH in "${!CONFIG_TRACE_MAP[@]}"; do
    TRACE_PATH="${CONFIG_TRACE_MAP[$CONFIG_PATH]}"
    CONFIG_FILENAME=$(basename "$CONFIG_PATH" .yaml)

    # Run the simulation with the specified configuration and trace file
    echo "Running eurekastore.py with config: $CONFIG_PATH and trace file: $TRACE_PATH"
    
    # Construct the log file name based on the configuration
    LOG_FILE="${LOG_DIR}/${CONFIG_FILENAME}-trace_output.txt"
    
    # Run the simulation and redirect both stdout and stderr to the log file
    PYTHONUNBUFFERED=1 python3 eurekastore.py "$CONFIG_PATH" "$TRACE_PATH" > "$LOG_FILE" 2>&1
    
    echo "Output saved to $LOG_FILE"
done

# Analyze and collate the results after all simulations are completed
echo "Analyzing and collating the results..."
python3 analyze_simulated_logs.py

echo "Results are available in analyze_simulated_logs.py"
