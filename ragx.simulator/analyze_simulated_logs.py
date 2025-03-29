import os
import re
import numpy as np
import csv

# Function to extract the relevant stats from each log file
def extract_stats_from_log(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    system_stats_found = False
    stats = {
        'nvme_read': None,
        'search': None,
        'scoring': None,
        'query_embedding': None
    }

    for line in lines:
        if '=== System Stats ===' in line:
            system_stats_found = True
            continue

        if system_stats_found:
            for stat in stats.keys():
                match = re.search(rf'{stat}: ([\d\.]+)', line)
                if match:
                    stats[stat] = float(match.group(1))
            
            if all(value is not None for value in stats.values()):
                break

    return stats if all(value is not None for value in stats.values()) else None

# Function to calculate stats and write them to a CSV file
def calculate_stats(directory, output_file):
    config_stats = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            parts = filename.split('-')
            config_key = f"{parts[0]}-{parts[1]}-{parts[2]}"

            log_file = os.path.join(directory, filename)
            stats = extract_stats_from_log(log_file)
            
            if stats:
                if config_key not in config_stats:
                    config_stats[config_key] = {'nvme_read': [], 'search': [], 'scoring': [], 'query_embedding': []}
                
                config_stats[config_key]['nvme_read'].append(stats['nvme_read'])
                config_stats[config_key]['search'].append(stats['search'])
                config_stats[config_key]['scoring'].append(stats['scoring'])
                config_stats[config_key]['query_embedding'].append(stats['query_embedding'])

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ["Configuration", "NVMe Read Median", "NVMe Read Average", "NVMe Read Min",
                      "Search Median", "Search Average", "Search Min",
                      "Scoring Median", "Scoring Average", "Scoring Min",
                      "Query Embedding Median", "Query Embedding Average", "Query Embedding Min",
                      "Total Median", "Total Average", "Total Min"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for config_key, stats in config_stats.items():
            total_times = [sum(values) for values in zip(stats['nvme_read'], stats['search'], stats['scoring'], stats['query_embedding'])]
            writer.writerow({
                "Configuration": config_key,
                "NVMe Read Median": np.median(stats['nvme_read']) / 1_000_000,
                "NVMe Read Average": np.mean(stats['nvme_read']) / 1_000_000,
                "NVMe Read Min": np.min(stats['nvme_read']) / 1_000_000,
                "Search Median": np.median(stats['search']) / 1_000_000,
                "Search Average": np.mean(stats['search']) / 1_000_000,
                "Search Min": np.min(stats['search']) / 1_000_000,
                "Scoring Median": np.median(stats['scoring']) / 1_000_000,
                "Scoring Average": np.mean(stats['scoring']) / 1_000_000,
                "Scoring Min": np.min(stats['scoring']) / 1_000_000,
                "Query Embedding Median": np.median(stats['query_embedding']) / 1_000_000,
                "Query Embedding Average": np.mean(stats['query_embedding']) / 1_000_000,
                "Query Embedding Min": np.min(stats['query_embedding']) / 1_000_000,
                "Total Median": np.median(total_times) / 1_000_000,
                "Total Average": np.mean(total_times) / 1_000_000,
                "Total Min": np.min(total_times) / 1_000_000,
            })

if __name__ == "__main__":
    log_directory = "simulation_logs/"
    output_csv = "ragx-results.csv"
    calculate_stats(log_directory, output_csv)
    print(f"Results saved to {output_csv}")
