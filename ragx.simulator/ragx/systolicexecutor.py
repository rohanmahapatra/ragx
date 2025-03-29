import subprocess
import csv
import os
from collections import defaultdict

class SystolicExecutor:
    def __init__(self, config, logger, stats):
        self.config = config
        self.cycle_cost = config.get("systolic_cycle_cost", 10)
        self.logger = logger
        self.genesys_config_path = config.get("genesys_config_path", "ragx/genesys/configs/")
        self.genesys_testdir = config.get("genesys_testdir", "ragx/genesys/fpga_sim_validation/test/")
        self.genesys_output_dir = config.get("genesys_output_dir", "ragx/genesys/test-results/")
        self.genesys_output_file = config.get("genesys_output_file", "ragx/genesys/test-results/test.csv")
        self.output_filename = "test-results/" + config['kernels']['embedding'] + ".csv"
        self.cache_filename = config.get("cache_filename", "execution_cache/embedding_cache.csv")
        self.stats = stats

    def execute(self, kernel, kernel_path, dimensions, batch_size):
        # Check the cache for existing results
        self.logger.info(f"here Systolic: Executing kernel {kernel} with dimensions {dimensions} and batch size {batch_size}.")
        self.logger.info(f"Systolic: Checking cache for existing results.")
        cached_result = self.check_cache(kernel, dimensions, batch_size)
        self.logger.info(f"cached_result is {cached_result}")
        if cached_result:
            self.logger.info("Systolic: Reusing cached result based on previous parameters.")
            if self.config.get("print_genesys_output", True):
                for column, total in cached_result.items():
                    print(f"{column}: {total}")
            return cached_result

        # Prepare the command to run Genesys
        command = [
            "python3",  "-u", "ragx/genesys/genesys_sim/genesys.py",
            self.genesys_config_path, kernel_path,
            "--mode", "energy"
        ]
        print(f"Systolic: GeneSys run command: {command}")
        
        try:
            result = subprocess.run(command, check=True, stdout=None, stderr=None, text=True)
            self.logger.info(f"Genesys output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing Genesys simulation: {e}")
            return None

        # Parse the resulting CSV to get the summed statistics
        stats_dict = self.parse_csv_and_sum(self.output_filename)
        # Cache the new results if they are not already in cache
        if not cached_result:
            self.update_cache(kernel, dimensions, batch_size, stats_dict)
        
        if self.config.get("print_genesys_output", True):
            for column, total in stats_dict.items():
                print(f"{column}: {total}")
        
        return stats_dict

    def check_cache(self, kernel, dimensions, batch_size):
        """Check if parameters exist in the cache CSV and return only relevant fields if available."""
        if not os.path.exists(self.cache_filename):
            return None

        # Define the specific fields we want to retrieve
        desired_fields = [
            'totCycles', 'totTime(us)',
            'wbuf_totalReadEnergy', 'bbuf_totalReadEnergy', 'obuf_readEnergy',
            'obuf_writeEnergy', 'vmem1_totalReadEnergy', 'vmem1_totalWriteEnergy',
            'vmem1_totalDDRReadEnergy', 'vmem1_totalDDRWriteEnergy', 
            'vmem2_totalReadEnergy', 'vmem2_totalWriteEnergy', 
            'vmem2_totalDDRWriteEnergy'
        ]

        try:
            with open(self.cache_filename, mode='r') as cache_file:
                reader = csv.DictReader(cache_file)
                for row in reader:
                    # Verify if the row matches the requested kernel, dimensions, and batch_size
                    if (row['kernel'] == kernel and
                        row['dimensions'] == str(dimensions) and
                        row['batch_size'] == str(batch_size)):
                        
                        # Collect only the desired fields from the row
                        return {key: float(value) if key in desired_fields else value 
                                for key, value in row.items() 
                                if key in desired_fields}
        except Exception as e:
            self.logger.error(f"Error reading cache file: {e}")
        
        return None


    def update_cache(self, kernel, dimensions, batch_size, stats_dict):
        """Append new results to the cache CSV file only if they don't already exist."""
        # Check if this entry already exists in cache
        if self.check_cache(kernel, dimensions, batch_size):
            self.logger.info("Systolic: Cache entry already exists, skipping write.")
            return  # Skip if entry is already cached
        
        file_exists = os.path.isfile(self.cache_filename)
        
        try:
            with open(self.cache_filename, mode='a', newline='') as cache_file:
                fieldnames = ['kernel', 'dimensions', 'batch_size'] + list(stats_dict.keys())
                writer = csv.DictWriter(cache_file, fieldnames=fieldnames)
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writeheader()

                # Prepare and write the new row
                row = {'kernel': kernel, 'dimensions': dimensions, 'batch_size': batch_size, **stats_dict}
                writer.writerow(row)
        except Exception as e:
            self.logger.error(f"Error updating cache file: {e}")

    def parse_csv_and_sum(self, file_path):
        """Parse CSV file and sum numeric values, returning a dictionary with results."""
        sums = defaultdict(float)

        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Ignore the first "ignore" row
                headers = next(reader)

                for row in reader:
                    for i in range(2, len(row)):
                        column_name = headers[i]
                        try:
                            numeric_value = float(row[i])
                            sums[column_name] += numeric_value
                        except ValueError:
                            continue  # Ignore non-numeric values

        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {file_path}")
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {e}")

        return dict(sums)
