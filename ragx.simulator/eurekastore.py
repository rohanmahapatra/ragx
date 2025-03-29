import random
import sys
import json
import logging
from config.configparser import ConfigParser
from ragx.ragx import RAGXAccelerator
from ragx.interconnect import Interconnect
from collections import defaultdict
from stats.stats import Stats


# Setting up custom logging levels
STATS_LEVEL_NUM = 25
SYSTEM_LEVEL_NUM = 15
logging.addLevelName(STATS_LEVEL_NUM, "STATS")
logging.addLevelName(SYSTEM_LEVEL_NUM, "SYSTEM")

# Custom methods for new logging levels
def stats(self, message, *args, **kwargs):
    if self.isEnabledFor(STATS_LEVEL_NUM):
        self._log(STATS_LEVEL_NUM, message, args, **kwargs)

def system(self, message, *args, **kwargs):
    if self.isEnabledFor(SYSTEM_LEVEL_NUM):
        self._log(SYSTEM_LEVEL_NUM, message, args, **kwargs)

# Add methods to Logger
logging.Logger.stats = stats
logging.Logger.system = system

# Configure the base logging level and format globally
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Function to create a customized logger for each component
def get_component_logger(component_name):
    # Create a specific logger for the component
    logger = logging.getLogger(component_name)
    logger.setLevel(logging.DEBUG)  # Set to desired level for this component

    # Define a custom formatter for this component
    formatter = logging.Formatter(f'%(asctime)s - {component_name} - %(levelname)s - %(message)s')

    # Create a console handler and set formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():  # Avoid duplicate handlers in case of repeated calls
        logger.addHandler(console_handler)

    return logger

# Create loggers for each component
systolic_logger = get_component_logger("SystolicExecutor")
vector_logger = get_component_logger("VectorExecutor")
scalar_logger = get_component_logger("ScalarExecutor")
interconnect_logger = get_component_logger("Interconnect")
memory_logger = get_component_logger("Memory")

logger = logging.getLogger(__name__)

class EurekaStoreSim:
    def __init__(self, trace_file, config):
        self.trace_file = trace_file
        self.config = config
        self.trace_data = []
        self.combined_sparse_postings = set()  # For tracking unique documents in sparse data
        self.sparse_statistics = {} 
        self.accelerators = []
        self.interconnect = None
        self.stats = {'total_cycles': 0}
        self.batch_size = self.config['query']['batch_size']
        self.query_dimensions = self.config['query']['dimensions']
        self.query_vector = [[1] * self.query_dimensions] * self.batch_size
        self.stats = Stats()
        self.execution_type = self.config['execution_mode']['type']
        
        if config['benchmark'] == 'splade' or config['benchmark'] == 'bm25':
            self.load_sparse_trace_file()
        else:
            self.load_trace_file()
        self.print_trace_stats()
        self.setup_accelerators()
        self.setup_interconnect()
        self.log_system_config()

    def load_trace_file(self):
        """Load trace data from JSON file into a unified structure."""
        with open(self.trace_file, 'r') as file:
            trace_json = json.load(file)
            
            for entry in trace_json:
                if 'node' in entry:
                    # Process as dense entry
                    node = entry["node"]
                    neighbors = entry["neighbors"]
                    partitions = entry["partitions"]
                    embedding_size = entry.get("embedding_size", 128)
                    data_size = entry.get("data_size", 50)
                    assigned_accelerator = entry.get("assigned_accelerator", 0 if self.execution_type != 'distributed' else None)

                    if self.execution_type == 'distributed' and assigned_accelerator is None:
                        raise KeyError(f"Trace entry missing 'assigned_accelerator' in distributed mode: {entry}")

                    # Calculate dense-specific metrics
                    neighbor_count = len(neighbors)
                    
                    # Handle `partitions` as a list, and calculate unique partition count
                    partition_count = len(set(partitions)) if isinstance(partitions, list) else 1
                    
                    # Calculate neighbors per partition
                    neighbors_per_partition = defaultdict(int)
                    if isinstance(partitions, list):
                        for partition, neighbor in zip(partitions, neighbors):
                            neighbors_per_partition[partition] += 1

                    # Add entry to unified trace_data structure
                    self.trace_data.append({
                        "type": "dense",
                        "node": node,
                        "token": None,
                        "neighbors": neighbors,
                        "neighbor_count": neighbor_count,
                        "partitions": partitions,
                        "partition_count": partition_count,
                        "neighbors_per_partition": dict(neighbors_per_partition),
                        "assigned_accelerator": assigned_accelerator,
                        "embedding_size": embedding_size,
                        "data_size": data_size,
                        "document_count": None,
                        "posting_size": None,
                    })

                elif 'token' in entry:
                    # Process as sparse entry
                    token = entry["token"]
                    documents = entry["documents"]
                    partitions = entry["partitions"]
                    posting_size = entry.get("posting_size", 128)
                    data_size = entry.get("data_size", 50)
                    document_count = len(documents)

                    # Update combined postings for sparse data statistics
                    self.combined_sparse_postings.update(documents)

                    # Add entry to unified trace_data structure
                    self.trace_data.append({
                        "type": "sparse",
                        "node": None,
                        "token": token,
                        "neighbors": documents,
                        "neighbor_count": document_count,
                        "partitions": partitions,
                        "partition_count": len(set(partitions)) if isinstance(partitions, list) else 1,
                        "neighbors_per_partition": None,  # Not applicable for sparse
                        "assigned_accelerator": None,  # Not applicable for sparse
                        "embedding_size": None,  # Not applicable for sparse
                        "data_size": data_size,
                        "document_count": document_count,
                        "posting_size": posting_size,
                    })

                else:
                    raise ValueError("Trace entry format not recognized. Each entry must contain either 'node' or 'token'.")

        # Calculate sparse statistics if sparse data was loaded
        if any(entry['type'] == 'sparse' for entry in self.trace_data):
            self._calculate_sparse_statistics()

        logger.info(f"Trace File: Loaded {len(self.trace_data)} entries from the trace file.")

    def load_sparse_trace_file(self):
        """Load trace data from JSON file into a unified structure."""
        with open(self.trace_file, 'r') as file:
            trace_json = json.load(file)
            for entry in trace_json:
                if 'node' in entry:
                    # For the new format, we need to generate neighbors based on "Number of Neighbors"
                    neighbors = [random.randint(1, 1_000_000) for _ in range(entry["Number of Neighbors"])]
                    partitions = [0]

                    # Extract other relevant fields
                    node = entry["node"]
                    embedding_size = entry.get("embedding_size", 128)
                    data_size = entry.get("data_size", 50)

                    # Calculate dense-specific metrics
                    neighbor_count = entry["Number of Neighbors"]
                    partition_count = len(set(partitions)) if isinstance(partitions, list) else 1

                    # Calculate neighbors per partition (assuming equal distribution if partitions are provided)
                    neighbors_per_partition = defaultdict(int)
                    for partition in partitions:
                        neighbors_per_partition[partition] += 1

                    # Add entry to unified trace data structure
                    self.trace_data.append({
                        "type": "dense",
                        "node": node,
                        "token": None,  # No token field in this trace format
                        "neighbors": neighbors,
                        "neighbor_count": neighbor_count,
                        "partitions": partitions,
                        "partition_count": partition_count,
                        "neighbors_per_partition": dict(neighbors_per_partition),
                        "assigned_accelerator": None,  # Assuming no assigned accelerator for now
                        "embedding_size": embedding_size,
                        "data_size": data_size,
                        "document_count": None,  # No document count in this trace format
                        "posting_size": neighbor_count,  # No posting size in this trace format
                    })
                else:
                    raise ValueError("Trace entry format not recognized. Entry must contain 'node'.")

        print(f"Trace File: Loaded {len(self.trace_data)} entries.")

        
    def get_query_vector_size(self, batch_size, query_dimensions):
        total_elements = int(batch_size) * int(query_dimensions) * int(self.config['query']['datatype_bytes'])
        return total_elements

    def get_doc_vector_size(self, batch_size, query_dimensions, num_docs):
        print (f"batch_size: {batch_size}, query_dimensions: {query_dimensions}, num_docs: {num_docs}")
        total_elements = int(batch_size) * int(query_dimensions) * int(self.config['query']['datatype_bytes']) * num_docs
        ## returns the total number of bytes in the query vector
        print (f"total_elements: {total_elements}")
        return total_elements
    
    def _calculate_sparse_statistics(self):
        """Calculate statistics for sparse data, such as unique documents and average postings per token."""
        total_postings = sum(entry['document_count'] for entry in self.trace_data if entry['type'] == 'sparse')
        avg_postings_per_token = total_postings / len([entry for entry in self.trace_data if entry['type'] == 'sparse'])
        unique_documents_count = len(self.combined_sparse_postings)

        # Store calculated statistics
        self.sparse_statistics = {
            "total_postings": total_postings,
            "avg_postings_per_token": avg_postings_per_token,
            "unique_documents_count": unique_documents_count
        }

    def print_trace_stats(self):
        """Print a summary of trace file statistics."""
        num_dense_entries = sum(1 for entry in self.trace_data if entry['type'] == 'dense')
        num_sparse_entries = sum(1 for entry in self.trace_data if entry['type'] == 'sparse')
        
        # Calculate average neighbor count for dense entries
        total_neighbors_dense = sum(entry['neighbor_count'] for entry in self.trace_data if entry['type'] == 'dense')
        avg_neighbors_per_node = total_neighbors_dense / num_dense_entries if num_dense_entries > 0 else 0

        # Sparse statistics
        total_postings = self.sparse_statistics.get("total_postings", 0)
        avg_postings_per_token = self.sparse_statistics.get("avg_postings_per_token", 0)
        unique_documents_count = self.sparse_statistics.get("unique_documents_count", 0)

        if self.config['execution_mode']['type'] == 'dense':
            logger.info("Trace File Statistics:")
            logger.info(f"  Total Entries: {len(self.trace_data)}")
            logger.info(f"  Dense Entries (Nodes): {num_dense_entries}")
            logger.info(f"  Average Neighbors per Dense Node: {avg_neighbors_per_node:.2f}")
        elif self.config['execution_mode']['type'] == 'sparse':
            logger.info("Trace File Statistics:")
            logger.info(f"  Total Entries: {len(self.trace_data)}")
            logger.info(f"  Sparse Entries (Tokens): {num_sparse_entries}")
            logger.info(f"  Total Postings in Sparse Data: {total_postings}")
            logger.info(f"  Average Postings per Sparse Token: {avg_postings_per_token:.2f}")
            logger.info(f"  Unique Documents in Sparse Data: {unique_documents_count}")
        else:
            logger.info("Trace File Statistics:")
            logger.info(f"  Total Entries: {len(self.trace_data)}")
            logger.info(f"  Dense Entries (Nodes): {num_dense_entries}")
            logger.info(f"  Sparse Entries (Tokens): {num_sparse_entries}")
            logger.info(f"  Average Neighbors per Dense Node: {avg_neighbors_per_node:.2f}")
            logger.info(f"  Average Postings per Sparse Token: {avg_postings_per_token:.2f}")
            logger.info(f"  Unique Documents in Sparse Data: {unique_documents_count}")


    def setup_accelerators(self):
        num_accelerators = self.config.get('num_accelerators', 1)
        for i in range(num_accelerators):
            accelerator = RAGXAccelerator(i, self.config, logger, self.stats)
            self.accelerators.append(accelerator)
        logger.system(f"Initialized {num_accelerators} RAGX accelerators.")

    def setup_interconnect(self):
        self.interconnect = Interconnect(self.config, logger,  self.stats)

    def log_system_config(self):
        """Log the system configuration details."""
        logger.system("System Configuration:")
        logger.system(f"Execution Mode: {self.execution_type}")
        logger.system(f"Number of Accelerators: {len(self.accelerators)}")
        logger.system("Accelerator Details:")
        for acc in self.accelerators:
            logger.system(f" - Accelerator ID: {acc.accelerator_id}")
        logger.system("Interconnect Properties:")
        logger.system(f" - Latency: {self.interconnect.latency}")
        logger.system(f" - Bandwidth: {self.interconnect.bandwidth}")
        logger.system(f" - Topology: {self.interconnect.topology}")
    
    def calculate_subbatch_size(self):
        max_subbatches = self.config['query']['max_subbatches']
        if self.batch_size == 1:
            return 1

        for subbatch_size in range(self.batch_size // max_subbatches, self.batch_size + 1):
            if self.batch_size % subbatch_size == 0 and self.batch_size // subbatch_size <= max_subbatches:
                return subbatch_size

        return self.batch_size
    
    def calculate_nvme_read_time(self, data_size):
        """Calculate NVMe read time for a given data size using bandwidth and latency values."""
        page_size = self.config['page_size']  # Page size in bytes
        nvme_bandwidth_gbps = self.config['nvme_bandwidth_gbps']  # NVMe bandwidth in GB/s
        nvme_latency_ns = self.config['nvme_latency_ns']  # NVMe latency in nanoseconds
        print (f"nvme_bandwidth_gbps: {nvme_bandwidth_gbps}")
        print (f"nvme_latency_ns: {nvme_latency_ns}")
        print (f"data_size: {data_size}")
        # Round data size to nearest page size
        pages = (data_size + page_size - 1) // page_size
        print (f"pages: {pages}")
        total_data_size = pages * page_size  # Adjusted data size for rounding

        # Calculate read time using both latency and bandwidth
        data_transfer_time_ns = (total_data_size / (nvme_bandwidth_gbps * (10**9 / 8))) * 1e9  # Convert GB/s to bytes/ns
        total_read_time = nvme_latency_ns + data_transfer_time_ns  # Include initial NVMe latency
        ## returns in ms
        total_read_time_us =  total_read_time/1e3 # convert to us
        return total_read_time_us

    def send_top_k_to_cpu_latency(self, top_k_size):
        """Calculate the latency for transferring top-k results to the CPU in microseconds."""
        d2h_pcie_latency_ns = self.config['d2h_pcie_latency_ns']  # PCIe latency in nanoseconds
        d2h_pcie_bandwidth_gbps = self.config['d2h_pcie_bandwidth_gbps']  # PCIe bandwidth in GB/s

        # Total data size to send: top_k elements * bytes per element
        data_size_bytes = top_k_size * self.config['query']['datatype_bytes']

        # Calculate transfer time using bandwidth
        data_transfer_time_ns = (data_size_bytes / (d2h_pcie_bandwidth_gbps * (10**9 / 8))) * 1e9  # Convert GB/s to bytes/ns
        total_transfer_latency_ns = d2h_pcie_latency_ns + data_transfer_time_ns  # Include PCIe latency

        # Convert from nanoseconds to microseconds
        total_transfer_latency_us = total_transfer_latency_ns / 1_000

        return total_transfer_latency_us

    
    def execute_standalone_dense(self):
        """Standalone dense retrieval with batch processing."""
        accelerator = self.accelerators[0]
        total_energy, total_latency = 0, 0
        cnt = 1
        for entry in self.trace_data:
            max_latency = 0
            logger.system(f"Processing entry {cnt} of {len(self.trace_data)}")
            node = entry["node"]
            neighbors = entry["neighbors"]
            num_neighbors = len(neighbors)
            targets = [node] + neighbors
            
            # metadata_size = self.config['metadata']['size_bytes'] * len(targets)
            
            # Metadata lookup and computation
            # metadata_lookup_latency = accelerator.load_data_from_dram(0, metadata_size)
            # metadata_compute_latency = accelerator.execute_task("search", node=node, neighbors=targets, num_dimensions=self.query_dimensions)
            # Calculate query vector size and NVMe read time
            data_size = self.get_doc_vector_size(self.batch_size, self.query_dimensions, len(targets))
            print (f"the data_size is {data_size}")
            print (f"query dimensions are ", self.query_dimensions)
            print (f"targers are {len(targets)}") 
            nvme_latency_us = self.calculate_nvme_read_time(data_size)
            self.stats.update_system_stat("latency_breakdown", nvme_latency_us, "nvme_read")
            print (f"the nvme_latency_us is {nvme_latency_us}")
            
            # Execute scoring task and update trace
            scoring_latency_us = accelerator.execute_task("scoring", node=node, neighbors=targets, num_dimensions=self.query_dimensions)
            # print (f"the scoring latency is {scoring_latency_us}")
            latency_us = scoring_latency_us + nvme_latency_us
            self.stats.update_system_stat("latency_breakdown", scoring_latency_us, "scoring")

            # print (f"the scoring_latency_us + nvme_latency_us is {latency_us}")
            # Update trace stats for scoring time and node-specific details
            self.stats.update_trace_stat(
                node_id=node,
                scoring_time=scoring_latency_us,
                data_size=data_size,
                num_neighbors=num_neighbors,
                nvme_read=nvme_latency_us,
            )
            
            # Optional reduce task for intermediate entries
            if cnt < len(self.trace_data):
                reduce_latency = accelerator.execute_task("reduce", node=self.trace_data[cnt - 2]["node"], neighbors=targets, num_dimensions=2)
                max_latency = max(latency_us, reduce_latency)
                # print (f"the reduce_latency is {reduce_latency}")
                # print (f"the max_latency is {max_latency}")
                # Update trace stats for reduce time if reduce task was executed
                self.stats.update_trace_stat(node_id=node, reduce_time=reduce_latency)
            else:
                max_latency = latency_us

            total_latency += max_latency
            cnt += 1

        # Final reduction on last node in trace data
        if len(self.trace_data) > 0:
            final_reduce_latency = accelerator.execute_task("reduce", node=self.trace_data[-1]["node"], neighbors=targets, num_dimensions=self.query_dimensions)
            total_latency += final_reduce_latency
            self.stats.update_trace_stat(node_id=self.trace_data[-1]["node"], reduce_time=final_reduce_latency)

        # Final Top-K transfer to CPU and update system latency breakdown
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])

        metadata_latency = self.config['metadata']['compute_latency']
        total_latency += top_k_latency + metadata_latency
        
        self.stats.update_system_stat("latency_breakdown", metadata_latency, "search")
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")
        self.stats.update_system_stat("total_latency", total_latency)


    def execute_distributed_dense(self):
        """Distributed dense retrieval with parallel scoring across accelerators."""
        total_energy, total_latency = 0, 0
        latencies = {acc.accelerator_id: [] for acc in self.accelerators}
        cnt = 1

        for entry in self.trace_data:
            max_latency = 0
            logger.system(f"Processing entry {cnt} of {len(self.trace_data)}")
            targets = [entry["node"]] + entry["neighbors"]
            
            assigned_acc = entry["partitions"]
            accelerator = self.accelerators[assigned_acc]

            # **Scoring**: Calculate document vector size and NVMe read time
            data_size = self.get_doc_vector_size(self.batch_size, self.query_dimensions, len(targets))
            nvme_latency = self.calculate_nvme_read_time(data_size)
            self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")

            # Execute scoring task
            scoring_latency = accelerator.execute_task("scoring", node=entry["node"], neighbors=targets, num_dimensions=self.query_dimensions)
            total_scoring_latency = scoring_latency + nvme_latency
            self.stats.update_trace_stat(
                node_id=entry["node"],
                scoring_time=scoring_latency,
                data_size=data_size,
                num_neighbors=len(entry["neighbors"]),
                nvme_read=nvme_latency,
            )

            # **Reduce**: Execute reduce task and compute maximum latency
            reduce_latency = accelerator.execute_task("reduce", node=entry["node"], neighbors=targets, num_dimensions=self.query_dimensions)
            max_latency = max(total_scoring_latency, reduce_latency)

            self.stats.update_trace_stat(node_id=entry["node"], reduce_time=reduce_latency)

            # Update accelerator stats for reduce latency
            self.stats.update_accelerator_stat(accelerator.accelerator_id, "scalar", "compute", reduce_latency)

            # Track the latency for the assigned accelerator
            latencies[assigned_acc].append(max_latency)
            total_latency += max_latency
            cnt += 1

        # Max latency across accelerators
        max_accel_latency = max(sum(latency_list) for latency_list in latencies.values())
        self.stats.update_system_stat("latency_breakdown", max_accel_latency, "scoring")

        # **Metadata**: Metadata computation latency at the end of the process
        metadata_latency = self.config['metadata']['compute_latency']
        self.stats.update_system_stat("latency_breakdown", metadata_latency, "search")

        # **Top-K Transfer**: Final Top-K transfer to CPU and update system latency breakdown
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])
        total_latency += top_k_latency + max_accel_latency
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")

        # Log and update final latency and energy
        # logger.stats(f"Distributed Dense Mode - Total Energy: {total_energy}, Max Latency: {total_latency}")
        self.stats.update_system_stat("total_latency", total_latency)
        self.stats.update_system_stat("total_energy", total_energy)

    def execute_dimension_split_dense(self):
        """Dimension-split dense retrieval with batch-level subbatching and pipelined latency hiding."""
        dims_per_acc = self.query_dimensions // len(self.accelerators)
        total_energy, total_latency = 0, 0
        subbatch_size = self.calculate_subbatch_size()
        
        print (f"dims_per_acc: {dims_per_acc}")
        print (f"subbatch_size: {subbatch_size}")
        all_reduce_size = self.get_query_vector_size(subbatch_size, dims_per_acc)
        print (f"all_reduce_size: {all_reduce_size}")
        previous_all_reduce_latency = 0


        for entry_idx, entry in enumerate(self.trace_data, start=1):
            logger.system(f"Processing entry {entry_idx} of {len(self.trace_data)}")
            targets = [entry["node"]] + entry["neighbors"]
            data_size = self.get_doc_vector_size(self.batch_size, dims_per_acc, len(targets))
            
            # **Scoring**: Calculate NVMe read time
            nvme_latency = self.calculate_nvme_read_time(data_size)
            self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")
            
            # Process subbatching
            for start in range(0, self.batch_size, subbatch_size):
                current_subbatch_size = min(subbatch_size, self.batch_size - start)
                current_compute_latency = 0

                # **Scoring**: Execute scoring for subbatch
                scoring_latency = self.accelerators[0].execute_task("scoring", neighbors=targets, num_dimensions=dims_per_acc, batch_size=current_subbatch_size)
                self.stats.update_system_stat("latency_breakdown", scoring_latency, "scoring")
                scoring_latency = scoring_latency + nvme_latency
                self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")
                current_compute_latency = max(current_compute_latency, scoring_latency)
            
                # **Reduce**: Execute reduce task for subbatch
                reduced_latency = self.accelerators[0].execute_task("reduce", node=entry["node"], neighbors=targets, num_dimensions=dims_per_acc)
                current_compute_latency = max(current_compute_latency, reduced_latency)
                # **Overlap Latency**: Latency with overlap between subbatch compute and reduce tasks

                # **All-Reduce**: Perform all-reduce
                previous_all_reduce_latency = self.interconnect.all_reduce(data_size=all_reduce_size)
                overlapped_latency = max(current_compute_latency, previous_all_reduce_latency)

                total_latency += overlapped_latency
                print (f"Overlapped Latency: {overlapped_latency}")
                print (f"previous_all_reduce_latency Latency: {previous_all_reduce_latency}")
                self.stats.update_system_stat("latency_breakdown", previous_all_reduce_latency, "interconnect")

            # Update trace stats for current entry (scoring and reduce times)
            self.stats.update_trace_stat(
                node_id=entry["node"],
                scoring_time=scoring_latency,
                data_size=data_size,
                num_neighbors=len(entry["neighbors"]),
                nvme_read=nvme_latency,
            )
            self.stats.update_trace_stat(node_id=entry["node"], reduce_time=reduced_latency)


        # **Metadata**: Metadata computation latency at the end of the process
        metadata_latency = self.config['metadata']['compute_latency']
        self.stats.update_system_stat("latency_breakdown", metadata_latency, "search")

        # **Top-K Transfer**: Final Top-K transfer to CPU and update system latency breakdown
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])
        total_latency += top_k_latency
        total_latency += nvme_latency
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")

        # Log and update final latency and energy
        logger.stats(f"Dimension-Split Dense Mode - Total Energy: {total_energy}, Total Latency: {total_latency}")
        self.stats.update_system_stat("total_latency", total_latency)
        self.stats.update_system_stat("total_energy", total_energy)


    
    def execute_standalone_sparse(self):
        """Standalone mode for sparse retrieval with batch processing."""
        accelerator = self.accelerators[0]
        total_energy, total_latency = 0, 0
        cnt = 1
        
        for entry in self.trace_data:
            max_latency = 0
            logger.system(f"Processing entry {cnt} of {len(self.trace_data)}")
            
            node = entry["node"]
            neighbors = entry["neighbors"]
            num_neighbors = entry["neighbor_count"]
            targets = [node] + neighbors

            # **Data Size & NVMe Read Time**: Calculate query vector size and NVMe read time
            data_size = len(neighbors) * self.config['query']['datatype_bytes']
            nvme_latency = self.calculate_nvme_read_time(data_size)
            
            # **Update NVMe Latency Stat**: Update system stat for NVMe read latency
            self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")

            # **Scoring Task**: Execute scoring task and update scoring latency
            scoring_latency = accelerator.execute_task("posting_list_scoring", node=node, neighbors=neighbors)
            self.stats.update_system_stat("latency_breakdown", scoring_latency, "scoring")
            
            # Calculate total latency so far (nvme + scoring)
            latency = scoring_latency + nvme_latency

            # **Update Scoring Stat**: Update trace stats with scoring latency and node-specific details
            self.stats.update_trace_stat(
                node_id=node,
                scoring_time=scoring_latency,
                data_size=data_size,
                num_neighbors=num_neighbors,
                nvme_read=nvme_latency
            )

            # Optional Reduce Task for Intermediate Entries
            if cnt < len(self.trace_data):
                reduce_latency = accelerator.execute_task("reduce", node=node, neighbors=num_neighbors, num_dimensions=num_neighbors)
                max_latency = max(latency, reduce_latency)
                
                # **Update Reduce Latency Stat**: Update trace stat with reduce latency
                self.stats.update_trace_stat(node_id=node, reduce_time=reduce_latency)
            else:
                max_latency = latency

            total_latency += max_latency
            cnt += 1

        # **Final Reduce Task**: Final reduction on the last node in trace data
        if len(self.trace_data) > 0:
            final_reduce_latency = accelerator.execute_task("reduce", node=self.trace_data[-1]["node"], neighbors=targets, num_dimensions=self.query_dimensions)
            total_latency += final_reduce_latency
            self.stats.update_trace_stat(node_id=self.trace_data[-1]["node"], reduce_time=final_reduce_latency)

        # **Final Top-K Transfer**: Final Top-K transfer to CPU and update top_k_transfer latency stat
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")

        # **Total Latency Update**: Final latency breakdown and total system latency
        total_latency += top_k_latency
        self.stats.update_system_stat("total_latency", total_latency)

        logger.stats(f"Standalone Sparse Mode - Total Energy: {total_energy}, Max Latency: {total_latency}")
        self.stats.update_system_stat("total_energy", total_energy)



    def execute_distributed_sparse(self):
        """Distributed sparse retrieval with parallel scoring across accelerators."""
        total_energy, total_latency = 0, 0
        latencies = {acc.accelerator_id: [] for acc in self.accelerators}
        cnt = 1

        for entry in self.trace_data:
            max_latency = 0
            logger.system(f"Processing entry {cnt} of {len(self.trace_data)}")
            node = entry["node"]
            neighbors = entry["neighbors"]
            assigned_acc = entry["assigned_accelerator"]
            accelerator = self.accelerators[assigned_acc]
            data_size = len(neighbors) * self.config['query']['datatype_bytes']
            
            # **Metadata**: Metadata lookup and latency
            metadata_latency = self.config['metadata']['compute_latency']

            # **NVMe Read Time**: Calculate NVMe read time
            nvme_latency = self.calculate_nvme_read_time(data_size)
            self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")

            # **Scoring Task**: Execute scoring task and calculate total latency
            scoring_latency = accelerator.execute_task("posting_list_scoring", node=node, neighbors=neighbors)
            latency = scoring_latency + nvme_latency + metadata_latency

            # **Reduce Task**: Perform reduce task and compute the maximum latency
            reduce_latency = accelerator.execute_task("reduce", node=node, neighbors=neighbors)
            max_latency = max(latency, reduce_latency)

            # **Update Accelerator Stats**: Track latency for the assigned accelerator
            self.stats.update_accelerator_stat(accelerator.accelerator_id, "scalar", "compute", reduce_latency)
            latencies[assigned_acc].append(max_latency)

            total_latency += max_latency
            cnt += 1

        # **Final Top-K Transfer**: Final Top-K transfer to CPU
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])
        total_latency += top_k_latency
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")

        # **Max Latency**: Track maximum latency across accelerators
        max_accel_latency = max(sum(latency_list) for latency_list in latencies.values())
        logger.stats(f"Distributed Sparse Mode - Total Energy: {total_energy}, Max Latency: {total_latency}")
        self.stats.update_system_stat("total_latency", total_latency)
        self.stats.update_system_stat("total_energy", total_energy)


    def execute_dimension_split_sparse(self):
        """Dimension-split sparse retrieval with batch-level subbatching and pipelined latency hiding."""
        dims_per_acc = self.query_dimensions // len(self.accelerators)
        total_energy, total_latency = 0, 0
        subbatch_size = self.calculate_subbatch_size()
        all_reduce_size = self.get_query_vector_size(subbatch_size, dims_per_acc)
        previous_all_reduce_latency = 0

        for entry_idx, entry in enumerate(self.trace_data, start=1):
            logger.system(f"Processing entry {entry_idx} of {len(self.trace_data)}")
            node = entry["node"]
            neighbors = entry["neighbors"]
            data_size = len(neighbors) * self.config['query']['datatype_bytes']

            # **NVMe Read Time**: Calculate NVMe read time
            nvme_latency = self.calculate_nvme_read_time(data_size)
            self.stats.update_system_stat("latency_breakdown", nvme_latency, "nvme_read")

            # **Subbatch Processing**: Process data in sub-batches across dimensions
            for start in range(0, len(neighbors), subbatch_size):
                current_subbatch_size = min(subbatch_size, len(neighbors) - start)
                current_compute_latency = 0

                # **Metadata**: Metadata lookup and latency
                metadata_latency = self.config['metadata']['compute_latency']

                # **Scoring Task**: Execute scoring task on each accelerator
                for accelerator in self.accelerators:
                    scoring_latency = accelerator.execute_task("dimension_split_posting_list", node=node, neighbors=neighbors[start:start+current_subbatch_size])
                    scoring_latency += metadata_latency
                    current_compute_latency = max(current_compute_latency, scoring_latency)

                # **All-Reduce**: Perform all-reduce operation across accelerators
                overlapped_latency = max(current_compute_latency, previous_all_reduce_latency)
                previous_all_reduce_latency = self.interconnect.all_reduce(data_size=all_reduce_size)
                total_latency += overlapped_latency

        # **Final Top-K Transfer**: Final Top-K transfer to CPU
        top_k_latency = self.send_top_k_to_cpu_latency(self.config['topk'] * self.config['query']['datatype_bytes'])
        total_latency += top_k_latency
        self.stats.update_system_stat("latency_breakdown", top_k_latency, "top_k_transfer")

        # **Energy and Latency Stats**:
        logger.stats(f"Dimension-Split Sparse Mode - Total Latency: {total_latency}")
        self.stats.update_system_stat("total_latency", total_latency)
        self.stats.update_system_stat("total_energy", total_energy)




    def run(self):
        mode = self.config['execution_mode']['parallelism']
        if mode not in ['standalone', 'distributed', 'dimension_split']:
            raise ValueError("Invalid execution mode. Choose from 'standalone', 'distributed', or 'dimension_split'.")

        if self.execution_type == 'dense':
            
            if mode == 'standalone':
                accelerator = self.accelerators[0]
                logger.info("Starting dense retrieval simulation...")
                query_size_bytes = self.get_query_vector_size(self.batch_size, self.query_dimensions)
                embed_time_us = accelerator.embed_query(query_size_bytes, 0)
                self.stats.update_system_stat("latency_breakdown", embed_time_us, "query_embedding")
                self.stats.update_system_stat("total_latency", embed_time_us)
                self.interconnect.broadcast(self.accelerators, data_size=query_size_bytes)
                logger.info("Executing standalone mode...")
                self.execute_standalone_dense()
            elif mode == 'distributed':
                logger.info("Executing distributed mode...")
                self.execute_distributed_dense()
            elif mode == 'dimension_split':
                logger.info("Executing dimension_split mode...")
                self.execute_dimension_split_dense()
        elif self.execution_type == 'sparse':
            logger.info("Starting sparse retrieval simulation...")
            if mode == 'standalone':
                logger.info("Executing standalone mode...")
                self.execute_standalone_sparse()
            elif mode == 'distributed':
                logger.info("Executing distributed mode...")
                self.execute_distributed_sparse()
            elif mode == 'dimension_split':
                logger.info("Executing dimension_split mode...")
                self.execute_dimension_split_sparse()

        logger.info("Simulation completed.")
        logger.info("Results.")

        self.stats.print_stats()

        # for acc in self.accelerators:
        #     logger.stats(f"Accelerator {acc.accelerator_id} cycles: {acc.stats['cycles']}")
        # logger.stats(f"Total simulation cycles: {self.stats['total_cycles']}")

if __name__ == "__main__":
    config_file_path = sys.argv[1]
    trace_file_path = sys.argv[2]
    print ("Units: Times in us, Data size in bytes, Bandwidth in GB")
    configparser = ConfigParser()
    config = configparser.load_config(config_file_path)
    sim = EurekaStoreSim(trace_file_path, config)
    sim.run()