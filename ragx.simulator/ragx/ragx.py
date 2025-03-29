from ragx.memory_unit import MemoryUnit
from ragx.systolicexecutor import SystolicExecutor
from ragx.vectorexecutor import VectorExecutor
from ragx.scalarexecutor import ScalarExecutor
from ragx.interconnect import Interconnect
from config.select_kernel import select_kernel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logging.basicConfig(level=STATS_LEVEL_NUM, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

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

    # Add the handler to the logger if it hasn't been added already
    if not logger.hasHandlers():  # Avoid duplicate handlers in case of repeated calls
        logger.addHandler(console_handler)

    return logger

# Create loggers for each component
systolic_logger = get_component_logger("SystolicExecutor")
vector_logger = get_component_logger("VectorExecutor")
scalar_logger = get_component_logger("ScalarExecutor")
interconnect_logger = get_component_logger("Interconnect")
memory_logger = get_component_logger("Memory")

systolic_logger.setLevel(STATS_LEVEL_NUM)

class RAGXAccelerator:
    def __init__(self, i, config, logger, stats):
        self.accelerator_id = i
        self.config = config
        self.logger = logger

        dram_cost = config.get('dram_cost', 1e-9)
        self.memory_unit = MemoryUnit(dram_cost, config, memory_logger, stats)
        self.systolic_executor = SystolicExecutor(config, systolic_logger, stats)
        self.vector_executor = VectorExecutor(config, vector_logger, stats)
        self.scalar_executor = ScalarExecutor(config, scalar_logger, stats, self.memory_unit)
        self.interconnect = Interconnect(config, interconnect_logger, stats)

        default_scratchpad_config = {
            'size': 1024,
            'latency': 10,
            'bandwidth': 256
        }
        scratchpad_configs = config.get('scratchpad_configs', [default_scratchpad_config] * 5)
        self.memory_unit.scratchpads = self.memory_unit.initialize_scratchpads(scratchpad_configs)

        self.stats = stats
    
    def execute_task(self, task_type, node=None, neighbors=None, num_dimensions=None, batch_size = 1, scratchpad_index=None, query_vector=None):
        # print(f"Executing task of type: {task_type}")
        if task_type == "embedding":
            if query_vector is None or scratchpad_index is None:
                raise ValueError("Embedding requires query_vector and scratchpad_index.")
            self.embed_query(query_vector, scratchpad_index)
            
        elif task_type == "scoring":
            # print(f"Executing scoring for node - {node}")
            if neighbors is None or num_dimensions is None:
                raise ValueError("Scoring requires neighbors and num_dimensions.")
            if self.config['execution_mode']['type'] == "dense":
                return self.dimension_split_scoring_dense(neighbors, num_dimensions)
            else:
                return self.dimension_split_scoring_sparse(neighbors)
            
        elif task_type in ["reduce"]:
            if neighbors is None:
                raise ValueError("Reduction requires neighbors.")
            print (f"Executing reduction for num_dimensions - {num_dimensions}")
            return self.perform_reduce(num_dimensions)

        elif task_type in ["all_reduce"]:
            if neighbors is None:
                raise ValueError("Reduction requires neighbors.")
            return self.perform_reduction(neighbors)
        
        elif task_type in ["search"]:
            if neighbors is None:
                raise ValueError("Search requires neighbors.")
            return self.perform_search(neighbors)
        
        elif task_type in ["posting_list_scoring"]:
            return self.dimension_split_scoring_sparse(neighbors)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        print (f"Task '{task_type}' executed for node {node}.")
        self.logger.info(f"Task '{task_type}' executed for node {node}.")

    def embed_query(self, query_size, scratchpad_index):
        """Handles the query embedding using systolic array with different embedding kernels."""
        self.logger.info(f"Embedding query using systolic array.")
        # data_size = query_size
        # self.load_data_from_dram(scratchpad_index, data_size)

        # kernel = self.config['kernels']['embedding']
        kernel, kernel_path = select_kernel(self.config, "embedding", self.config['benchmark'], self.config['dataset_size'],\
                                             self.config['query']['batch_size'], self.config['execution_mode']['parallelism'], base_path=self.config['kernels']['base_directory'])
        
        dimensions = self.config['query']['dimensions']
        batch_size = self.config['query']['batch_size']
        
        # self.logger.info("Here is the kernel path: ", kernel_path)
        genesys_stats = self.systolic_executor.execute(kernel, kernel_path, dimensions, batch_size)
        
        latency_us = genesys_stats.get('totTime(us)')
        # print(f"Embedding query took {latency_us} us.")
        if genesys_stats.get('totTime(us)') is None:
            self.logger.error("Failed to execute query embedding due to compute time retrieval failure.")
            return 0
        else:
            return latency_us


    def dimension_split_scoring_dense(self, neighbors, num_dimensions):
        """Dense execution: executes scoring across a single vector processor using the appropriate kernel."""
        num_docs = len(neighbors)
        vector_processors = self.config['vector_processor']['num_processors']
        cores_per_processor = self.config['vector_processor']['num_lanes']

        docs_per_processor = (num_docs + vector_processors - 1) // vector_processors
        self.logger.info(f"Executing scoring for {num_docs} documents; {num_dimensions} dimensions, {docs_per_processor} documents per processor.")

        # Selecting the appropriate kernel based on the configuration
        kernel, kernel_path = select_kernel(self.config, "scoring", self.config['benchmark'], self.config['dataset_size'],\
                                             self.config['query']['batch_size'], self.config['execution_mode']['parallelism'], base_path=self.config['kernels']['base_directory'])

        if kernel is None:
            self.logger.error("Kernel not defined for vector processor.")
            return
        # Compute time for processing all documents
        vector_stats = self.vector_executor.execute(kernel, kernel_path, num_dimensions,  self.config['query']['batch_size'], neighbors)

        latency_us = vector_stats.get('totTime(us)')
        
        # print (f"Latency for {num_docs} documents: {latency_us} us")
        if vector_stats.get('totTime(us)') is None:
            self.logger.error("Failed to execute query embedding due to compute time retrieval failure.")
            return 0
        else:
            return latency_us
        
        # for vp in range(vector_processors):
        #     docs_for_vp = neighbors[vp * docs_per_processor : min((vp + 1) * docs_per_processor, num_docs)]
            
        #     self.logger.info(f"Documents assigned to vector processor {vp}: {docs_for_vp}")

        #     for doc in docs_for_vp:
        #         if not isinstance(doc, (list, tuple)):
        #             self.logger.error(f"Expected a document but got: {doc}")
        #             continue
                
        #         dims_per_core = (num_dimensions + cores_per_processor - 1) // cores_per_processor
        #         vector_data = []
                
        #         for core in range(cores_per_processor):
        #             dim_start = core * dims_per_core
        #             dim_end = min((core + 1) * dims_per_core, num_dimensions)
                    
        #             if dim_start < dim_end:
        #                 vector_data.append(doc[dim_start:dim_end])  
                
        #         kernel_type = self.config.get('embedding_model', 'default_kernel') 
        #         cycles = self.vector_executor.execute(vector_data, kernel_type, len(docs_for_vp), num_dimensions)
        #         total_cycles += cycles

        #         scalar_cycles = self.scalar_executor.execute(dims_per_core)
        #         total_cycles += scalar_cycles
        
        # Pipeline latency calculation
        # pipeline_latency = (vector_processors - 1) * self.config['vector_processor']['pipeline_cycles_per_processor']
        # total_cycles += compute_time + pipeline_latency

        # Energy calculation: multiply by number of vector processors
        # energy_consumption = compute_time * vector_processors  # Assuming energy per cycle can be calculated this way

        # Update the stats for scoring
        # self.stats.update_memory_cycles("scoring_dense", total_cycles)
        # self.stats.update_energy("scoring_dense", energy_consumption)  # Assuming you have a method to update energy stats
        # self.logger.info(f"Dense scoring task completed in {total_cycles} cycles (including pipeline latency).")
        
        return compute_time_us


    def dimension_split_scoring_sparse(self, neighbors):
        """Sparse processing for document distribution across processors without embedding steps."""
        num_docs = len(neighbors)
        print (f"Executing scoring for {num_docs} documents.")
        vector_processors = self.config['vector_processor']['num_processors']
        
        # docs_per_processor = (num_docs + vector_processors - 1) // vector_processors
        # total_cycles = 0

        # Selecting the appropriate kernel based on the configuration
        kernel, kernel_path = select_kernel(self.config, "scoring", self.config['benchmark'], self.config['dataset_size'],\
                                             self.config['query']['batch_size'], self.config['execution_mode']['parallelism'], base_path=self.config['kernels']['base_directory'])
        if kernel is None:
            self.logger.error("Kernel not defined for vector processor.")
            return

        # Compute time for processing all documents
        vector_stats = self.vector_executor.execute(kernel, kernel_path, len(neighbors),  self.config['query']['batch_size'], neighbors)
        
        # for column, total in vector_stats.items():
        #     print(f"{column}: {total}")

        latency_us = vector_stats.get('totTime(us)')
        # print (f"Latency for {num_docs} documents: {latency_us} us")
        if vector_stats.get('totTime(us)') is None:
            self.logger.error("Failed to execute query embedding due to compute time retrieval failure.")
            return 0
        else:
            return latency_us
        
        # for vp in range(vector_processors):
        #     docs_for_vp = neighbors[vp * docs_per_processor : min((vp + 1) * docs_per_processor, num_docs)]
            
        #     for doc in docs_for_vp:
        #         cycles = self.vector_executor.execute(doc)
        #         total_cycles += cycles

        #         scalar_cycles = self.scalar_executor.execute(len(doc))
        #         total_cycles += scalar_cycles

        # pipeline_latency = (vector_processors - 1) * self.config['vector_processor']['pipeline_cycles_per_processor']
        # total_cycles += pipeline_latency

        # Update the stats for sparse scoring
        # self.stats.update_memory_cycles("scoring_sparse", total_cycles)
        # self.logger.info(f"Sparse scoring task completed in {total_cycles} cycles including pipeline latency.")

        # return total_cycles

    def load_data_from_dram(self, scratchpad_index, data_size):
        load_energy = self.memory_unit.load_from_dram_to_scratchpad(scratchpad_index, data_size)
        load_cycles = data_size
        
        # Update stats for memory transfers
        # self.stats.record_memory_transfer("DRAM_to_executor", load_cycles, load_energy)
        self.logger.info(f"Loaded {data_size} units from DRAM in {load_cycles} cycles.")
        return load_cycles

    def store_data_to_dram(self, scratchpad_index, data_size):
        store_energy = self.memory_unit.store_to_dram_from_scratchpad(scratchpad_index, data_size)
        store_cycles = data_size

        # Update stats for memory transfers
        # self.stats.record_memory_transfer("executor_to_DRAM", store_cycles, store_energy)
        self.logger.info(f"Stored {data_size} units to DRAM in {store_cycles} cycles.")
        return store_cycles

    def perform_reduction(self, neighbors):
        """Perform reduction on neighbors and track cycles and energy."""
        data_size = len(neighbors)
        vector_cycles = self.vector_executor.execute(neighbors) 
        # reduction_energy = self.vector_executor.get_energy_consumption()

        # Update stats for reduction task
        self.stats.update_compute_cycles("reduction", vector_cycles)
        # self.stats.update_energy("reduction", reduction_energy)
        self.logger.info(f"Reduction executed in {vector_cycles} cycles.")
        return vector_cycles
    
    def perform_search(self, neighbors):
        """Perform search on metadata and track cycles and energy."""
        data_size = len(neighbors)
        scalar_cycles = self.scalar_executor.execute("search", data_size)
        # reduction_energy = self.vector_executor.get_energy_consumption()

        # Update stats for reduction task
        # self.stats.update_compute_cycles("reduction", vector_cycles)
        # self.stats.update_energy("reduction", reduction_energy)
        self.logger.info(f"Reduction executed in {scalar_cycles} cycles.")
        return scalar_cycles
    

    def perform_reduce(self, num_dimensions):
        """Perform reduction on neighbors and track cycles and energy."""
        scalar_us, energy = self.scalar_executor.execute("addition", num_dimensions)
        # reduction_energy = self.scalar_executor.get_energy_consumption()

        # Update stats for reduction task
        # self.stats.update_compute_cycles("reduction", scalar_cycles)
        # self.stats.update_energy("reduction", reduction_energy)
        # self.logger.info(f"Reduction executed in {scalar_cycles} cycles with energy {reduction_energy}.")
        return scalar_us
    
    def handle_kernel_request(self, kernel_name, data):
        """Choose the appropriate kernel for a task."""
        if kernel_name == "all_reduce":
            self.interconnect.perform_collective("all_reduce", data)
        else:
            self.logger.info(f"Kernel '{kernel_name}' executed.")
