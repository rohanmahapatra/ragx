from ragx.memory_unit import MemoryUnit
from ragx.navigator import Navigator  # Might not be used currently
from ragx.systolicexecutor import SystolicExecutor
from ragx.vectorexecutor import VectorExecutor
from ragx.scalarexecutor import ScalarExecutor
from ragx.interconnect import Interconnect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGXAccelerator:
    def __init__(self, i, config, logger):
        self.accelerator_id = i
        self.config = config
        self.logger = logger

        # Initialize the memory unit with correct parameters
        dram_cost = config.get('dram_cost', 1e-9)  # Example default DRAM cost
        self.memory_unit = MemoryUnit(dram_cost, config, logger)

        # Initialize other components
        self.systolic_executor = SystolicExecutor(config, logger)
        self.vector_executor = VectorExecutor(config, logger)
        self.scalar_executor = ScalarExecutor(config, logger)
        self.interconnect = Interconnect(config, logger)

        # Initialize scratchpads with five default configurations
        default_scratchpad_config = {
            'size': 1024,  # Default size for a scratchpad
            'latency': 10,  # Default latency for a scratchpad
            'bandwidth': 256  # Default bandwidth for a scratchpad
        }

        # Create a list of five default scratchpads
        default_scratchpad_configs = [default_scratchpad_config] * 5

        # Use configuration file or default scratchpad configs
        scratchpad_configs = config.get('scratchpad_configs', default_scratchpad_configs)

        # Initialize scratchpads using the configurations
        self.memory_unit.scratchpads = self.memory_unit.initialize_scratchpads(scratchpad_configs)

        self.stats = {
            "cycles": 0,
            "energy": 0,
            "memory_transfers": {
                "DRAM_to_executor": {"cycles": 0, "energy": 0},
                "executor_to_DRAM": {"cycles": 0, "energy": 0},
                "executor_to_register_file": {"cycles": 0, "energy": 0},
                "register_file_to_executor": {"cycles": 0, "energy": 0}
            }
        }

    def load_data_from_dram(self, scratchpad_index, data_size):
        """Loads data from DRAM to a scratchpad."""
        load_energy = self.memory_unit.load_from_dram_to_scratchpad(scratchpad_index, data_size)
        load_cycles = data_size  # Placeholder for cycles
        self.stats["cycles"] += load_cycles
        self.stats["energy"] += load_energy
        self.stats["memory_transfers"]["DRAM_to_executor"]["cycles"] += load_cycles
        self.stats["memory_transfers"]["DRAM_to_executor"]["energy"] += load_energy
        self.logger.info(f"Loaded {data_size} units from DRAM in {load_cycles} cycles.")
        return load_cycles  # Return cycles for further processing

    def store_data_to_dram(self, scratchpad_index, data_size):
        """Stores data from a scratchpad to DRAM."""
        store_energy = self.memory_unit.store_to_dram_from_scratchpad(scratchpad_index, data_size)
        store_cycles = data_size  # Placeholder for cycles
        self.stats["cycles"] += store_cycles
        self.stats["energy"] += store_energy
        self.stats["memory_transfers"]["executor_to_DRAM"]["cycles"] += store_cycles
        self.stats["memory_transfers"]["executor_to_DRAM"]["energy"] += store_energy
        self.logger.info(f"Stored {data_size} units to DRAM in {store_cycles} cycles.")
        return store_cycles  # Return cycles for further processing

    def load_data_from_register_file(self, data_size):
        """Loads data from the register file to an executor."""
        load_energy = self.memory_unit.load_from_register_file(data_size)
        load_cycles = data_size  # Placeholder for cycles
        self.stats["cycles"] += load_cycles
        self.stats["energy"] += load_energy
        self.stats["memory_transfers"]["register_file_to_executor"]["cycles"] += load_cycles
        self.stats["memory_transfers"]["register_file_to_executor"]["energy"] += load_energy
        self.logger.info(f"Loaded {data_size} units from register file in {load_cycles} cycles.")
        return load_cycles  # Return cycles for further processing

    def store_data_to_register_file(self, data_size):
        """Stores data from an executor to the register file."""
        store_energy = self.memory_unit.store_to_register_file(data_size)
        store_cycles = data_size  # Placeholder for cycles
        self.stats["cycles"] += store_cycles
        self.stats["energy"] += store_energy
        self.stats["memory_transfers"]["executor_to_register_file"]["cycles"] += store_cycles
        self.stats["memory_transfers"]["executor_to_register_file"]["energy"] += store_energy
        self.logger.info(f"Stored {data_size} units to register file in {store_cycles} cycles.")
        return store_cycles  # Return cycles for further processing

    def embed_query(self, query, scratchpad_index):
        """Embeds a query using the systolic executor."""
        # Load data from DRAM to scratchpad
        self.load_data_from_dram(scratchpad_index, len(query))
        
        # Properly execute the systolic_executor process
        compute_time = len(query) * self.systolic_executor.cycle_cost
        self.stats['cycles'] += compute_time
        self.logger.info(f"Embedding query completed in {compute_time} cycles.")
        
        # Store data back to DRAM from scratchpad
        self.store_data_to_dram(scratchpad_index, len(query))

    def perform_reduction(self, neighbors):
        """Perform reduction on neighbors."""
        data_size = len(neighbors)
        vector_cycles = self.vector_executor.execute(neighbors) 
        reduction_energy = self.vector_executor.get_energy_consumption()
        self.logger.info(f"Reduction executed in {vector_cycles} cycles with energy {reduction_energy}.")
        return vector_cycles, reduction_energy

    def execute_vector_task(self, neighbors, num_dimensions):
        """Distribute neighbors across N vector processors and M engines."""
        num_docs = len(neighbors)
        vector_processors = self.config['vector_processor']['num_processors']
        engines_per_processor = self.config['vector_processor']['num_lanes']
        
        # Calculate how many docs each processor will handle
        docs_per_processor = (num_docs + vector_processors - 1) // vector_processors  # Ceiling division
        vector_cycles = 0
        total_energy = 0
        
        for vp in range(vector_processors):
            docs_for_vp = neighbors[vp * docs_per_processor : min((vp + 1) * docs_per_processor, num_docs)]
            
            # Split docs among engines in this vector processor
            for engine_id in range(engines_per_processor):
                # Simulate vector execution
                ## todo: decide the datasize and then just use the precompiled kernel
                cycles = self.vector_executor.execute(docs_for_vp)
                
                # Ensure cycles has a fallback value
                if cycles is None:
                    cycles = 0  # Fallback value if execute returns None
                
                vector_cycles = max(vector_cycles, cycles)  # Take the maximum as the execution time

        # Accounting for latency for the last vector processor
        pipeline_latency = (vector_processors - 1) * self.config['vector_processor']['pipeline_cycles_per_processor']
        total_cycles = vector_cycles + pipeline_latency
        
        # Update stats with pipelining cycles
        self.stats["cycles"] += total_cycles
        self.logger.info(f"Vector task completed in {total_cycles} cycles.")
        return total_cycles

    def execute_task(self, task_type, node, neighbors, num_dimensions, scratchpad_index):
        data_size = len(neighbors)
        self.load_data_from_dram(scratchpad_index, data_size)
        
        if task_type == "embedding":
            self.embed_query(node, scratchpad_index)
        
        elif task_type == "scoring":
            # Perform element-wise operations on vector processors
            vector_cycles = self.execute_vector_task(neighbors, num_dimensions)
            
            # Move results from vector registers to scalar registers
            move_register_cycles = self.store_data_to_register_file(data_size)
            
            # Perform scalar operations
            scalar_cycles = self.execute_scalar_operations(data_size)

            # Store results back to DRAM
            store_to_dram_cycles = self.store_data_to_dram(scratchpad_index, data_size)

            total_cycles = vector_cycles + scalar_cycles + move_register_cycles + store_to_dram_cycles
            self.stats["cycles"] += total_cycles
            self.logger.info(f"Scoring task for node {node} completed in {total_cycles} cycles.")

        elif task_type in ["reduce", "all_reduce"]:
            reduction_result = self.perform_reduction(neighbors)
            self.logger.info(f"Reduction task for node {node} completed with result: {reduction_result}.")

        self.logger.info(f"RAGXAccelerator: {task_type} task for {node} completed.")

    def execute_scalar_operations(self, data_size):
        """Perform operations using scalar units."""
        scalar_units = self.config['scalar_unit']['num_units']
        
        # Simulate scalar execution
        cycles = self.scalar_executor.execute(data_size)
        return cycles  # Return the number of cycles after completion

    def handle_kernel_request(self, kernel_name, data):
        """Choose the appropriate kernel for a task."""
        if kernel_name == "all_reduce":
            self.interconnect.perform_collective("all_reduce", data)
        else:
            self.logger.info(f"Kernel '{kernel_name}' executed.")