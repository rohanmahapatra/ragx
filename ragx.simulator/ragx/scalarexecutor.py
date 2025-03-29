from math import ceil


class ScalarExecutor:
    def __init__(self, config, logger, stats, memory_unit):
        # Define base cycle costs for operations
        self.operation_cycle_costs = {
            "addition": 5,              # Fixed cost for addition
            "multiplication": 8,        # Fixed cost for multiplication
            "division": 10,             # Fixed cost for division
            "square_root": 15,          # Fixed cost for square root
            "reduce_dimensions": 20,    # Fixed cost for dimension reduction
            "search": 1000              # Fixed cost for search
        }
        self.element_cycle_cost = config.get("element_cycle_cost", 1)  # Cost per element
        self.logger = logger
        self.stats = stats
        self.memory_unit = memory_unit
        self.frequency_ghz = 1  # 1 GHz frequency

    def execute(self, operation, data_size, scratchpad_index=0, accel_id=None, node_id=None):
        """Perform a specific operation and return the cycles, energy consumed, and latency in µs."""
        if operation not in self.operation_cycle_costs:
            raise ValueError(f"Unknown operation: {operation}")

        # print(f"Executing operation: {operation} on data_size: {data_size} with scratchpad_index: {scratchpad_index}")
        # Step 1: Calculate the operation's total cycles
        # print (f"Executing operation: {operation} on data_size: {data_size} with scratchpad_index: {scratchpad_index}")
        base_cost = self.operation_cycle_costs[operation]
        total_cycles = base_cost + (self.element_cycle_cost * data_size)

        # Step 2: Account for memory access time and energy
        memory_energy = 0
        memory_cycles = 0

        # Load data from DRAM to Scratchpad
        dram_load_energy = self.memory_unit.load_from_dram_to_scratchpad(scratchpad_index, data_size)
        memory_energy += dram_load_energy

        # Scratchpad access cycles: Assume each access takes a base number of cycles
        num_accesses = ceil(data_size / (self.memory_unit.scratchpads[scratchpad_index]['data_width'] // 8))
        memory_cycles += num_accesses * self.element_cycle_cost  # Customize this per access cycle cost if needed

        # Total cycles include operation and memory access
        total_cycles += memory_cycles

        # Step 3: Calculate total energy and latency in microseconds
        operation_energy = self.get_energy_consumption(total_cycles - memory_cycles)
        total_energy = memory_energy + operation_energy
        latency_us = total_cycles / (self.frequency_ghz * 1e3)  # Convert cycles at 1 GHz to µs

        # Log the operation and computed cycles
        self.logger.info(f"Performing {operation} on data size {data_size}. "
                         f"Operation cycles: {total_cycles}, Total energy: {total_energy} nJ, "
                         f"Latency: {latency_us} µs.")

        # Update stats
        self.update_stats(operation, total_cycles, total_energy, accel_id, node_id)

        return latency_us, total_energy

    def get_energy_consumption(self, cycles):
        """Calculate energy consumed based on the number of cycles."""
        energy_per_cycle = 0.5  # Arbitrary energy cost per cycle
        return cycles * energy_per_cycle

    def update_stats(self, operation, total_cycles, energy, accel_id, node_id):
        """Update relevant statistics in the Stats object."""
        
        # Update the accelerator stats for the operation (if an accelerator ID is provided)
        if accel_id:
            self.stats.update_accelerator_stat(accel_id, "scalar", "compute", total_cycles)
            self.stats.update_accelerator_stat(accel_id, "scalar", "energy", energy)

        # Update node-specific stats (if node ID is provided)
        if node_id:
            self.stats.update_trace_stat("nodes", total_cycles, node_id=node_id)
            self.stats.update_trace_stat("energy", energy, node_id=node_id)

        # Update global stats for the operation
        self.stats.update_system_stat("total_latency", total_cycles)
        self.stats.update_system_stat("total_energy", energy)
