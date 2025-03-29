import math

class Interconnect:
    def __init__(self, config, logger, stats):
        self.logger = logger
        self.stats = stats
        self.num_accelerators = config['num_accelerators']
        self.bandwidth = config['interconnect']['bandwidth']  # in GB/s
        self.latency = config['interconnect']['latency_ns']  # base latency in nanoseconds
        self.topology = config['interconnect']['topology']  # 'ring', 'mesh', 'tree'
        self.setup_topology()
        self.logger.system(f"Interconnect initialized with {self.num_accelerators} accelerators, topology: {self.topology}")

    def setup_topology(self):
        """Setup the interconnect topology."""
        if self.topology == 'ring':
            self.logger.system("Topology setup as a ring.")
        elif self.topology == 'mesh':
            self.logger.system("Topology setup as a mesh.")
        elif self.topology == 'tree':
            self.logger.system("Topology setup as a tree.")
        else:
            raise ValueError(f"Unsupported topology type: {self.topology}")

    def calculate_comm_time(self, data_size):
        """Calculate communication time based on bandwidth and latency."""
        # Convert data size to GB and compute transfer time in microseconds
        transfer_time_us = (data_size / (1024 ** 3) / self.bandwidth) * 1_000_000
        latency_us = self.latency / 1000  # Convert base latency to microseconds
        return transfer_time_us + latency_us  # Total communication time in µs

    def update_stats(self, stat_type, value, data_size=None):
        """Helper function to update stats efficiently."""
        if stat_type == "all_reduce" or stat_type == "reduce_scatter":
            self.stats.update_interconnect_stat("latency", value)
            if data_size:
                self.stats.update_interconnect_stat("data_transfer_size", data_size)
        else:
            self.stats.update_system_stat("latency_breakdown", value, "interconnect")
            self.stats.update_interconnect_stat("latency", value)
            if data_size:
                self.stats.update_interconnect_stat("data_transfer_size", data_size)

    def all_reduce(self, data_size, algorithm='ring'):
        """All-reduce operation with different algorithms."""
        comm_time = self.calculate_ring_all_reduce(data_size) if algorithm == 'ring' else self.calculate_tree_all_reduce(data_size)
        self.update_stats("all_reduce", comm_time, data_size)
        self.logger.stats(f"All-reduce ({algorithm}) completed with communication time {comm_time} µs.")
        return comm_time

    def calculate_ring_all_reduce(self, data_size):
        """Ring algorithm for all-reduce communication."""
        return (self.num_accelerators - 1) * self.calculate_comm_time(data_size)

    def calculate_tree_all_reduce(self, data_size):
        """Tree algorithm for all-reduce communication."""
        return math.ceil(math.log2(self.num_accelerators)) * self.calculate_comm_time(data_size)

    def scatter(self, source_id, data_size):
        """Scatter operation."""
        comm_time = self.calculate_comm_time(data_size / self.num_accelerators) * (self.num_accelerators - 1)
        self.update_stats("scatter", comm_time, data_size)
        self.logger.stats(f"Scatter completed with communication time {comm_time} µs.")
        return comm_time

    def broadcast(self, source_id, data_size):
        """Broadcast operation."""
        comm_time = self.calculate_comm_time(data_size) * (self.num_accelerators - 1)
        self.update_stats("broadcast", comm_time, data_size)
        self.logger.stats(f"Broadcast completed with communication time {comm_time} µs.")
        return comm_time

    def gather(self, target_id, data_size):
        """Gather operation."""
        comm_time = self.calculate_comm_time(data_size / self.num_accelerators) * (self.num_accelerators - 1)
        self.update_stats("gather", comm_time, data_size)
        self.logger.stats(f"Gather completed with communication time {comm_time} µs.")
        return comm_time

    def point_to_point(self, src_id, dst_id, data_size):
        """Point-to-point communication."""
        comm_time = self.calculate_comm_time(data_size)
        self.update_stats("point_to_point", comm_time, data_size)
        self.logger.stats(f"Point-to-point from {src_id} to {dst_id} completed in {comm_time} µs.")
        return comm_time

    def reduce_scatter(self, data_size, algorithm='ring'):
        """Reduce-scatter operation."""
        comm_time = self.calculate_ring_all_reduce(data_size) if algorithm == 'ring' else self.calculate_tree_all_reduce(data_size)
        self.update_stats("reduce_scatter", comm_time, data_size)
        self.logger.stats(f"Reduce-Scatter ({algorithm}) completed with communication time {comm_time} µs.")
        return comm_time

    def all_gather(self, data_size, algorithm='ring'):
        """All-gather operation."""
        comm_time = self.calculate_ring_all_gather(data_size) if algorithm == 'ring' else self.calculate_tree_all_gather(data_size)
        self.update_stats("all_gather", comm_time, data_size)
        self.logger.stats(f"All-gather ({algorithm}) completed with communication time {comm_time} µs.")
        return comm_time

    def calculate_ring_all_gather(self, data_size):
        """Ring algorithm for all-gather communication."""
        return (self.num_accelerators - 1) * self.calculate_comm_time(data_size)

    def calculate_tree_all_gather(self, data_size):
        """Tree algorithm for all-gather communication."""
        return math.ceil(math.log2(self.num_accelerators)) * self.calculate_comm_time(data_size)
