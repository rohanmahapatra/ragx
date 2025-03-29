from collections import defaultdict

class Stats:
    """Tracks comprehensive execution stats including system-wide, accelerator-specific, and node-by-node statistics."""

    def __init__(self):
        # System-wide stats
        self.system_stats = {
            "total_latency": 0,
            "total_energy": 0,
            "num_accelerators": 0,
            "latency_breakdown": {
                "query_embedding": 0,
                "nvme_read": 0,
                "interconnect": 0,
                "top_k_transfer": 0,
                "search": 0,
                "scoring": 0,
            }
        }

        # Trace stats (for individual nodes and other metrics)
         # Trace stats (for individual nodes and other metrics)
        self.trace_stats = {
            "nodes": defaultdict(lambda: {
                "scoring_time": 0,
                "reduce_time": 0,
                "embedding_time": 0,
                "energy": 0,
                "data_size": 0,
                "num_neighbors": 0,
                "metadata_latency": 0,
                "nvme_read": 0,
            }),
            "reduction": 0,
            "scoring": 0,
            "embedding_time": 0,
            "energy": 0
        }

        # Accelerator stats (including systolic, vector, scalar, etc.)
        self.accelerator_stats = defaultdict(lambda: {
            "cycles": 0,
            "scratchpad_energy": 0,
            "read_write_times": {"read": 0, "write": 0},
            "stalls": 0,
            "energy": 0,
            "systolic": {"compute": 0, "memory_transfer": {"cycles": 0, "energy": 0}, "stalls": 0, "energy": 0},
            "vector": {"compute": 0, "memory_transfer": {"cycles": 0, "energy": 0}, "stalls": 0, "energy": 0},
            "scalar": {"compute": 0, "memory_transfer": {"cycles": 0, "energy": 0}, "stalls": 0, "energy": 0},
            "nodes_processed": defaultdict(lambda: {"scoring_cycles": 0, "reduce_cycles": 0, "data_transfer": {"size": 0, "latency": 0}})
        })

        # Interconnect stats
        self.interconnect_stats = {
            "data_transfer_size": 0,
            "latency": 0
        }

        # Energy stats for the system
        self.system_energy_stats = {
            "total_energy": 0,
            "interconnect_energy": 0,
            "accelerator_energy": 0
        }

    # Update system stats with nested structure handling
    def update_system_stat(self, name, value, subkey=None):
        """Updates a system-level stat. If `subkey` is provided, updates that specific sub-stat."""
        if subkey:
            if name in self.system_stats and subkey in self.system_stats[name]:
                self.system_stats[name][subkey] += value
            else:
                raise ValueError(f"Unknown system stat '{name}' or subkey '{subkey}'.")
        elif name in self.system_stats:
            self.system_stats[name] += value
        else:
            raise ValueError(f"Unknown system stat '{name}'.")

    # Update trace stats for node-specific or general stats
    def update_trace_stat(self, node_id, scoring_time=None, reduce_time=None, embedding_time=None, energy=None, data_size=None, num_neighbors=None, nvme_read=None, metadata_latency=None):
        """Updates a trace-level stat for a node, only updating values that are explicitly provided."""

        # print(f"Updating trace stat for node {node_id}")

        # Only update each stat if a non-None value is provided
        if scoring_time is not None:
            self.trace_stats["nodes"][node_id]["scoring_time"] += scoring_time
        if reduce_time is not None:
            self.trace_stats["nodes"][node_id]["reduce_time"] += reduce_time
        if embedding_time is not None:
            self.trace_stats["nodes"][node_id]["embedding_time"] += embedding_time
        if energy is not None:
            self.trace_stats["nodes"][node_id]["energy"] += energy
        if data_size is not None:
            self.trace_stats["nodes"][node_id]["data_size"] = data_size
        if num_neighbors is not None:
            self.trace_stats["nodes"][node_id]["num_neighbors"] = num_neighbors
        if metadata_latency is not None:
            self.trace_stats["nodes"][node_id]["metadata_latency"] = metadata_latency
        if nvme_read is not None:
            self.trace_stats["nodes"][node_id]["nvme_read"] = nvme_read
        # Print debug information to verify correct updates
        # print(f"Updated trace stats for node {node_id}: {self.trace_stats['nodes'][node_id]}")



    # Update accelerator stats with optional nested structure handling
    def update_accelerator_stat(self, accel_id, category, name, value, subkey=None):
        """Updates a stat for a specific accelerator category, with optional nested subkey support."""
        if category not in ["systolic", "vector", "scalar"]:
            raise ValueError(f"Unknown category '{category}' for accelerator '{accel_id}'.")

        if subkey:
            if name in self.accelerator_stats[accel_id][category] and subkey in self.accelerator_stats[accel_id][category][name]:
                self.accelerator_stats[accel_id][category][name][subkey] += value
            else:
                raise ValueError(f"Unknown stat '{name}' or subkey '{subkey}' for category '{category}' in accelerator '{accel_id}'.")
        elif name in self.accelerator_stats[accel_id][category]:
            self.accelerator_stats[accel_id][category][name] += value
        else:
            raise ValueError(f"Unknown stat '{name}' for category '{category}' in accelerator '{accel_id}'.")

    # Update interconnect stats
    def update_interconnect_stat(self, name, value):
        """Updates an interconnect-level stat."""
        if name in self.interconnect_stats:
            self.interconnect_stats[name] += value
        else:
            raise ValueError(f"Unknown interconnect stat '{name}'.")

    # Update system energy stats
    def update_system_energy_stat(self, name, value):
        """Updates a system energy stat."""
        if name in self.system_energy_stats:
            self.system_energy_stats[name] += value
        else:
            raise ValueError(f"Unknown system energy stat '{name}'.")

    
    # Print stats (updated to show detailed trace information)
    def print_stats(self):
        """Prints all stats in a structured format."""
        print("\n=== Trace Stats ===")
        for key, value in self.trace_stats.items():
            if key == "nodes":
                print("Nodes:")
                for node_id, node_stats in value.items():
                    print(f"  Node {node_id}:")
                    for node_key, node_value in node_stats.items():
                        print(f"    {node_key}: {node_value}")
            else:
                print(f"{key}: {value}")

        print("\n=== Accelerator Stats ===")
        for accel_id, accel_stats in self.accelerator_stats.items():
            print(f"Accelerator {accel_id}:")
            for category in ["systolic", "vector", "scalar"]:
                print(f"  {category}:")
                for key, value in accel_stats[category].items():
                    print(f"    {key}: {value}")
            for key, value in accel_stats.items():
                if isinstance(value, dict) and key not in ["systolic", "vector", "scalar"]:
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")

        print("\n=== Interconnect Stats ===")
        for key, value in self.interconnect_stats.items():
            print(f"{key}: {value}")

        print("\n=== System Energy Stats ===")
        for key, value in self.system_energy_stats.items():
            print(f"{key}: {value}")
        
        print("\n\n=== System Stats ===")
        for key, value in self.system_stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")