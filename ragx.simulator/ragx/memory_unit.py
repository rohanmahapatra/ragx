import os
from ragx.genesys.cacti_sweep import CactiSweep
from math import ceil

class MemoryUnit:
    def __init__(self, dram_cost, config, logger, stats):
        tech_node = 45
        dir_path = os.path.join(os.path.dirname(__file__), 'genesys', 'sram')
        sram_opt_dict = {'technology (u)': tech_node * 1.e-3}
        self.sram_obj = CactiSweep(
            bin_file=os.path.join(dir_path, 'cacti/cacti'),
            csv_file=os.path.join(dir_path, 'cacti_sweep.csv'),
            default_json=os.path.join(dir_path, 'default.json'),
            default_dict=sram_opt_dict
        )
        self.dram_cost = dram_cost
        self.logger = logger
        self.stats = stats

        # Configuration for various memory units
        self.ibuf_size = config.get("ibuf_size", 512)
        self.wbuf_size = config.get("wbuf_size", 256)
        self.obuf_size = config.get("obuf_size", 128)
        self.vbuf_size = config.get("vbuf_size", [1024, 1024])
        self.register_file_size = config.get("register_file_size", 1024)

        self.register_read_energy = 0.5
        self.register_write_energy = 0.5
        self.scratchpads = []

    def initialize_scratchpads(self, scratchpad_configs):
        self.scratchpads = []
        for config in scratchpad_configs:
            scratchpad = {
                "size": config.get("size", 512),
                "banks": config.get("banks", 1),
                "bank_depth": config.get("bank_depth", 256),
                "data_width": config.get("data_width", 32),
            }
            self.scratchpads.append(scratchpad)
        return self.scratchpads

    def get_sram_energy_costs(self, scratchpad):
        total_sram_size = scratchpad['banks'] * scratchpad['bank_depth'] * scratchpad['data_width'] // 8
        cfg_dict = {
            'size (bytes)': total_sram_size,
            'block size (bytes)': scratchpad['data_width'] // 8,
            'read-write port': 1
        }
        sram_data = self.sram_obj.get_data_clean(cfg_dict)

        # Use .iloc[0] to extract the single value from the Series
        read_energy = float(sram_data['read_energy_nJ'].iloc[0]) / (scratchpad['data_width'] // 8)
        write_energy = float(sram_data['write_energy_nJ'].iloc[0]) / (scratchpad['data_width'] // 8)
        leak_power = float(sram_data['leak_power_mW'].iloc[0]) * scratchpad['banks']
        area = float(sram_data['area_mm^2'].iloc[0]) * scratchpad['banks']

        return read_energy, write_energy, leak_power, area


    def compute_read_energy(self, scratchpad, num_accesses):
        read_energy, _, _, _ = self.get_sram_energy_costs(scratchpad)
        total_read_energy = num_accesses * read_energy

        # self.stats.update_energy("read_energy", total_read_energy)
        return num_accesses, total_read_energy

    def compute_write_energy(self, scratchpad, num_accesses):
        _, write_energy, _, _ = self.get_sram_energy_costs(scratchpad)
        total_write_energy = num_accesses * write_energy

        self.stats.update_energy("write_energy", total_write_energy)
        return num_accesses, total_write_energy

    def dram_read_energy(self, num_tiles, tile_size):
        total_data_size_bits = num_tiles * tile_size * 8
        total_dram_read_energy = total_data_size_bits * self.dram_cost

        # self.stats.record_memory_transfer("DRAM_to_executor", num_tiles, total_dram_read_energy)
        return num_tiles, total_data_size_bits, total_dram_read_energy

    def dram_write_energy(self, num_tiles, tile_size):
        total_data_size_bits = num_tiles * tile_size * 8
        total_dram_write_energy = total_data_size_bits * self.dram_cost

        self.stats.record_memory_transfer("executor_to_DRAM", num_tiles, total_dram_write_energy)
        return num_tiles, total_data_size_bits, total_dram_write_energy

    def load_from_dram_to_scratchpad(self, scratchpad_index, data_size):
        scratchpad = self.scratchpads[scratchpad_index]
        num_accesses = ceil(data_size / (scratchpad['data_width'] // 8))
        num_tiles = ceil(data_size / scratchpad['size'])

        dram_reads, _, dram_read_energy = self.dram_read_energy(num_tiles, scratchpad['size'])
        scratchpad_reads, scratchpad_read_energy = self.compute_read_energy(scratchpad, num_accesses)

        total_energy = dram_read_energy + scratchpad_read_energy
        # self.stats.update_memory_cycles("DRAM_to_executor", dram_reads)
        # self.stats.update_energy("load_to_scratchpad", total_energy)

        self.logger.info(f"Loaded {data_size} bytes from DRAM to Scratchpad {scratchpad_index}: "
                         f"{dram_reads} DRAM reads, {num_accesses} Scratchpad reads, "
                         f"Total energy: {total_energy} nJ.")

        return total_energy

    def store_to_dram_from_scratchpad(self, scratchpad_index, data_size):
        scratchpad = self.scratchpads[scratchpad_index]
        num_accesses = ceil(data_size / (scratchpad['data_width'] // 8))
        num_tiles = ceil(data_size / scratchpad['size'])

        scratchpad_writes, scratchpad_write_energy = self.compute_write_energy(scratchpad, num_accesses)
        dram_writes, _, dram_write_energy = self.dram_write_energy(num_tiles, scratchpad['size'])

        total_energy = scratchpad_write_energy + dram_write_energy
        self.stats.update_memory_cycles("executor_to_DRAM", dram_writes)
        self.stats.update_energy("store_to_dram", total_energy)

        self.logger.info(f"Stored {data_size} bytes from Scratchpad {scratchpad_index} to DRAM: "
                         f"{num_accesses} Scratchpad writes, {dram_writes} DRAM writes, "
                         f"Total energy: {total_energy} nJ.")

        return total_energy

    def transfer_between_scratchpads(self, src_index, dest_index, data_size):
        src_scratchpad = self.scratchpads[src_index]
        dest_scratchpad = self.scratchpads[dest_index]

        num_accesses = ceil(data_size / (src_scratchpad['data_width'] // 8))
        scratchpad_reads, scratchpad_read_energy = self.compute_read_energy(src_scratchpad, num_accesses)
        scratchpad_writes, scratchpad_write_energy = self.compute_write_energy(dest_scratchpad, num_accesses)

        total_energy = scratchpad_read_energy + scratchpad_write_energy
        self.stats.update_energy("scratchpad_transfer", total_energy)

        self.logger.info(f"Transferred {data_size} bytes from Scratchpad {src_index} to Scratchpad {dest_index}: "
                         f"{scratchpad_reads} Scratchpad reads, {scratchpad_writes} Scratchpad writes, "
                         f"Total energy: {total_energy} nJ.")

        return total_energy

    def load_from_register_file(self, data_size):
        num_accesses = ceil(data_size / (32 // 8))
        register_read_energy = self.get_register_energy('read')
        total_energy = num_accesses * register_read_energy

        self.stats.record_memory_transfer("register_file_to_executor", num_accesses, total_energy)
        self.logger.info(f"Loaded {data_size} bytes from Register File: {num_accesses} reads, "
                         f"Total energy: {total_energy} nJ.")

        return total_energy

    def store_to_register_file(self, data_size):
        num_accesses = ceil(data_size / (32 // 8))
        register_write_energy = self.get_register_energy('write')
        total_energy = num_accesses * register_write_energy

        self.stats.record_memory_transfer("executor_to_register_file", num_accesses, total_energy)
        self.logger.info(f"Stored {data_size} bytes to Register File: {num_accesses} writes, "
                         f"Total energy: {total_energy} nJ.")

        return total_energy

    def get_register_energy(self, operation):
        if operation == 'read':
            return self.register_read_energy
        elif operation == 'write':
            return self.register_write_energy
        else:
            raise ValueError(f"Invalid operation: {operation}")
