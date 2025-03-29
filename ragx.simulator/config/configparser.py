import yaml

class ConfigParser:
    def __init__(self):
        self.config = None

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # print("Config type:", type(self.config))  # Should be <class 'dict'>
        # print("Config content:", self.config)  # Check actual content
        return self.config

    def parse_config(self):
        """
        Parse and print the configuration settings.
        """
        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        print("**** Configs ****\n")
        print("Mode:", self.config.get('mode', ''))
        print("Distance Metric:", self.config.get('distance_metric', ''))

        # Parse Genesys config
        genesys_config = self.config.get('genesys', {})
        print("\nGenesys Configuration:")
        print("  Size N:", genesys_config.get('size_n', ''))
        print("  Size M:", genesys_config.get('size_m', ''))
        print("  Memory Size:", genesys_config.get('memory_size', ''))
        print("  Memory Bandwidth:", genesys_config.get('memory_bandwidth', ''))

        # Parse Vector Processor config
        vector_processor_config = self.config.get('vector_processor', {})
        print("\nVector Processor Configuration:")
        print("  Number of Processors:", vector_processor_config.get('num_processors', ''))
        print("  Number of Lanes:", vector_processor_config.get('num_lanes', ''))
        print("  Memory Size:", vector_processor_config.get('memory_size', ''))

        # Parse Scalar Engine config
        scalar_unit_config = self.config.get('scalar_unit', {})
        print("\nScalar Engine Configuration:")
        print("  Number of Engines:", scalar_unit_config.get('num_engines', ''))
        print("  Energy per Cycle:", scalar_unit_config.get('energy_per_cycle', ''))

        # Parse EurekaStoreSim config
        eurekastoresim_config = self.config.get('eurekastoresim', {})
        print("\nEurekaStoreSim Configuration:")
        print("  Total Memory Bandwidth:", eurekastoresim_config.get('total_memory_bandwidth', ''))
        print("\n")