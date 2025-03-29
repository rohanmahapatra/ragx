class Navigator:
    """Navigator is responsible for fetching data and activating tasks in the pipeline."""
    def __init__(self, env, memory_unit, distributor, config, logger):
        self.env = env
        self.memory_unit = memory_unit
        self.distributor = distributor
        self.config = config
        self.logger = logger
        self.pending_tasks = []

    def fetch_data(self, data_request):
        """Fetches data from memory and sends it to the distributor."""
        data_size = data_request['size']
        
        # Load data from DRAM to scratchpad (SimPy)
        yield self.env.process(self.memory_unit.load_from_dram_to_scratchpad(data_size))
        
        # Send the data to the distributor to schedule on backend
        self.distributor.enqueue_task(data_request)
        self.logger.info(f"Navigator fetched data of size {data_size} and sent to distributor.")

    def manage_data_flow(self, data_requests):
        """Manages multiple data fetching tasks. Can overlap if there are no dependencies."""
        processes = []
        
        for request in data_requests:
            if not self.has_dependency(request):
                process = self.env.process(self.fetch_data(request))
                processes.append(process)
            else:
                # Add to pending if there is a dependency
                self.pending_tasks.append(request)

        yield self.env.all_of(processes)  # Wait for all non-dependent tasks to complete

    def resolve_dependencies(self):
        """Resolve dependencies and activate any tasks that were stalled."""
        for task in list(self.pending_tasks):
            if not self.has_dependency(task):
                self.pending_tasks.remove(task)
                self.env.process(self.fetch_data(task))
                
    def has_dependency(self, data_request):
        """Check if a data request has a dependency."""
        return 'depends_on' in data_request and not data_request['depends_on'].triggered
