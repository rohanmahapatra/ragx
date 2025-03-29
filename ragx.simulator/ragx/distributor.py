class Distributor:
    """Distributor handles scheduling and dispatching tasks to backend executors."""
    def __init__(self, env, config, executors, logger):
        self.env = env
        self.config = config
        self.executors = executors
        self.logger = logger
        self.task_queue = simpy.Store(env)
        self.processing_tasks = []
    
    def enqueue_task(self, data_request):
        """Add a new task to the processing queue."""
        self.task_queue.put(data_request)

    def associate_kernel(self, task):
        """Associates a task with the appropriate kernel and dispatches it."""
        kernel = task.get('kernel')
        target_executor = self.select_executor(kernel)
        
        if target_executor:
            self.logger.info(f"Task with kernel '{kernel}' sent to {target_executor.__class__.__name__}")
            process = self.env.process(target_executor.execute(task))
            self.processing_tasks.append(process)
            return process

    def select_executor(self, kernel):
        """Chooses an appropriate backend executor based on the kernel."""
        if kernel == "embedding":
            return self.executors['systolic']
        elif kernel in ["vector_add", "dot_product"]:
            return self.executors['vector']
        elif kernel in ["reduce", "all_reduce", "accumulate"]:
            return self.executors['scalar']
        return None

    def process_tasks(self):
        """Process tasks from the queue as they arrive."""
        while True:
            task = yield self.task_queue.get()  # Get the next task
            process = self.associate_kernel(task)  # Dispatch to appropriate executor
            
            if 'depends_on' in task:
                # If there's a dependency, the navigator must wait for completion
                task['depends_on'] = process