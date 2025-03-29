import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_kernel_exists(kernel_path):
    """
    Check if the directory for the specified kernel exists.

    Args:
        kernel_path (str): The full path to the kernel directory.

    Returns:
        bool: True if the kernel directory exists, False otherwise.
    """
    if os.path.isdir(kernel_path):
        logger.info(f"Kernel directory found: {kernel_path}")
        return True
    else:
        logger.error(f"Kernel directory does not exist: {kernel_path}")
        return False


def select_kernel(config, computation, benchmark, dataset_size, batch_size, execution_mode, base_path="/path/to"):
    """
    Select the appropriate kernel and its path based on benchmark, dataset size, batch size, and execution mode.

    Args:
        benchmark (str): The benchmark name (e.g., 'SPLADE', 'ColBERT', etc.).
        dataset_size (str): The dataset size (e.g., '500K', '5M', '50M').
        batch_size (int): The batch size (e.g., 1, 8, 64, 256, 1024).
        execution_mode (str): The mode of execution (e.g., 'train', 'test', 'eval').
        base_path (str): The base directory path where kernels are stored.
    
    Returns:
        tuple: The kernel name and its full directory path.

    Example:
        # To select the training kernel for the SPLADE benchmark with a 500K dataset and batch size of 8:
        kernel, path = select_kernel('SPLADE', '500K', 8, 'train')
        # This would return:
        # kernel = 'SPLADE_500K_Batch8_Train_Kernel'
        # path = '/path/to/SPLADE/500K/train/batch8/kernel'

    Directory and Kernel Naming Structure:
    - Kernels are stored in directories according to the following structure:
      {base_path}/{benchmark}/{dataset_size}/{execution_mode}/batch{batch_size}/kernel

      For example, with `benchmark='SPLADE'`, `dataset_size='500K'`, `batch_size=8`, and `execution_mode='train'`,
      the kernel path will be:
          /path/to/SPLADE/500K/train/batch8/kernel

    - Kernel names follow this naming convention:
      {benchmark}_{dataset_size}_Batch{batch_size}_{ExecutionMode}_Kernel

      So for `benchmark='SPLADE'`, `dataset_size='500K'`, `batch_size=8`, and `execution_mode='train'`,
      the kernel name would be:
          SPLADE_500K_Batch8_Train_Kernel
    """

    # Define all valid options for benchmarks, dataset sizes, batch sizes, and execution modes
    valid_benchmarks = ['splade', 'colbert', 'doc2vec', 'gtr', 'bm25']
    valid_dataset_sizes = ['500K', '5M', '50M', '500M']
    valid_batch_sizes = [1, 8, 64, 256, 1024]
    valid_execution_modes = ['standalone', 'dimension_split']

    # Validate inputs
    if benchmark not in valid_benchmarks:
        logger.error(f"Invalid benchmark '{benchmark}' specified.")
        return 'Default_Kernel', os.path.join(base_path, "default")
    if dataset_size not in valid_dataset_sizes:
        logger.error(f"Invalid dataset size '{dataset_size}' specified.")
        return 'Default_Kernel', os.path.join(base_path, "default")
    if batch_size not in valid_batch_sizes:
        logger.error(f"Invalid batch size '{batch_size}' specified.")
        return 'Default_Kernel', os.path.join(base_path, "default")
    if execution_mode not in valid_execution_modes:
        logger.error(f"Invalid execution mode '{execution_mode}' specified.")
        return 'Default_Kernel', os.path.join(base_path, "default")

    # Construct the kernel name and path
    # kernel_name = f"{benchmark}_{dataset_size}_Batch{batch_size}_{execution_mode.capitalize()}_Kernel"
    if computation == 'embedding':
        kernel_name = config['kernels']['embedding']
    else:
        kernel_name = config['kernels']['scoring']
        
        # Remove the existing batch size by splitting on "_b" and keeping only the first part
        # remove_batch = kernel_name.split('_b')[0]

        # # Extract the remaining part after the batch size
        # remaining_parts = kernel_name.split('_b', 1)[1].split('_', 1)  # Split into [batch_size_and_remainder, remainder]
        # if len(remaining_parts) > 1:
        #     remainder = remaining_parts[1]
        # else:
        #     remainder = ""
        
        # Construct the new kernel name with the specified batch size
        # kernel_name = f"{remove_batch}_b{batch_size}_{remainder}"

    print (f"Kernel name: {kernel_name}")
    kernel_path = os.path.join(base_path, benchmark, dataset_size, execution_mode, f"batch{batch_size}", kernel_name)
    print (f"Kernel path: {kernel_path}")

    # Ensure the directory exists
    # create_kernel_directory(kernel_path)
    check_kernel_exists(kernel_path)

    logger.info(f"Selected kernel: {kernel_name}, Path: {kernel_path}")
    return kernel_name, kernel_path

def create_kernel_directory(path):
    """
    Create the directory if it does not already exist.

    Args:
        path (str): The directory path to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory checked/created at: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

# # Example Usage
# if __name__ == "__main__":
#     kernel, path = select_kernel('SPLADE', '500K', 8, 'train')
#     print(f"Kernel: {kernel}, Path: {path}")
