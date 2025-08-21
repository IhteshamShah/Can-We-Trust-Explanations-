import os
import logging
'''
setting up a single log file outside your project folder and 
let all project files (explainer.py, utils.py, main.py) write to the same log.

'''
# Get parent directory (one step back from project folder)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Log directory
log_dir = os.path.join(parent_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = os.path.join(log_dir, "project.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filemode="a"  # append mode (use "w" to overwrite each run)
)

# Get logger function
def get_logger(name):
    return logging.getLogger(name)
