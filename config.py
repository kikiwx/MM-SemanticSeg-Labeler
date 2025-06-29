import os

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Data Paths
JSON_PATH = "D:/dataset/test_images/grasp_test_annotation.json"  # Path to your JSON annotation file
IMAGE_ROOT = "D:/dataset/test_images"                             # Root directory for images

# Output Directories
RESULTS_DIR = "results"             # Directory to save segmentation results
GOOD_EXAMPLES_DIR = "good_examples" # Directory to save good examples
BAD_EXAMPLES_DIR = "bad_examples"   # Directory to save bad examples

# Application Settings
POINT_HISTORY_LIMIT = 100 # Maximum number of undo/redo steps for points

# Gradio Server Settings
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7861
SHOW_API = False
SHOW_ERROR = False
SHARE = False


