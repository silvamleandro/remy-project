# get_bagfile.py

# Imports
from pathlib import Path

import constants # constants.py
import os
import sys


# Reindex broken bag file
os.system(f"rosbag reindex {str(Path.home())}/{constants.PROJECT_NAME}/data/{sys.argv[1]}.bag.active")
# Remove generated backup file
os.system(f"rm {str(Path.home())}/{constants.PROJECT_NAME}/data/{sys.argv[1]}.bag.orig.active")
# Rename file after reindex to .bag
os.system(f"mv {str(Path.home())}/{constants.PROJECT_NAME}/data/{sys.argv[1]}.bag.active {str(Path.home())}/{constants.PROJECT_NAME}/data/{sys.argv[1]}.bag")
