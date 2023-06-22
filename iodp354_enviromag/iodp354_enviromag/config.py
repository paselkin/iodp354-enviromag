from pathlib import Path  # pathlib is seriously awesome!
import git
import sys

# Change sys.path to import other files
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# Establish base directory for repository 
repo = git.Repo('.', search_parent_directories=True)
    
# Directories for data relative to this one
data_dir = Path(repo.working_tree_dir) / 'data'
raw_dir = data_dir / 'raw'
meta_dir = data_dir / 'metadata'
processed_dir = data_dir / 'processed'
images_dir = data_dir / 'images'

