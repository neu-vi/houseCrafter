#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT
from pathlib import Path

class Data_Process_Config(object):
    # This class is for data processing and visualization.
    def __init__(self, dataset_name, proj_dir='examples'):
        self.root_path = Path(proj_dir).joinpath('datasets').joinpath(dataset_name)
