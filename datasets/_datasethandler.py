
import sys, os
from pathlib import Path
import importlib, importlib.util
sys.dont_write_bytecode = True

import torch, torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms

class DataSetHandler:
    """
    This is the main dataset handler, containing the API with training
    programs from abroad. The idea is to modularize the dataframe
    organization and retrieval part. Afterward, dataset creation is possible
    from customized usage of these dataframes.
    """

    def __init__(self):
        self.filePath = Path(__file__).absolute() # strip single quotes from ends
        self.mainPath = os.path.dirname(self.filePath)
        self.dataset_2_path = {
            'c_ham':          os.path.join(self.mainPath, 'C_HAM10000'),
            'c_tinyimagenet': os.path.join(self.mainPath, 'C_TINY_IMAGENET200'),
            'd_pascal':       os.path.join(self.mainPath, 'D_PASCAL_VOC2012'),
            's_ham' :         os.path.join(self.mainPath, 'S_HAM10000'),
            's_glas':         os.path.join(self.mainPath, 'S_MICCAI2015_GlaS'),
            's_pascal':       os.path.join(self.mainPath, 'S_PASCAL_VOC2012') 
        }
        self.classify_sets = {
            'c_ham',
            'c_tinyimagenet', 
            'd_pascal',
            's_glas' 
        }

    def initialize_dataframes(self, dataset_name, split=False):
        """ Currently only supports: tinyimagenet, pascal, ham, glas """

        # spec = importlib.util.spec_from_file_location("collect.Collect", 
        #                                 self.dataset_2_path[dataset_name])
        # collect = importlib.util.module_from_spec(spec) 
        # spec.loader.exec_module(collect)
        # Collector = collect.Collect()
        # sys.path.append(self.dataset_2_path[dataset_name])

        # from C_TINY_IMAGENET200.collect import Collect
        # Collector = Collect()
        # sys.path.remove(self.dataset_2_path[dataset_name])

        if dataset_name == 'c_ham':
            import C_HAM10000.collect as c_ham
            Collector = c_ham.Collect()
        elif dataset_name == 'c_tinyimagenet':
            import C_TINY_IMAGENET200.collect as c_tinyimagenet
            Collector = c_tinyimagenet.Collect()
        elif dataset_name == 'd_pascal':
            import D_PASCAL_VOC2012.collect as d_pascal
            Collector = d_pascal.Collect()
        elif dataset_name == 's_glas':
            import S_MICCAI2015_GlaS.collect as s_glas
            Collector = s_glas.Collect()
        else: 
            raise NameError(f"Name ({dataset_name}) not a classification dataset.")

        traindf, valdf = Collector.get_trainval_dataframes()
        testdf = Collector.get_test_dataframe()

        return traindf, valdf, testdf


    def initialize_segmentation_dataframe(self, dataset_name):
        pass






