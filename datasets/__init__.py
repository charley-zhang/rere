r"""
Module: __init__ for datasets package

Defines API for datasets package, global dataset variables.
"""

import sys
import os
from pathlib import Path


# Visible default handlers
from . import mnist
# TODO from . import glas
# TODO from . import ham10k



### ======================================================================== ###
### * ### * ### * ### *     Datasets API Definitions     * ### * ### * ### * ### 
### ======================================================================== ###

# API list
__all__ = [
    'set_datasets_path', 'get_list',
    'get_dataset',
]

# Initialize constants and values
DATASETS_PATH = os.path.join(os.path.dirname(Path(__file__).absolute()), 
                             'data')
DATASETNAME_2_MODULE = {
    'mnist': mnist,
#    'glas': glas,
}


def set_datasets_path(path):
    r"""
    Changes the datasets path that will be used in functionality to retrieve
    data. Update before getting any datahandler objects.
    """
    assert os.path.exists(path), f"Path does not exist: ({path})"
    
    global DATASETS_PATH 
    DATASETS_PATH = path 
    print(f"Changing dataset path to: ({path})..")
    get_list()


def get_list(disp=True):
    r""" 
    Returns and optionally. prints a list of datasets available and basic information.
    """
    dataset_names = _get_available_dataset_names()
    if disp: 
        print("\nAvailable Datasets:")
        # print(f"path = ({DATASETS_PATH})")
        if not dataset_names:
            print(f" No datasets in set directory ({DATASETS_PATH}).")
        for dn in dataset_names:
            print(f" > {dn}")
            # DATASETNAME_2_MODULE[dn].get_info(DATASETS_PATH, disp=disp, list=True)
        print()
    return dataset_names


def get_dataset(dataset_name, path='', name=''):
    assert dataset_name in DATASETNAME_2_MODULE
    if not path:
        path = os.path.join(DATASETS_PATH, dataset_name)
    return DATASETNAME_2_MODULE[dataset_name].DatasetHandler(path, name=name)




### ======================================================================== ###
### * ### * ### * ### *    Private Helper Definitions    * ### * ### * ### * ### 
### ======================================================================== ###

def _get_available_dataset_names(path=None):
    dataset_names = []
    path = DATASETS_PATH if not path else path
    for pn in [f for f in os.listdir(path) if os.path.isdir(
                os.path.join(path, f))]:
        if pn in DATASETNAME_2_MODULE.keys():
            dataset_names.append(pn)
    return dataset_names




""" Deprecated.

def get_info(dataset_name, disp=True):
    ''' 
    Returns and optionally prints a list of datasets available and its information.

    Parameters:
    dataset_name - name of dataset
    print - boolean value, if True prints to console
    detailed - in addition to counts/labels, displays img sizes, file types
    '''

    '''
    Basic:
        Display dataset name, tasks available, train size/labels avail,
        val size/labels avail, test size/labels avail.
    Detailed:
        In addition to above: under train, val, test show img sizes as well as
        file types, RGB status, norms
    '''
    assert dataset_name.lower() in DATASETNAME_2_MODULE.keys()
    ret_dict = DATASETNAME_2_MODULE[dataset_name].get_info(DATASETS_PATH,
                                                          disp=disp,
                                                          list=False)
    return ret_dict

def get_dataframe(dataset_name):
    f'''
    # When called, returns a new dataframe object for the dataset given.
    '''
    assert dataset_name.lower() in DATASETNAME_2_MODULE.keys()
    return DATASETNAME_2_MODULE[dataset_name].get_dataframe(DATASETS_PATH)


def get_dataset_handler(dataset_name, df=None):
    f'''
    # When called, returns a new handler object for the dataset given.
    # With the handler comes with its static info methods as well as new df.
    '''
    assert dataset_name.lower() in DATASETNAME_2_MODULE.keys()
    if df is not None:
        return DATASETNAME_2_MODULE[dataset_name].get_dataset_handler(
            DATASETS_PATH, df=df)
    return DATASETNAME_2_MODULE[dataset_name].get_dataset_handler(DATASETS_PATH)
"""