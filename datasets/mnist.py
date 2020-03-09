r"""
Module: mnist

Defines the handler as well as other related functinality for mnist.
"""

### TODO: finish display info, mean/std, init df from list (utils read list, etc)

import sys, os
from collections import OrderedDict
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from rere.utils import images, files
from .basehandlers import VisionDatasetHandler


class DatasetHandler(VisionDatasetHandler):
    r"""
    Dataset handler for mnist (aka just a df wrapper with extra info)
    """

    def __init__(self, descriptor, name=''):
        ### Descriptor can either be path to mnist or a df
        self.name = 'mnist' if not name else name
        self.classnames, self.classmap = None, None
        if isinstance(descriptor, str):
            if os.path.isdir(descriptor):
                self.df = self._get_dataframe(descriptor)
            elif os.path.isfile(descriptor):
                with open(descriptor, 'rb') as f:
                    self.df = pickle.load(f)
        elif isinstance(descriptor, pd.DataFrame):
            self.df = descriptor
        else:
            raise TypeError(f"descriptor of type({type(descriptor)}) isn't supported")
        if not self.classnames:
            self.update()
        with open('./df.pkl', 'wb') as f:
            pickle.dump(self.df, f)


    def update(self, newdf=None):
        ''' Update self.classnames and optionally self.norm_means/stds'''
        if newdf:
            assert isinstance(newdf, pd.DataFrame)
            self.df = newdf

        self.classnames, self.classmap = [], []
        cid_2_cname = {}
        for idx, row in self.df.iterrows():
            ci, cn = row['label']
            if ci in cid_2_cname:
                assert cn == cid_2_cname[ci]
            else:
                cid_2_cname[ci] = cn
        sortedkeys = sorted(cid_2_cname.keys())
        for k in sortedkeys:
            self.classnames.append(cid_2_cname[k])
            self.classmap.append(k)


    def get_classidx(self, descriptor):
        r"""
        Converts classid or classname (defined in OG dataset) to classidx.
        Makes use of instance's self.classnames & self.classmap
        """
        if isinstance(descriptor, str):
            return self.classnames.index(descriptor)
        elif isinstance(descriptor, int):
            return self.classmap.index(descriptor)
        else:
            raise ValueError(f"descriptor ({descriptor}) type not valid.")

        def get_classid(self, descriptor):
            r"""
            Converts classidx or classname to classid (defined in OG ds).
            Makes use of instance's self.classnames & self.classmap
            """
            if isinstance(descriptor, str):
                classidx = self.classnames.index(descriptor)
                return self.classmap[classidx]
            elif isinstance(descriptor, int):
                return self.classmap[descriptor]
            else:
                raise ValueError(f"descriptor ({descriptor}) type not valid.")


    def get_classname(self, value, is_cidx=True):
        r"""
        Converts classidx or classid to classname.
        Makes use of instance's self.classnames & self.classmap
        """
        assert isinstance(value, int) and value >= 0
        assert value <= len(self.classmap) or value <= max(self.classmap)
        if is_cidx:
            return self.classnames[value]
        else: # is classid
            classidx = self.classmap.index[value]
            return self.classnames[classidx]


    def _get_dataframe(self, ds_dir):
        r"""
        Generates a dataframe based on the dataset lists directory.
        (1) Checks if 'dslistpath/mnist' directory exists
        (2) Creates mnist dataframe based on structure of mnist folder

        This function is called both by the module API to create a vanilla
        MNIST df and by the handler constructor when a df isn't given.
        """
        from rere.utils import files
        assert os.path.isdir(ds_dir)
        subset_dirs = files.list_dirs(ds_dir, subnames=['subset'], fullpath=True)

        df_dict = OrderedDict([ ('id', []),
                                ('name', []),
                                ('size', []),
                                ('subsetid', []),
                                ('subsetname', []),
                                ('data', []),
                                ('label', []),
                                ('tags', []),
                                ('values', [])])
        
        ## Populate df dict
        #  Loop over subset names (e.g. train)
        self.classnames, self.classmap = [], []
        for subset_id, subset_dirname in enumerate(subset_dirs): 
            parts = subset_dirname.split('_')
            assert len(parts) == 2
            subset_name = parts[1].lower()
            subset_path = os.path.join(ds_dir, subset_dirname)

            class_dirnames = files.natural_sort([d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d)) \
                                and d[0] != '.'])
            # loop over classes in subset
            for classid_order, class_dirname in enumerate(class_dirnames):
                parts = class_dirname.split('_')
                assert len(parts) == 2 and isinstance(int(parts[0]), int)
                classid = int(parts[0])
                classname = parts[1]
                self.classnames.append(classname)
                self.classmap.append(classid)

                # no file sort needed because id already defined in fn
                filepaths = [f for f in os.listdir(os.path.join(
                    subset_path, class_dirname)) if os.path.isfile(
                        os.path.join(subset_path, class_dirname, f)) and \
                        os.path.splitext(f)[-1] == '.png']
                for filename in filepaths: # loop over files in class
                    fnparts = filename.split('_')
                    imgid = int(fnparts[0])
                    imgname = fnparts[1][:-4]
                    filepath = os.path.join(subset_path, class_dirname, filename)

                    #* with everything defined, populate dict..
                    df_dict['id'].append(imgid)
                    df_dict['name'].append(imgname)
                    df_dict['size'].append(images.get_dimensions(filepath))
                    df_dict['subsetid'].append(subset_id)
                    df_dict['subsetname'].append(subset_name)
                    df_dict['data'].append(filepath)
                    df_dict['label'].append((classid, classname))
                    df_dict['tags'].append(None)
                    df_dict['values'].append(None)

        df = pd.DataFrame(df_dict).set_index('id').sort_index()
        return df

        
    


# class _MnistDefaultDataset(Dataset):
#     def __init__(self, dataframe, transform=None):
#         self.df = dataframe
#         self.transform = transform
#         self.classes = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
#     def __len__(self):@classmethod
#         return len(self.df)
#     def __getitem__(self, idx):
#         X = Image.open(self.df['path'].iloc[idx])
#         y = self.df.loc[idx,:]
#         # y = torch.tensor(int(self.df['label'].iloc[idx]))
#         if self.transform:
#             X = self.transform(X)
#         return X, y


### ======================================================================== ###
### * ### * ### * ### *         Deprecated Archive       * ### * ### * ### * ### 
### ======================================================================== ###


# class MnistHandler(DatasetHandler):
    
#     name = 'mnist'
#     description = 'Collection of hand written digits.'
#     def __init__(self, dslistpath, df=None):
#         self.dslistpath = dslistpath
#         self.path = os.path.join(dslistpath, 'mnist')
#         assert os.path.isdir(self.dslistpath) and os.path.isdir(self.path)
        
#         self.parentclass = self.__class__

#         if df is not None:  # pd.DataFrame does not support boolean convert
#             assert isinstance(df, pd.DataFrame)
#             if not hasattr(df, 'name'):
#                 df.name = self.parentclass.name
#             if not hasattr(df, 'description'):
#                 df.description = self.parentclass.description
#             self.df = df
#             self.has_custom_df = True
#             self.classnames, self.classmap = self._get_classnames_classmap(self.df)
#         else:
#             ret = self._get_dataframe_from_dslistpath(dslistpath, 
#                     get_class_names_map=True)
#             self.df, self.classnames, self.classmap = ret
#             self.has_custom_df = False
        
#         super(MnistHandler, self).__init__()  # checks implementation