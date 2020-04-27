r"""
Module: basehandlers

Defines basehandlers for various dataset handling tasks.
"""

import sys, os
import pandas as pd
import time
from collections import OrderedDict

from PIL import Image

from rere.utils import files, images


### ======================================================================== ###
### * ### * ### * ### *     Default Class Definitions    * ### * ### * ### * ### 
### ======================================================================== ###


class VisionDatasetHandler:
    r"""
    Base DatasetHandler for all vision datasets.
    Assumes: all examples have filename/path(s) as data (str or list).
    """

    def get_norm_mean_stds(self, subsets=[], tags=[]):
        imglist = []
        for _, r in self.df.iterrows():
            if subsets:
                if r['subsetname'] not in subsets:
                    continue
            valid = False if tags else True
            if tags:
                for t in tags:
                    if t in r['tags']:
                        valid = True
                        break
            if valid:
                data = r['data']
                if isinstance(data, list):
                    for item in data:
                        imglist += images._create_img_list(item)
                else:
                    imglist += images._create_img_list(data)
        return images.get_mean_std_norms(imglist)

    @staticmethod
    def get_info(df, disp=True, list=False):
        r"""
        Prints formatted stats (if disp) and returns dict of information.

        Displays:
            Display dataset name, tasks available, train size/labels avail,
            val size/labels avail, test size/labels avail.
            In addition to above: under train, val, test show img sizes as well as
            file types, RGB status, norms
        """
        assert df is not None and isinstance(df, pd.DataFrame)
        
        if list and disp:
            classes = df['classid'].unique()
            print(f"{df.name} - {len(classes)} classes (total: {len(df)})")
            return
        else:
            ### info prep
            info_dict = {
                'name': df.name,
                'description': df.description,
                'size': len(df),
                'classification': True, 
                'segmentation': True,
                'subset_names': [],
            }
            classification_ids = df.classid.unique()
            if len(classification_ids) == 1 and classification_ids[0] is None:
                info_dict['classification'] = False
            segmentation_ids = df.segpath.unique()
            if len(segmentation_ids) == 1 and segmentation_ids[0] is None:
                info_dict['segmentation'] = False
            
            info_dict['subset_names'] = sorted(df.subsetname.unique().tolist(),
                                                reverse=True)
            
            list_subsets = info_dict['subset_names']
            if list_subsets:
                for subsetname in list_subsets:
                    # print(f"  computing info for subset {subsetname}..", end='')
                    subset_df = df[df['subsetname'] == subsetname]
                    subsetinfo_dict = {
                        'name': 'subsetinfo_' + subsetname,
                        'size': len(subset_df),
                        'labelled': True, # to change
                        'filetypes_2_count': None,
                        'imgsizes_2_count': None,
                    }
                    classification_ids = df.classid.unique()
                    has_cls_labels, has_seg_labels = True, True
                    if len(classification_ids) == 1 and classification_ids[0] is None:
                        has_cls_labels = False
                    segmentation_ids = df.segpath.unique()
                    if len(segmentation_ids) == 1 and segmentation_ids[0] is None:
                        has_seg_labels = False
                    subsetinfo_dict['labelled'] = has_cls_labels or has_seg_labels

                    filetypes_2_count = {}
                    imgsizes_2_count = {}
                    for idx, row in subset_df.iterrows():
                        ext = os.path.splitext(row['path'])[1:]

                        # ST = time.time()
                        
                        # imgobj = Image.open(row['path'])
                        # imgchannels = len(imgobj.getbands())
                        # imgw, imh = imgobj.size
                        imgw, imh = images.get_dimensions(row['path'])
                        
                        # print(f"OG time to get imgsize: {time.time()-ST}")
                        # sys.exit()
                        imgsize = (1, imgw, imh)
                        
                        # 0.00631403923034668

                        if ext in filetypes_2_count:
                            filetypes_2_count[ext] += 1
                        else:
                            filetypes_2_count[ext] = 1
                        if imgsize in imgsizes_2_count:
                            imgsizes_2_count[imgsize] += 1
                        else:
                            imgsizes_2_count[imgsize] = 1
                    subsetinfo_dict['filetypes_2_count'] = filetypes_2_count
                    subsetinfo_dict['imgsizes_2_count'] = imgsizes_2_count
                    info_dict[subsetinfo_dict['name']] = subsetinfo_dict
                    # print(f".done")


            ## displaying
            if disp:
                # overall info
                print(f"\n[Overview]\n----------")
                print(f"{info_dict['name']} - {info_dict['description']}")
                print(f"Total examples: {len(df)}")
                print(f"Total subsets: {len(list_subsets)}.. ")
                print(f"\t{list_subsets}\n\n")

                # info for each class
                for i, subsetname in enumerate(info_dict['subset_names']):
                    subsetinfo_dict = info_dict['subsetinfo_' + subsetname]
                    print(f"\nSubset {i+1} - {subsetname}\n- - - - -")
                    print(f"\tSize: {len(df['subsetname'] == subsetname)}, " 
                          f"Has Labels: {subsetinfo_dict['labelled']}")
                    print(f"\tFile Types & Counts..")
                    for ext, cnt in subsetinfo_dict['filetypes_2_count'].items():
                        print(f"\t\t{ext}\t{cnt}")
                    print(f"\tImage Sizes & Counts..")
                    for size, cnt in subsetinfo_dict['imgsizes_2_count'].items():
                        print(f"\t\t{size}\t{cnt}")
                print("=====================================================\n")
            return info_dict


class MultiClassHandler(VisionDatasetHandler):
    
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
        
    
    ### Helper Methods ###
    @staticmethod
    def _get_classnames_classmap(descriptor):
        r"""
        Gets list of classnames with the name mapped to its class idx.
            > descriptor: df or subsetdir (train, test, val in correct format)
        """
        classnames, classmap = [], []
        if isinstance(descriptor, str):
            subsetpath = descriptor
            assert os.path.isdir(subsetpath)
            dirnames = natural_sort([d for d in os.listdir(subsetpath)
                        if os.path.isdir(os.path.join(subsetpath, d))])
            for cidx, dn in enumerate(dirnames):
                parts = dn.split('_')
                assert len(parts) == 2 and parts[0].isdigit() and cidx == int(parts[0])
                assert os.listdir(os.path.join(subsetpath, dn)) # dir populated
                classnames.append(parts[1])
                classmap.append(int(parts[0]))
            return classnames, classmap
        elif isinstance(descriptor, pd.DataFrame):
            df = descriptor
            cid_2_cname = {}
            for idx, row in df.iterrows():
                ci, cn = row['label']
                if ci in cid_2_cname:
                    assert cn == cid_2_cname[ci]
                else:
                    cid_2_cname[ci] = cn
            sortedkeys = sorted(cid_2_cname.keys())
            for k in sortedkeys:
                classnames.append(cid_2_cname[k])
                classmap.append(k)
            print(f"classnames: {classnames}")
            print(f"classmap: {classmap}")
            
            return classnames, classmap
        else:
            raise ValueError(f'descriptor ({descriptor}) type is incorrect.')



### ======================================================================== ###
### * ### * ### * ### *         Deprecated Archive       * ### * ### * ### * ### 
### ======================================================================== ###


# class VisionDFHandler:
#     r"""
#     Base class for all vision dataset handlers of each new dataset. 
#     Checks implementation, defines shared functionality.
#     """

#     ### Check if vital functions/values are implemented ###

#     def __init__(self):
#         """Checks if proper values are initiated. Call at end of constructor."""
#         if not hasattr(self, 'df') or self.df is None:
#             raise NotImplementedError('Attr df is not initialized.')
#         if not hasattr(self, 'has_custom_df'):
#             raise NotImplementedError('Attr has_custom_df does not exist.')
#         if not hasattr(self, 'classnames') or not self.classnames:
#             # classnames maps name to classidx (its index in list)
#             raise NotImplementedError('Attr classnames is not initialized.')
#         if not hasattr(self, 'classmap') or not self.classmap:
#             # classmap maps class id to classidx (its index in list)
#             raise NotImplementedError('Attr classmap is not initialized.')
#         if not hasattr(self, 'name'):
#             raise NotImplementedError('Attr name does not exist.')

#     def get_dataframe(self, descriptor):
#         raise NotImplementedError()



#     ### Implementing Base Functionality ##

#     def get_classidx(self, descriptor):
#         r"""
#         Converts classid or classname (defined in OG dataset) to classidx.
#         Makes use of instance's self.classnames & self.classmap
#         """
#         if isinstance(descriptor, str):
#             return self.classnames.index(descriptor)
#         elif isinstance(descriptor, int):
#             return self.classmap.index(descriptor)
#         else:
#             raise ValueError(f"descriptor ({descriptor}) type not valid.")

#     def get_classid(self, descriptor):
#         r"""
#         Converts classidx or classname to classid (defined in OG ds).
#         Makes use of instance's self.classnames & self.classmap
#         """
#         if isinstance(descriptor, str):
#             classidx = self.classnames.index(descriptor)
#             return self.classmap[classidx]
#         elif isinstance(descriptor, int):
#             return self.classmap[descriptor]
#         else:
#             raise ValueError(f"descriptor ({descriptor}) type not valid.")

#     def get_classname(self, value, is_cidx=True):
#         r"""
#         Converts classidx or classid to classname.
#         Makes use of instance's self.classnames & self.classmap
#         """
#         assert isinstance(value, int) and value >= 0
#         assert value <= len(self.classmap) or value <= max(self.classmap)
#         if is_cidx:
#             return self.classnames[value]
#         else: # is classid
#             classidx = self.classmap.index[value]
#             return self.classnames[classidx]
        
    
#     ### Helper Methods ###
#     @staticmethod
#     def _get_classnames_classmap(descriptor):
#         r"""
#         Gets list of classnames with the name mapped to its class idx.
#             > descriptor: df or subsetdir (train, test, val in correct format)
#         """
#         classnames, classmap = [], []
#         if isinstance(descriptor, str):
#             subsetpath = descriptor
#             assert os.path.isdir(subsetpath)
#             dirnames = natural_sort([d for d in os.listdir(subsetpath)
#                         if os.path.isdir(os.path.join(subsetpath, d))])
#             for cidx, dn in enumerate(dirnames):
#                 parts = dn.split('_')
#                 assert len(parts) == 2 and parts[0].isdigit() and cidx == int(parts[0])
#                 assert os.listdir(os.path.join(subsetpath, dn)) # dir populated
#                 classnames.append(parts[1])
#                 classmap.append(int(parts[0]))
#             return classnames, classmap
#         elif isinstance(descriptor, pd.DataFrame):
#             df = descriptor
#             cid_2_cname = {}
#             for idx, row in df.iterrows():
#                 ci, cn = row['classid'], row['classname']
#                 if ci in cid_2_cname:
#                     assert cn == cid_2_cname[ci]
#                 else:
#                     cid_2_cname[ci] = cn
#             sortedkeys = sorted(cid_2_cname.keys())
#             for k in sortedkeys:
#                 classnames.append(cid_2_cname[k])
#                 classmap.append(k)
#             print(f"classnames: {classnames}")
#             print(f"classmap: {classmap}")
            
#             ST = time.time()
#             # from rere.utils.image import get_image_size
#             # print(get_image_size(df.iloc[0]['path']))
#             import imageio
            
#             print(imageio.imread(df.iloc[0]['path']).shape)
#             print(f"elapsed time for 1 image size fetch: {time.time()-ST}")
#             sys.exit()
            
#             return classnames, classmap
#         else:
#             raise ValueError(f'descriptor ({descriptor}) type is incorrect.')


#     def get_mean_std_norms(self, df=None, dims=None):
#         raise NotImplementedError()

#     def get_dataframe(self):
#         raise NotImplementedError()

#     def get_torch_dataset(self, df, transforms=None):
#         raise NotImplementedError()

#     def get_list_str(self):
#         raise NotImplementedError()

#     def get_info_str(self, detailed=False):
#         raise NotImplementedError()