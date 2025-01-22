import os
import pickle
import math
import random
import numpy as np
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        # if num_shots >= 1:
        seed = cfg.SEED
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
        
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, val = data["train"], data["rem_train"], data["val"]
        else:
            
            if num_shots >= 1:
                train= self.generate_fewshot_dataset(train, num_shots=num_shots)
            else:
                imbalanced_shot = [1,2,4,8,16]
                train= self.generate_new_fewshot_dataset(train, num_shots=num_shots,imbalanced_shot=imbalanced_shot)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            data = {"train": train, "val": val}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def read_imagenet_split(filepath, path_prefix):
        '''Read train/val/test split from a json file.'''
        def _convert(items,s):
            '''Convert a list of items to a list of dict.'''
            lst = []
            if s == 'test':
                for impath, label, classname in items:
                    _,img = os.path.split(impath)
                    # print(img)
                    impath = 'val/' + img
                    impath = os.path.join(path_prefix, impath)
                    # print(impath,'=========')
                    item = Datum(impath=impath, label=int(label), classname=classname)
                    # item = {'impath': impath,
                    #         'label': int(label),
                    #         'classname': classname}
                    lst.append(item)
            else:
                for impath, label, classname in items:
                    impath = os.path.join(path_prefix, impath)
                    item = Datum(impath=impath, label=int(label), classname=classname)
                    lst.append(item)
            return lst

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"],'train')
        val = _convert(split["val"],'val')
        test = _convert(split["test"],'test')

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

    def generate_new_fewshot_dataset(
        self, *data_sources, num_shots=-1, imbalanced_shot,repeat=False ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots == 0:
            if len(data_sources) == 1:
                return data_sources[0], data_sources[0]
            return data_sources, data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []
        res_output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            remaining_dataset = []  # 用于存储剩余样本

            rand_index = np.random.randint(0, 5, size=(len(tracker),)) 
            shot_index = imbalanced_shot[rand_index]
            for label, items in tracker.items():
                num_shots = shot_index[label]
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                    remaining_items = [item for item in items if item not in sampled_items]
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                        remaining_items = items  # 全部样本都保留
                    else:
                        sampled_items = items
                        remaining_items = []  # 没有剩余样本

                dataset.extend(sampled_items)
                remaining_dataset.extend(remaining_items)

            output.append(dataset)
            res_output.append(remaining_dataset)

        if len(output) == 1:
            return output[0], res_output[0]

        return output, res_output