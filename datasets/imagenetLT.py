import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNetLT(DatasetBase):

    dataset_dir = "imagenetlt"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, "imagenet")
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.lt_dir = os.path.join(self.dataset_dir, "ImageNet_LT")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        preprocessed = os.path.join(self.split_fewshot_dir, "imagenet-lt.pkl")

        if os.path.exists(preprocessed):
            print(f"Loading preprocessed long-tailed data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, test= data["train"], data["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "test")

            data = {"train": train, "test": test}
            with open(preprocessed, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split):
        
        items = []
        text_file = os.path.join(self.lt_dir, f"ImageNet_LT_{split}.txt")

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                if split == 'test':
                    line[0] = line[0].replace("val/", "val2/")
                impath = os.path.join(self.image_dir, line[0])
                item = Datum(impath=impath, label=int(line[1]), classname=classnames[line[0].split('/')[1]])
                items.append(item)

        return items
