import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet
import pickle
from .oxford_pets import OxfordPets
import numpy as np

TO_BE_IGNORED = ["README.txt"]


@DATASET_REGISTRY.register()
class ImageNetR(DatasetBase):
    """ImageNet-R(endition).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-rendition"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)
        
        dataset = os.path.join(root, 'imagenet')
        split_fewshot_dir = os.path.join(dataset, "split_fewshot")
        
        head = 16
        preprocessed = os.path.join(split_fewshot_dir, f"shot_{cfg.DATASET.NUM_SHOTS}-{head}-seed_{cfg.SEED}.pkl")
        with open(preprocessed, "rb") as file:
            da = pickle.load(file)
            train = da["train"]

        # train = OxfordPets.subsample_classes(train, subsample=cfg.DATASET.SUBSAMPLE_CLASSES)

        super().__init__(train_x=train, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []
        cls_names = np.array(list(classnames.keys()))
        self.class_index = [np.where(cls_names == value)[0][0] for value in folders]
        # print(len(self.class_index))
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            # label = np.where(cls_names == folder)[0][0]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
