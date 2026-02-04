import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet
import pickle
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNetV2(DatasetBase):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

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
        folders = list(classnames.keys())
        items = []
        self.class_index = list(range(1000))
        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=int(label), classname=classname)
                items.append(item)

        return items
