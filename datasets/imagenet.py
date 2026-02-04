import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_path = os.path.join(self.dataset_dir, "split_ImageNet.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_imagenet_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # if os.path.exists(self.preprocessed):
        #     with open(self.preprocessed, "rb") as f:
        #         preprocessed = pickle.load(f)
        #         train = preprocessed["train"]
        #         test = preprocessed["test"]
        # else:
        #     text_file = os.path.join(self.dataset_dir, "classnames.txt")
        #     classnames = self.read_classnames(text_file)
        #     train = self.read_data(classnames, "train")
        #     # Follow standard practice to perform evaluation on the val set
        #     # Also used as the val set (so evaluate the last-step model)
        #     test = self.read_data(classnames, "val")

        #     preprocessed = {"train": train, "test": test}
        #     with open(self.preprocessed, "wb") as f:
        #         pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        
        seed = cfg.SEED
        head = cfg.Head
        cont_dis = cfg.cont_dis
        
        alpha = cfg.alpha
     
        if cont_dis==1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"cont_dis_{alpha}_shot_{num_shots}-{head}-seed_{seed}.pkl")
        else:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-{head}-seed_{seed}.pkl")
        
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, rem_train = data["train"], data["rem_train"]
        else:
            if num_shots >= 1:
                train, rem_train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            else:
                train, rem_train = self.generate_new_fewshot_dataset(train, num_shots=num_shots,head=head,cont_dis=cont_dis,alpha=alpha,seed=seed)

            data = {"train": train, "rem_train": rem_train}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        rem_train, train, test = OxfordPets.subsample_classes(rem_train,train, test, subsample=subsample)

        super().__init__(rem_train=rem_train,train_x=train, val=test, test=test)

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

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
