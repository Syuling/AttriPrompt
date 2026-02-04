import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        # if num_shots >= 1:
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
                train, rem_train, val = data["train"], data["rem_train"], data["val"]
        else:
            if num_shots >= 1:
                train, rem_train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            else:
                train, rem_train = self.generate_new_fewshot_dataset(train, num_shots=num_shots,head=head,cont_dis=cont_dis,alpha=alpha,seed=seed)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            data = {"train": train, "rem_train": rem_train, "val": val}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        rem_train, train, val, test = OxfordPets.subsample_classes(rem_train,train, val, test, subsample=subsample)

        super().__init__(rem_train=rem_train,train_x=train, val=val, test=test)
