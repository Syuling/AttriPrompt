import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "annual crop land": "Annual Crop Land",
    "Forest": "Forest",
    "Herbaceous Vegetation Land": "Herbaceous Vegetation Land",
    "Highway or Road": "Highway or Road",
    "Industrial Buildings": "industrial buildings or commercial buildings",
    "Pasture Land": "Pasture Land",
    "Permanent Crop Land": "Permanent Crop Land",
    "Residential Buildings": "residential buildings or homes or apartments",
    "River": "River",
    "Sea or Lake": "Sea or Lake",
}


@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):

    dataset_dir = "eurosat"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        # if num_shots >= 1:
        seed = cfg.SEED
        head = cfg.Head
        cont_dis = cfg.cont_dis
        if cont_dis==1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"cont_dis_shot_{num_shots}-{head}-seed_{seed}.pkl")
        else:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-{head}-seed_{seed}.pkl")
        #8和16对调 seed=1:8 16	1	2	8	1	1	2	16	16 --》16 8	1	2	16	1	1	2	8	8
        
        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, rem_train, val = data["train"], data["rem_train"], data["val"]
        else:
            if num_shots >= 1:
                train, rem_train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            else:
                train, rem_train = self.generate_new_fewshot_dataset(train, num_shots=num_shots,head=head,cont_dis=cont_dis,seed=seed)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            data = {"train": train, "rem_train": rem_train, "val": val}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        rem_train, train, val, test = OxfordPets.subsample_classes(rem_train,train, val, test, subsample=subsample)

        super().__init__(rem_train=rem_train,train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
