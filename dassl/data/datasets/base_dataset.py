import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import numpy as np

from dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, rem_train=None, train_x=None, train_u=None, val=None, test=None):
        self._rem_train = rem_train
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)
        

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        assert len(source_domains) > 0, "source_domains (list) is empty"
        assert len(target_domains) > 0, "target_domains (list) is empty"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
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

            for label, items in tracker.items():
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

    def get_shot_index(self, head, num_classes, cont_dis=None, alpha=None, seed=None):
        if cont_dis == 1:
            n_max = head
            n_min = 1
            # alpha =0.6
            cls_idx = np.arange(1, num_classes + 1)
            cls_counts_cont = 1 / (cls_idx ** alpha)

            # 归一化到 [1, head]
            cls_counts = (cls_counts_cont - cls_counts_cont.min()) / (cls_counts_cont.max() - cls_counts_cont.min())
            cls_counts = cls_counts * (n_max - n_min) + n_min
            cls_counts = np.round(cls_counts).astype(int)

            # 修正首尾
            cls_counts[0] = n_max
            cls_counts[-1] = n_min

            np.random.seed(seed)
            np.random.shuffle(cls_counts)
            shot_index = cls_counts
            print('类别样本数： ',cls_counts)
                
        else:
            rand_index = np.random.randint(0, 5, size=(num_classes,)) 
            shots = np.array([1,2,4,8]+[head])
            shot_index = shots[rand_index]
            print('类别样本数： ',shot_index)

        return shot_index
    
    def generate_new_fewshot_dataset(
        self, *data_sources, num_shots=-1, head=None, cont_dis=None, alpha=None, seed=None, repeat=False):
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
        # res_output = []
        shot_indexes = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            # remaining_dataset = []  # 用于存储剩余样本
            
            num_classes = len(tracker)
            shot_index = self.get_shot_index(head,num_classes,cont_dis,alpha,seed)
            shot_indexes.append(shot_index)
            for label, items in tracker.items():
                num_shots = shot_index[label]
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                    # remaining_items = [item for item in items if item not in sampled_items]
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                        # remaining_items = items  # 全部样本都保留
                    else:
                        sampled_items = items
                        # remaining_items = []  # 没有剩余样本

                dataset.extend(sampled_items)
                # remaining_dataset.extend(remaining_items)

            output.append(dataset)
            # res_output.append(remaining_dataset)

        if len(output) == 1:
            return output[0], shot_indexes[0]

        return output, shot_indexes

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output
