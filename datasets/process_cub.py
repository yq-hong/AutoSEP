import os
import pickle
import random
import copy
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import numpy as np
import PIL.Image
import tarfile
import urllib.request

import torch
from torch.utils.data import Dataset

N_ATTRIBUTES = 312
N_CLASSES = 200


def extract_data(data_dir, train_val_split_ids=None):
    '''
    train_val_split_ids = {
        'train': [ids belonging to the train split],
        'val': [ids belonging to the val split],
    }
    '''
    cwd = os.getcwd()
    data_path = join(cwd, data_dir + '/images')
    val_ratio = 0.2

    path_to_id_map = dict()  # map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])

    attribute_labels_all = ddict(list)  # map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list)  # map from image id to a list of attribute certainties
    # map from image id to a list of attribute labels calibrated for uncertainty
    attribute_uncertain_labels_all = ddict(list)
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4: 1},  # calibrate main label based on uncertainty label
                       0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    is_train_test = dict()  # map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    train_val_data, test_data = [], []

    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()  # sort by class index
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')]
        # classfile_list.sort()
        for cf in classfile_list:
            img_id = path_to_id_map[join(folder_path, cf)]
            img_path = join(folder_path, cf)
            metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                        'attribute_label': attribute_labels_all[img_id],
                        'attribute_certainty': attribute_certainties_all[img_id],
                        'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
            if is_train_test[img_id]:
                train_val_data.append(metadata)
            else:
                test_data.append(metadata)

    train_data, val_data = [], []
    if train_val_split_ids is not None:
        assert 'train' in train_val_split_ids
        assert 'val' in train_val_split_ids
        assert np.array_equal(np.sort([d['id'] for d in train_val_data]),
                              np.sort(train_val_split_ids['train'] + train_val_split_ids['val']))
        for id_ in train_val_split_ids['train']:
            d = [d_ for d_ in train_val_data if d_['id'] == id_][0]
            train_data.append(d)
        for id_ in train_val_split_ids['val']:
            d = [d_ for d_ in train_val_data if d_['id'] == id_][0]
            val_data.append(d)
    else:
        random.shuffle(train_val_data)
        split = int(val_ratio * len(train_val_data))
        train_data = train_val_data[split:]
        val_data = train_val_data[: split]
    print('Size of train set:', len(train_data))
    return train_data, val_data, test_data


def extract_driver(save_dir, data_dir, ref_data_dir=None):
    train_val_split_ids = None
    if ref_data_dir is not None:
        with open(os.path.join(ref_data_dir, 'train.pkl'), 'rb') as f:
            ref_train_data = pickle.load(f)
        with open(os.path.join(ref_data_dir, 'val.pkl'), 'rb') as f:
            ref_val_data = pickle.load(f)
        train_val_split_ids = {'train': [d['id'] for d in ref_train_data],
                               'val': [d['id'] for d in ref_val_data], }
        del ref_train_data, ref_val_data

    train_data, val_data, test_data = extract_data(data_dir, train_val_split_ids)

    for dataset in ['train', 'val', 'test']:
        print("Processing %s set" % dataset)
        with open(os.path.join(save_dir, dataset + '.pkl'), 'wb') as f:
            if 'train' == dataset:
                pickle.dump(train_data, f)
            elif 'val' == dataset:
                pickle.dump(val_data, f)
            else:
                pickle.dump(test_data, f)


def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()


def get_class_attributes_data(min_class_count, out_dir, modify_data_dir='', keep_instance_data=False):

    data = pickle.load(open(os.path.join(modify_data_dir, 'train.pkl'), 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1:  # not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    # check where 0 count = 1 count, set the corresponding class attribute label to be 1
    equal_count = np.where(class_attr_min_label == class_attr_max_label)
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    # select attributes that are present (on a class level) in at least [min_class_count] classes
    mask = np.where(attr_class_count >= min_class_count)[0]
    class_attr_label_masked = class_attr_max_label[:, mask]
    if keep_instance_data:
        collapse_fn = lambda d: list(np.array(d['attribute_label'])[mask])
    else:
        collapse_fn = lambda d: list(class_attr_label_masked[d['class_label'], :])
    create_new_dataset(out_dir, 'attribute_label', collapse_fn, data_dir=modify_data_dir)


class CUBDataset(Dataset):

    def __init__(self, root_dir, split, transform=None, return_attribute=True, download=True,
                 voted_concept_labels=True):
        '''https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
        '''
        assert (split in ['train', 'val', 'test', 'train_test', 'train_val'])

        src_dir = os.path.join(root_dir, 'CUB_200_2011')
        self.images_dir = os.path.join(src_dir, 'images')

        processed_dir = os.path.join(root_dir, 'CUB_processed')
        unvoted_dir = os.path.join(root_dir, 'CUB_unvoted')

        if (not os.path.isdir(self.images_dir)) or (not os.path.isdir(processed_dir)):
            print('Downloading data...')
            self.download(root_dir)

        if not os.path.isdir(unvoted_dir):
            print('Creating unvoted data...')
            self.create_unvoted(root_dir)

        self.meta_data = []
        for split_ in split.split('_'):
            if voted_concept_labels:
                meta_pkl_path = os.path.join(processed_dir, f'class_attr_data_10/{split_}.pkl')
            else:
                meta_pkl_path = os.path.join(unvoted_dir, f'{split_}.pkl')
            with open(meta_pkl_path, 'rb') as f:
                self.meta_data += pickle.load(f)

        self.transform = transform
        self.return_attribute = return_attribute

        with open(os.path.join(src_dir, 'attributes/attributes.txt'), 'r') as f:
            all_attr_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])
        # https://github.com/yewsiang/ConceptBottleneck/blob/a2fd8184ad609bf0fb258c0b1c7a0cc44989f68f/CUB/generate_new_data.py#L65
        selected_attr_indices = np.array([1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40,
                                          44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80,
                                          84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126,
                                          131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168,
                                          172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202, 203,
                                          208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239,
                                          240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
                                          283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311])
        self.attr_names = all_attr_names[selected_attr_indices]

        # load class_names
        with open(os.path.join(src_dir, 'classes.txt'), 'r') as f:
            self.class_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])

    def download(self, root_dir):
        ''' https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/README.md
        '''
        fname_url_pairs = [
            ("CUB_200_2011",
             "https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/contents/blob/"),
            ("CUB_processed",
             "https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/"),
        ]

        for fname, url in fname_url_pairs:
            expand_dir = os.path.join(root_dir, fname)
            tar_path = os.path.join(root_dir, f"{fname}.tar.gz")
            os.makedirs(expand_dir, exist_ok=True)

            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path) as f:
                f.extractall(expand_dir)

            os.remove(tar_path)

    def create_unvoted(self, root_dir, min_class_count=10):
        src_dir = os.path.join(root_dir, 'CUB_200_2011')
        processed_dir = os.path.join(root_dir, 'CUB_processed/class_attr_data_10')
        raw_dir = os.path.join(root_dir, 'CUB_raw')
        unvoted_dir = os.path.join(root_dir, 'CUB_unvoted')

        assert os.path.isdir(src_dir) and os.path.isdir(processed_dir)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(unvoted_dir, exist_ok=True)

        extract_driver(raw_dir, src_dir, ref_data_dir=processed_dir)
        get_class_attributes_data(min_class_count, unvoted_dir, modify_data_dir=raw_dir, keep_instance_data=True)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fname = os.path.join(self.images_dir,
                                 *self.meta_data[idx]['img_path'].split('/')[-2:])
        X = PIL.Image.open(img_fname).convert("RGB")

        if self.return_attribute:
            Y = torch.as_tensor(self.meta_data[idx]['attribute_label'])
        else:
            Y = torch.as_tensor(self.meta_data[idx]['class_label'])

        if self.transform:
            X = self.transform(X)

        return X, Y


if __name__ == '__main__':
    dset_trn = CUBDataset(root_dir="/data/CUB/", split="train", return_attribute=False)
