import os
import random
import pickle
import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import generator
import api_utils as utils
from autosep.llm_text_compare import predict_with_compare, select_k_from_n_excluding_i

random.seed(42)


class DataProcessor(ABC):
    def __init__(self, data_dir, file_name='', max_threads=1):
        self.data_dir = data_dir
        self.file_name = file_name
        self.max_threads = max_threads

    @abstractmethod
    def evaluate(self, predictor, prompt, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass


def process_example(ex, predictor, pred_prompt, attr=None):
    img_path = ex['img_path']
    pred = predictor.inference(pred_prompt, [img_path], attr)
    return ex, pred, attr


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, exs, pred_prompts=None, attribute_cache=None, model_name='gemini'):
        labels, preds, texts, attributes = [], [], [], []
        if model_name == 'gemini' or model_name == 'gpt4o' or 'sglang' in model_name:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
                if attribute_cache != None:
                    futures = [executor.submit(process_example, ex, predictor, pred_prompts[f'{prompt}'],
                                               attribute_cache[f'{prompt}'][f'{ex}']) for ex in exs]
                else:
                    futures = [executor.submit(process_example, ex, predictor, prompt, None)
                               for ex in exs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                                      total=len(futures), desc='running prediction on examples (parallel)'):
                    ex, pred, attr = future.result()
                    if pred != None:
                        if attribute_cache != None:
                            texts.append(ex['img_path'])
                            attributes.append(attr)
                        else:
                            texts.append(ex['text'])
                        labels.append(ex['label'])
                        preds.append(pred)
        else:
            for ex in tqdm(exs, desc='running prediction on examples (single)'):
                ex, pred, attr = process_example(ex, predictor, pred_prompts[f'{prompt}'],
                                                 attribute_cache[f'{prompt}'][f'{ex}'])
                if pred != None:
                    if attribute_cache != None:
                        texts.append(ex['img_path'])
                        attributes.append(attr)
                    else:
                        texts.append(ex['text'])
                    labels.append(ex['label'])
                    preds.append(pred)

        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds, attributes

    def evaluate(self, predictor, prompt, test_exs, pred_prompts=None, attribute_cache=None, model_name='gemini'):
        while True:
            try:
                f1, texts, labels, preds, attributes = self.run_evaluate(predictor, prompt, test_exs, pred_prompts,
                                                                         attribute_cache, model_name)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds, attributes

    def prepare_samples(self, prompt, test_exs, attribute_cache, n=100):
        labels = []
        preds = []
        texts = []
        attributes = []
        for ex in test_exs[:n]:
            attributes.append(attribute_cache[f'{prompt}'][f'{ex}'])
            texts.append(ex['img_path'])
            labels.append(0)
            preds.append(1)

        return texts, labels, preds, attributes

    def compare_evaluate(self, prompt, exs, attribute_cache, model_name='gemini'):
        false_exs_dict = {}

        for i in range(len(exs)):
            false_idx = select_k_from_n_excluding_i(len(exs), 2, i)  # compare
            false_exs_dict[f'{exs[i]}'] = [exs[idx] for idx in false_idx]

        true_exs, false_exs, preds, true_attrs, false_attrs = [], [], [], [], []
        if model_name == 'gemini' or model_name == 'gpt4o' or 'sglang' in model_name:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [executor.submit(predict_with_compare, true_ex, false_ex, prompt,
                                           attribute_cache[f'{prompt}'], model_name)
                           for true_ex in exs for false_ex in false_exs_dict[f'{true_ex}']]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                                      total=len(futures), desc='running comparison on examples (parallel)'):
                    answer, true_ex, false_ex, prompt = future.result()
                    true_exs.append(true_ex)
                    false_exs.append(false_ex)
                    preds.append(answer)
                    true_attrs.append(attribute_cache[f'{prompt}'][f'{true_ex}'])
                    false_attrs.append(attribute_cache[f'{prompt}'][f'{false_ex}'])
        else:
            for true_ex in tqdm(exs, desc='running comparison on examples (single)'):
                for false_ex in false_exs_dict[f'{true_ex}'][:3]:
                    answer, true_ex, false_ex, prompt = predict_with_compare(true_ex, false_ex, prompt,
                                                                             attribute_cache[f'{prompt}'], model_name)
                    true_exs.append(true_ex)
                    false_exs.append(false_ex)
                    preds.append(answer)
                    true_attrs.append(attribute_cache[f'{prompt}'][f'{true_ex}'])
                    false_attrs.append(attribute_cache[f'{prompt}'][f'{false_ex}'])

        return true_exs, false_exs, preds, true_attrs, false_attrs


class MyClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return MyClassificationTask.categories[pred]


class iNaturalistMultiTask(MyClassificationTask):
    def __init__(self, data_dir, file_name, max_threads, args):
        super().__init__(data_dir, file_name, max_threads)
        if 'butterfly' in args.task_name:
            self.num2classname = ['Symbrenthia lilaea', 'Claudina Crescent', 'Elada Checkerspot']
            self.directories = ['01976_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Symbrenthia_lilaea',
                                '01978_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Tegosa_claudina',
                                '01979_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Texola_elada']
        elif 'grass' in args.task_name:
            self.num2classname = ['Arctic Lupine (Lupinus arcticus)',
                                  'Silvery Lupine (Lupinus argenteus)',
                                  'Arizona Lupine (Lupinus arizonicus)']
            self.directories = [
                '07999_Plantae_Tracheophyta_Magnoliopsida_Fabales_Fabaceae_Lupinus_arcticus',
                '08000_Plantae_Tracheophyta_Magnoliopsida_Fabales_Fabaceae_Lupinus_argenteus',
                '08001_Plantae_Tracheophyta_Magnoliopsida_Fabales_Fabaceae_Lupinus_arizonicus']
        else:
            raise Exception(f'Unsupported subcategories: {args.task_name}')

    def stringify_prediction(self, pred):
        return self.num2classname[pred]

    def get_examples(self, mode='train'):
        directories = [f'{self.data_dir}/iNaturalist/{mode}/' + direct for direct in self.directories]

        img_paths, labels = [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            img_paths += paths
            labels += [i for _ in range(len(paths))]

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{i}', 'label': labels[i], 'label_name': self.stringify_prediction(labels[i]),
                        'img_path': img_paths[i]})
        return exs

    def get_few_shot_examples(self, n_shots=1, seed=42):
        random.seed(seed)
        directories = [f'{self.data_dir}/iNaturalist/train/' + direct for direct in self.directories]

        img_paths = []
        exs = [[] for _ in range(3)]
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            idxs = [len(img_paths) + j for j in range(len(paths))]
            img_paths += paths

            select = random.sample(range(len(paths)), n_shots)
            for k in select:
                exs[i].append({'id': f'train-{idxs[k]}', 'label': i,
                               'label_name': self.stringify_prediction(i), 'img_path': paths[k]})
        return exs

    def get_even_exs(self, mode='train', n_exs=10):
        directories = [f'{self.data_dir}/iNaturalist/{mode}/' + direct for direct in self.directories]

        count = 0
        img_paths, labels, idxs = [], [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            idxs += [count + j for j in range(len(paths[:n_exs]))]
            img_paths += paths[:n_exs]
            labels += [i for _ in range(len(paths[:n_exs]))]
            count += len(paths)

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{idxs[i]}', 'label': labels[i],
                        'label_name': self.stringify_prediction(labels[i]), 'img_path': img_paths[i]})
        return exs

    def get_attr(self, mode, prompt, exs, gpt_generator=None, generate=False, exp=1):
        if generate:
            attrs = {}
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache, self.max_threads)
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{self.data_dir}/../autosep/results/{exp}_iNat_grass/{exp}_{mode}_attr.json', 'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs


class CUBMultiTask(MyClassificationTask):
    def __init__(self, data_dir, file_name, max_threads, args):
        super().__init__(data_dir, file_name, max_threads)
        if 'cuckoo' in args.task_name:
            self.num2classname = ['Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo']
            self.name2label = {'black billed cuckoo': 0, 'mangrove cuckoo': 1, 'yellow billed cuckoo': 2}
        elif 'vireo' in args.task_name:
            self.num2classname = ['Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo']
            self.name2label = {'philadelphia vireo': 0, 'red eyed vireo': 1, 'warbling vireo': 2}
        elif 'oriole' in args.task_name:
            self.num2classname = ['Hooded Oriole', 'Orchard Oriole', 'Scott Oriole']
            self.name2label = {'hooded oriole': 0, 'orchard oriole': 1, 'scott oriole': 2}
        else:
            raise Exception(f'Unsupported subcategories: {args.task_name}')

    def stringify_prediction(self, pred):
        return self.num2classname[pred]

    def get_examples(self, mode='train'):
        meta_data = []
        meta_pkl_path = os.path.join(self.data_dir, f'CUB/CUB_raw/{mode}.pkl')
        with open(meta_pkl_path, 'rb') as f:
            meta_data += pickle.load(f)

        exs = []
        for i in range(len(meta_data)):
            img_path = meta_data[i]['img_path']
            cls_name = img_path.split('/')[-2].split('.')[1].replace('_', ' ').strip()
            if cls_name in self.num2classname:
                label = self.name2label[utils.clean_text(cls_name)]
                exs.append({'id': f'{mode}-{i}', 'label': label,
                            'label_name': self.stringify_prediction(label), 'img_path': img_path})
        return exs

    def get_few_shot_examples(self, n_shots=1, seed=42):
        random.seed(seed)
        exs = [[] for _ in range(3)]
        idxs = [[] for _ in range(3)]
        select = [[] for _ in range(3)]

        meta_data = []
        meta_pkl_path = os.path.join(self.data_dir, f'CUB/CUB_raw/train.pkl')
        with open(meta_pkl_path, 'rb') as f:
            meta_data += pickle.load(f)

        for i in range(len(meta_data)):
            img_path = meta_data[i]['img_path']
            cls_name = img_path.split('/')[-2].split('.')[1].replace('_', ' ').strip()
            if cls_name in self.num2classname:
                label = self.name2label[utils.clean_text(cls_name)]
                idxs[label].append(i)

        for i in range(3):
            select[i] = random.sample(idxs[i], n_shots)
            for idx in select[i]:
                img_path = meta_data[idx]['img_path']
                cls_name = img_path.split('/')[-2].split('.')[1].replace('_', ' ').strip()
                label = self.name2label[utils.clean_text(cls_name)]
                exs[i].append({'id': f'train-{idx}', 'label': label,
                               'label_name': self.stringify_prediction(label), 'img_path': img_path})
        return exs

    def get_even_exs(self, mode='train', n_exs=10):
        meta_data = []
        meta_pkl_path = os.path.join(self.data_dir, f'CUB/CUB_raw/{mode}.pkl')
        with open(meta_pkl_path, 'rb') as f:
            meta_data += pickle.load(f)

        exs = []
        counts = [0 for _ in range(3)]
        for i in range(len(meta_data)):
            img_path = meta_data[i]['img_path']
            cls_name = img_path.split('/')[-2].split('.')[1].replace('_', ' ').strip()
            if cls_name in self.num2classname:
                label = self.name2label[utils.clean_text(cls_name)]
                if counts[label] < n_exs:
                    exs.append({'id': f'{mode}-{i}', 'label': label,
                                'label_name': self.stringify_prediction(label), 'img_path': img_path})
                    counts[label] += 1
        return exs

    def get_meta_exs(self, mode='train', n_exs=10):
        meta_data = []
        meta_pkl_path = os.path.join(self.data_dir, f'CUB/CUB_raw/{mode}.pkl')
        with open(meta_pkl_path, 'rb') as f:
            meta_data += pickle.load(f)

        exs = []
        counts = [0 for _ in range(3)]
        for i in range(len(meta_data)):
            img_path = meta_data[i]['img_path']
            cls_name = img_path.split('/')[-2].split('.')[1].replace('_', ' ').strip()
            if cls_name in self.num2classname:
                label = self.name2label[utils.clean_text(cls_name)]
                if counts[label] < n_exs:
                    exs.append({'id': f'{mode}-{i}', 'label': label,
                                'label_name': self.stringify_prediction(label), 'img_path': img_path,
                                'attributes': meta_data[i]['attribute_label']})
                    counts[label] += 1
        return exs

    def get_attr(self, mode, prompt, exs, gpt_generator=None, generate=False, exp=1):
        if generate:
            attrs = {}
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache, self.max_threads)
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{self.data_dir}/../autosep/results/{exp}_CUB_cuckoo/{exp}_{mode}_attr.json', 'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs


class StanfordDogMultiTask(MyClassificationTask):
    def __init__(self, data_dir, file_name, max_threads, args):
        super().__init__(data_dir, file_name, max_threads)
        if 'terrier' in args.task_name:
            self.num2classname = ['Lakeland Terrier', 'Norwich Terrier', 'Cairn Terrier']
            self.directories = ['n02095570-Lakeland_terrier', 'n02094258-Norwich_terrier', 'n02096177-cairn']
        else:
            raise Exception(f'Unsupported subcategories: {args.task_name}')

    def stringify_prediction(self, pred):
        return self.num2classname[pred]

    def get_examples(self, mode='train'):
        directories = [f'{self.data_dir}/Stanford_dogs/images/{mode}/' + direct for direct in self.directories]

        img_paths, labels = [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            img_paths += paths
            labels += [i for _ in range(len(paths))]

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{i}', 'label': labels[i], 'label_name': self.stringify_prediction(labels[i]),
                        'img_path': img_paths[i]})
        return exs

    def get_few_shot_examples(self, n_shots=1, seed=42):
        random.seed(seed)
        directories = [f'{self.data_dir}/Stanford_dogs/images/train/' + direct for direct in self.directories]

        img_paths = []
        exs = [[] for _ in range(3)]
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            idxs = [len(img_paths) + j for j in range(len(paths))]
            img_paths += paths

            select = random.sample(range(len(paths)), n_shots)
            for k in select:
                exs[i].append({'id': f'train-{idxs[k]}', 'label': i,
                               'label_name': self.stringify_prediction(i), 'img_path': paths[k]})
        return exs

    def get_even_exs(self, mode='train', n_exs=10):
        directories = [f'{self.data_dir}/Stanford_dogs/images/{mode}/' + direct for direct in self.directories]

        count = 0
        img_paths, labels, idxs = [], [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            paths = [os.path.join(directories[i], f) for f in file_names]
            idxs += [count + j for j in range(len(paths[:n_exs]))]
            img_paths += paths[:n_exs]
            labels += [i for _ in range(len(paths[:n_exs]))]
            count += len(paths)

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{idxs[i]}', 'label': labels[i],
                        'label_name': self.stringify_prediction(labels[i]), 'img_path': img_paths[i]})
        return exs

    def get_attr(self, mode, prompt, exs, gpt_generator=None, generate=False, exp=1):
        if generate:
            attrs = {}
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache,
                                                          self.max_threads)
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{self.data_dir}/../autosep/results/{exp}_Stanford_terrier/{exp}_{mode}_attr.json',
                      'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs


class VegFruMultiTask(MyClassificationTask):
    def __init__(self, data_dir, file_name, max_threads, args):
        super().__init__(data_dir, file_name, max_threads)
        self.train_ratio = args.train_ratio

        if '1' in args.task_name:
            self.num2classname = ["Dandelion", "Shepherd's purse", "Prickly lettuce"]
            self.directories = ["dandelion", "shepherd's_purse", "prickly_lettuce"]
            self.filtered = {"dandelion": [], }
        elif '2' in args.task_name:
            self.num2classname = ["Leek", "Green Chinese onion", "Bunching onion"]
            self.directories = ["leek", "green_Chinese_onion", "bunching_onion"]
            self.filtered = {}
        else:
            raise Exception(f'Unsupported subcategories: {args.task_name}')

    def stringify_prediction(self, pred):
        return self.num2classname[pred]

    def get_examples(self, mode='train'):
        directories = [f'{self.data_dir}/vegfru/veg200_images/' + direct for direct in self.directories]

        img_paths, labels = [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            full_paths = [os.path.join(directories[i], f) for f in file_names]
            exclude_indices = self.filtered.get(self.directories[i], [])
            filtered_paths = [name for name in full_paths if
                              int(name.split("/")[-1].split(".")[0].split("_")[-1]) not in exclude_indices]

            # Split by index
            split_idx = int(len(filtered_paths) * self.train_ratio)
            if mode == "train":
                paths = filtered_paths[:split_idx]
            else:
                paths = filtered_paths[split_idx:]
            img_paths += paths
            labels += [i for _ in range(len(paths))]

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{i}', 'label': labels[i], 'label_name': self.stringify_prediction(labels[i]),
                        'img_path': img_paths[i]})

        return exs

    def get_few_shot_examples(self, n_shots=1, seed=42):
        random.seed(seed)
        directories = [f'{self.data_dir}/vegfru/veg200_images/' + direct for direct in self.directories]

        img_paths = []
        exs = [[] for _ in range(3)]
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            full_paths = [os.path.join(directories[i], f) for f in file_names]
            exclude_indices = self.filtered.get(self.directories[i], [])
            filtered_paths = [name for name in full_paths if
                              int(name.split("/")[-1].split(".")[0].split("_")[-1]) not in exclude_indices]

            split_idx = int(len(filtered_paths) * self.train_ratio)
            paths = filtered_paths[:split_idx]
            idxs = [len(img_paths) + j for j in range(len(paths))]
            img_paths += paths

            select = random.sample(range(len(paths)), n_shots)
            for k in select:
                exs[i].append({'id': f'train-{idxs[k]}', 'label': i,
                               'label_name': self.stringify_prediction(i), 'img_path': paths[k]})
        return exs

    def get_even_exs(self, mode='train', n_exs=10):
        directories = [f'{self.data_dir}/vegfru/veg200_images/' + direct for direct in self.directories]
        count = 0
        img_paths, labels, idxs = [], [], []
        for i in range(len(directories)):
            file_names = os.listdir(directories[i])
            full_paths = [os.path.join(directories[i], f) for f in file_names]
            exclude_indices = self.filtered.get(self.directories[i], [])
            filtered_paths = [name for name in full_paths if
                              int(name.split("/")[-1].split(".")[0].split("_")[-1]) not in exclude_indices]
            split_idx = int(len(filtered_paths) * self.train_ratio)
            if mode == "train":
                paths = filtered_paths[:split_idx]
            else:
                paths = filtered_paths[split_idx:]
            idxs += [count + j for j in range(len(paths[:n_exs]))]
            img_paths += paths[:n_exs]
            labels += [i for _ in range(len(paths[:n_exs]))]
            count += len(paths)

        exs = []
        for i in range(len(img_paths)):
            exs.append({'id': f'{mode}-{idxs[i]}', 'label': labels[i],
                        'label_name': self.stringify_prediction(labels[i]), 'img_path': img_paths[i]})
        return exs

    def get_attr(self, mode, prompt, exs, gpt_generator=None, generate=False, exp=1):
        if generate:
            attrs = {}
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache, self.max_threads)
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{self.data_dir}/../autosep/results/{exp}_vegfru/{exp}_{mode}_attr.json', 'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs
