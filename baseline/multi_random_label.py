import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
import random
import datetime
import argparse
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import api_utils as utils
from get_utils import get_predictor, get_task_class


def get_class_options(args):
    if args.task_name == 'CUB_cuckoo':
        text = "Your task is to classify the image to three birds: A. Black-billed Cuckoo, B. Mangrove Cuckoo, C. Yellow-billed Cuckoo.\n"
        # choices = ["A. Black-billed Cuckoo", "B. Mangrove Cuckoo", "C. Yellow-billed Cuckoo"]
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'CUB_oriole':
        text = "Your task is to classify the image to three birds: A. Hooded Oriole, B. Orchard Oriole, C. Scott Oriole.\n"
        # choices = ["A. Hooded Oriole", "B. Orchard Oriole", "C. Scott Oriole"]
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'CUB_vireo':
        text = "Your task is to classify the image to three birds: A. Philadelphia Vireo, B. Red-eyed Vireo, C. Warbling Vireo.\n"
        # choices = ["A. Philadelphia Vireo", "B. Red-eyed Vireo", "C. Warbling Vireo"]
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'iNat_butterfly':
        text = "Your task is to classify the image to three butterflies: A. Symbrenthia lilaea, B. Claudina Crescent, C. Elada Checkerspot.\n"
        # choices = ["A. Symbrenthia lilaea", "B. Claudina Crescent", "C. Elada Checkerspot"]
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'iNat_grass':
        text = "Your task is to classify the image to three plants: A. Arctic Lupine (Lupinus arcticus), B. Silvery Lupine (Lupinus argenteus), C. Arizona Lupine (Lupinus arizonicus).\n"
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'Stanford_terrier':
        text = "Your task is to classify the image to three dogs: A. Lakeland Terrier, B. Norwich Terrier, C. Cairn Terrier.\n"
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'vegfru_greens':
        text = "Your task is to classify the image to three vegetables: A. Dandelion, B. Shepherd's purse, C. Prickly lettuce.\n"
        choices = ["A", "B", "C"]
        return text, choices
    elif args.task_name == 'vegfru_allium':
        text = "Your task is to classify the image to three vegetables: A. Leek, B. Green Chinese onion, C. Bunching onion.\n"
        choices = ["A", "B", "C"]
        return text, choices
    else:
        raise Exception(f'Unsupported task: {args.task_name}')


def process_example(ex, predictor, prompt, example_paths):
    img_path = ex['img_path']

    if example_paths != None:
        if args.model == 'gemini':
            paths = [img_path]
            pred = predictor.inference(prompt, paths, few_shot_files=example_paths)
        else:
            paths = example_paths + [img_path]
            pred = predictor.inference(prompt, paths)
    else:
        raise Exception(f'Unsupported input for the model.')

    return ex, pred


def run_evaluate(predictor, prompt, exs, few_shot_paths):
    ids = []
    labels = []
    preds = []
    img_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_example, ex, predictor, prompt, few_shot_paths) for ex in exs]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                              total=len(futures), desc='running prediction on examples'):
            ex, pred = future.result()
            if pred != None:
                img_paths.append(ex['img_path'])
                labels.append(ex['label'])
                preds.append(pred)
                ids.append(ex['id'])
            else:
                print(f"No prediction for {ex['id']}\t{ex['img_path']}")
                with open(args.out, 'a') as outf:
                    outf.write(f"No prediction for {ex['id']}\t{ex['img_path']}\n")

    correct_count = sum(1 for a, b in zip(labels, preds) if a == b)
    accuracy = correct_count / len(exs)
    # accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    conf_matrix = confusion_matrix(labels, preds)
    return f1, accuracy, conf_matrix, img_paths, labels, preds, ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='CUB_cuckoo',
                        choices=['iNat_butterfly', 'iNat_grass', 'Stanford_terrier',
                                 'CUB_cuckoo', 'CUB_oriole', 'CUB_vireo', 'vegfru_greens', 'vegfru_allium'])
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--random_labels', action='store_true', default=False,
                        help="random label or multiple images")
    parser.add_argument('--n_test', default=100, type=int)
    parser.add_argument('--n_examples', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--max_threads', default=8, type=int)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    if args.random_labels:
        args.out = f'results/{args.model}/random_labels/multi_random_labels_{args.task_name}_n{args.n_examples}_seed{args.seed}.txt'
    else:
        args.out = f'results/{args.model}/random_labels/multi_img_example_{args.task_name}_n{args.n_examples}_seed{args.seed}.txt'
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')
    task = get_task_class(args)
    gpt4 = get_predictor(configs)
    if args.model == 'gemini1':
        utils.clear_gemini_img_files(True)

    exs = task.get_even_exs(args.mode, args.n_test)

    exs_train = task.get_examples(args.mode)
    select = random.sample(range(len(exs_train)), args.n_examples)
    random_shot_paths = []
    with open(args.out, 'a') as outf:
        outf.write(f'Few shot examples-------------------------\n')
        for idx in select:
            outf.write(f"{exs_train[idx]['img_path']}\n")
            random_shot_paths.append(exs_train[idx]['img_path'])

    if args.model == 'gemini':
        file_list, _ = utils.get_gemini_upload_file(random_shot_paths)
    else:
        file_list = random_shot_paths

    text, choices = get_class_options(args)
    if args.random_labels:
        for i in range(len(file_list)):
            text += f"\nThe classification of the {i + 1} image is: " + choices[random.randint(0, 2)] + "\n"
    else:
        text += f'\nThe first {len(file_list)} images show distinct types of birds.\n'
    prompt = text + f"\nThe classification of the last image is: "
    # if args.model == 'gemini' or args.model == 'sglang_qwen':
    prompt += "(Answer Letter A or B or C)"
    with open(args.out, 'a') as outf:
        outf.write(f'prompt-------------------------\n')
        outf.write(f'{prompt}\n\n')

    if args.parallel:
        f1, acc, conf_matrix, texts, labels, preds, ids = run_evaluate(gpt4, prompt, exs, file_list)
    else:
        preds, labels, ids, texts = [], [], [], []
        for i in tqdm(range(len(exs)), desc='Predicting...'):
            _, pred = process_example(exs[i], gpt4, prompt, file_list)
            if pred != None:
                preds.append(pred)
                labels.append(exs[i]['label'])
                ids.append(exs[i]['id'])
                texts.append((exs[i]['img_path']))
            else:
                print(f"No prediction for {exs[i]['id']}\t{exs[i]['img_path']}")
                with open(args.out, 'a') as outf:
                    outf.write(f"No prediction for {exs[i]['id']}\t{exs[i]['img_path']}\n")
        correct_count = sum(1 for a, b in zip(labels, preds) if a == b)
        acc = correct_count / len(exs)
        # acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        conf_matrix = confusion_matrix(labels, preds)

    with open(args.out, 'a') as outf:
        outf.write(f'\nAccuracy: {acc}\tF1: {f1}\n')
        outf.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        outf.write(f"Id\tPred\tLabel\tPath\n")
        for i in range(len(labels)):
            if labels[i] != preds[i]:
                outf.write(f'{ids[i]}\t{preds[i]}\t{labels[i]}\t{texts[i]}\n')

    print('DONE!')
