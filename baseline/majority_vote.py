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
import generator
from get_utils import get_predictor, get_task_class


def process_example(ex, predictor, prompt, attr=None, n_votes=5):
    img_path = ex['img_path']
    pred = predictor.inference_majority_vote(prompt, [img_path], attr, n_votes=n_votes)
    return ex, pred


def run_evaluate(predictor, prompt, exs, attributes_dict=None, n_votes=5):
    ids = []
    labels = []
    preds = []
    img_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        if attributes_dict == None:
            futures = [executor.submit(process_example, ex, predictor, prompt, None, n_votes) for ex in exs]
        else:
            futures = [executor.submit(process_example, ex, predictor, prompt, attributes_dict[f'{ex["id"]}'], n_votes)
                       for ex in exs]
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
                try:
                    with open(args.out, 'a') as outf:
                        outf.write(f"No prediction for {ex['id']}\t{ex['img_path']}\n")
                except:
                    pass

    correct_count = sum(1 for a, b in zip(labels, preds) if a == b)
    accuracy = correct_count / len(exs)
    # accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    conf_matrix = confusion_matrix(labels, preds)
    return f1, accuracy, conf_matrix, img_paths, labels, preds, ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='CUB_cuckoo',
                        choices=['iNat_butterfly', 'iNat_lupine', 'Stanford_terrier',
                                 'CUB_cuckoo', 'CUB_oriole', 'CUB_vireo', 'vegfru_greens', 'vegfru_allium'])
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--n_test', default=100, type=int)
    parser.add_argument('--attributes', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--n_votes', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--max_threads', default=8, type=int)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    if args.attributes:
        args.out = f'results/{args.model}/majority_vote/majority_vote_{args.task_name}_{args.n_votes}vote_temp{args.temperature}_attr{args.out_num}.txt'
    else:
        args.out = f'results/{args.model}/majority_vote/majority_vote_{args.task_name}_{args.n_votes}vote_temp{args.temperature}_{args.out_num}.txt'
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')
    task = get_task_class(args)
    gpt4 = get_predictor(configs)

    exs = task.get_even_exs(args.mode, args.n_test)

    prompt_path = f'../prompts/{args.task_name}_multi.md' if args.attributes else f'../prompts/{args.task_name}_multi_zero.md'
    prompt = open(prompt_path).read()
    with open(args.out, 'a') as outf:
        outf.write(f'prompt-------------------------\n')
        outf.write(f'{prompt}\n\n')

    generate_prompt = open(f'../prompts/{args.task_name}_generate.md').read()
    if args.attributes:
        with open(args.out, 'a') as outf:
            outf.write(f'\ngenerate_prompt-------------------------\n')
            outf.write(f'{generate_prompt}\n\n')
    gpt_generator = generator.AttrGredictor(configs)

    if args.attributes:
        if args.generate:
            attrs = task.get_attr(args.mode, generate_prompt, exs, gpt_generator, generate=True, model_name=args.model)
        else:
            attrs = task.get_attr(args.mode, generate_prompt, exs)
    else:
        attrs = None

    if args.parallel:
        f1, acc, conf_matrix, texts, labels, preds, ids = run_evaluate(gpt4, prompt, exs, attrs, args.n_votes)
    else:
        preds, labels, ids, texts = [], [], [], []
        for i in tqdm(range(len(exs)), desc='Predicting...'):
            if args.attributes:
                _, pred = process_example(exs[i], gpt4, prompt, attrs[f'{exs[i]["id"]}'], args.n_votes)
            else:
                _, pred = process_example(exs[i], gpt4, prompt, None, args.n_votes)
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
