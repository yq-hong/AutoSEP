import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import random
import datetime
from collections import Counter
import clip
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from get_utils import get_predictor, get_task_class


def parser_args():
    parser = argparse.ArgumentParser(description='preprocessing parameters')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--task_name', default='CUB_cuckoo',
                        choices=['iNat_butterfly', 'iNat_lupine', 'Stanford_terrier',
                                 'CUB_cuckoo', 'CUB_oriole', 'CUB_vireo', 'vegfru_greens', 'vegfru_allium'])
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--n_test', default=100, type=int)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--n_examples', default=5, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--vlm_model', type=str, default='ViT-L/14')
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--max_threads', default=8, type=int)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = parser_args()
    args.out = f'results/{args.model}/clip/clip_{args.task_name}_{args.mode}_{args.n_clusters}cluster_{args.n_examples}exs_seed{args.seed}.txt'
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')
    task = get_task_class(args)
    gpt4 = get_predictor(configs)

    exs = task.get_even_exs(args.mode, args.n_test)
    ex_paths = [ex['img_path'] for ex in exs]

    model, preprocess = clip.load(args.vlm_model, device=args.device)
    ex_tensor = torch.cat([preprocess(Image.open(img_name)).unsqueeze(0).to(args.device) for img_name in ex_paths])
    with torch.no_grad():
        ex_embeddings = model.encode_image(ex_tensor).to(args.device)
    ex_embeddings = ex_embeddings.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, init='k-means++', random_state=42)
    kmeans.fit(ex_embeddings)
    labels = kmeans.labels_
    with open(args.out, 'a') as outf:
        outf.write(f'clustering-------------------------\n')
        outf.write(f'cluster label\t{labels.tolist()}\n')
        outf.write(f"true label\t{[ex['label'] for ex in exs]}\n\n")

    prompt = open(f'../prompts/{args.task_name}_multi_zero.md').read().replace("this image shows", "these image show")
    with open(args.out, 'a') as outf:
        outf.write(f'prompt-------------------------\n')
        outf.write(f'{prompt}\n\n')

    cluster_preds, cluster_labels, cluster_exs = [], [], []
    for i in range(args.n_clusters):
        exs_list = [ex for ex, label in zip(exs, labels) if label == i]
        select = random.sample(range(len(exs_list)), min(args.n_examples, len(exs_list)))
        select_paths = [exs_list[idx]['img_path'] for idx in select]
        with open(args.out, 'a') as outf:
            outf.write(f'Select {i} cluster examples-------------------------\n')
            for idx in select:
                outf.write(f"{exs_list[idx]['img_path']}\n")
        # response = utils.google_gemini(prompt, select_paths, max_tokens=6, temperature=args.temperature)[0]
        preds = []
        for p in select_paths:
            single_pred = gpt4.inference(prompt, [p])
            if single_pred != None:
                preds.append(single_pred)
        if len(preds) > 0:
            vote_pred = Counter(preds).most_common(1)[0][0]
        else:
            vote_pred = None
        if vote_pred != None:
            cluster_preds += [vote_pred for _ in range(len(exs_list))]
            cluster_labels += [ex['label'] for ex in exs_list]
            cluster_exs += exs_list
            with open(args.out, 'a') as outf:
                outf.write(f'Pred label {vote_pred}.\t{preds}\n')
        else:
            with open(args.out, 'a') as outf:
                outf.write(f'No prediction for cluster {i}\n')

    acc = accuracy_score(cluster_labels, cluster_preds)
    f1 = f1_score(cluster_labels, cluster_preds, average='micro')
    conf_matrix = confusion_matrix(cluster_labels, cluster_preds)

    with open(args.out, 'a') as outf:
        outf.write(f'\nAccuracy: {acc}\tF1: {f1}\n')
        outf.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        outf.write(f"Id\tPred\tLabel\tPath\n")
        for i in range(len(cluster_labels)):
            if cluster_labels[i] != cluster_preds[i]:
                outf.write(
                    f"{cluster_exs[i]['id']}\t{cluster_preds[i]}\t{cluster_labels[i]}\t{cluster_exs[i]['img_path']}\n")

    print("Done!")


if __name__ == '__main__':
    main()
