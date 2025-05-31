import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import datetime
import time
import json
import random
import argparse
import scorers
import optimizers
import generator
import evaluators
from get_utils import get_predictor, get_task_class, get_exs

random.seed(42)


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')


def get_scorer(scorer):
    if scorer == 'compare':
        return scorers.CachedCompareScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='CUB_cuckoo',
                        choices=['iNat_butterfly', 'iNat_grass', 'Stanford_terrier',
                                'CUB_cuckoo', 'CUB_oriole', 'CUB_vireo', 'vegfru_1', 'vegfru_2'])
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--gradient_model', default='gemini')
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--max_threads', default=8, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_train', default=30, type=int, help='# instances per class')
    parser.add_argument('--n_val', default=30, type=int)
    parser.add_argument('--n_test', default=30, type=int)

    parser.add_argument('--minibatch_size', default=60, type=int, help='# total instances per minibatch')
    parser.add_argument('--n_gradients', default=4, type=int, help='# generated gradients per prompt')
    parser.add_argument('--errors_per_gradient', default=4, type=int,
                        help='# error examples used to generate one gradient')
    parser.add_argument('--gradients_per_error', default=1, type=int, help='# gradient reasons per error')
    parser.add_argument('--steps_per_gradient', default=1, type=int, help='# new prompts per gradient reason')
    parser.add_argument('--mc_samples_per_step', default=1, type=int, help='# synonyms')
    parser.add_argument('--max_expansion_factor', default=5, type=int, help='maximum # prompts after expansion')

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="compare", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--eval_budget', default=30, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)

    parser.add_argument('--reject_on_errors', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_seeds', default=10, type=int, help='# shuffle seeds in scorer')
    parser.add_argument('--test_eval', action='store_true', default=False)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.out = f"results/{args.out_num}_{args.task_name}/apo_multi_{args.task_name}_{args.out_num}.txt"
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    if args.evaluator != "bf1":
        configs['eval_budget'] = (configs['samples_per_eval'] * configs['eval_rounds']
                                  * configs['eval_prompts_per_round'])
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    task = get_task_class(args)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(configs)
    bf_eval = get_evaluator('bf')(configs)
    gpt4 = get_predictor(configs)
    gpt_generator = generator.AttrGredictor(configs)
    optimizer = optimizers.ProTeGi(configs, evaluator, scorer, args.max_threads, bf_eval)

    train_exs, val_exs, test_exs = get_exs(args, task)

    candidates = [open(f'../prompts/{args.task_name}_generate.md').read()]
    pred_prompt = open(f'../prompts/{args.task_name}_multi.md').read()
    with open(args.out, 'a') as outf:
        outf.write(f'pred_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')

    attribute_cache, test_attr_cache, pred_prompts = {}, {}, {}
    for prompt in candidates:
        attribute_cache[f'{prompt}'] = {}
        attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs,
                                                          attribute_cache, args.max_threads)

        test_attr_cache[f'{prompt}'] = {}
        if args.test_eval:
            test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs,
                                                              test_attr_cache, args.max_threads)
            pred_prompts[f'{prompt}'] = pred_prompt

    for round in tqdm(range(configs['rounds'] + 1)):
        print("STARTING ROUND ", round)
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
        start = time.time()

        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs, attribute_cache=attribute_cache)
            for prompt in candidates:
                if f'{prompt}' not in attribute_cache:
                    attribute_cache[f'{prompt}'] = {}
                    attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs,
                                                                      attribute_cache, args.max_threads)

        scores = optimizer.score_candidates(candidates, gpt4, train_exs, attribute_cache=attribute_cache)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        candidates = candidates[:configs['beam_size']]
        scores = scores[:configs['beam_size']]

        with open(args.out, 'a') as outf:
            outf.write(f'{time.time() - start}\n')
            for c in candidates:
                outf.write(json.dumps(c) + '\n')
            outf.write(f'{scores}\n')

        metrics = []
        for prompt in candidates:
            if f'{prompt}' not in test_attr_cache:
                test_attr_cache[f'{prompt}'] = {}
                if args.test_eval:
                    test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs,
                                                                      test_attr_cache, args.max_threads)
                    pred_prompts[f'{prompt}'] = pred_prompt

        if args.test_eval:
            for candidate, score in zip(candidates, scores):
                f1, texts, labels, preds, attr = task.evaluate(gpt4, candidate, test_exs, pred_prompts=pred_prompts,
                                                               attribute_cache=test_attr_cache, model_name=args.model)
                metrics.append(f1)
            with open(args.out, 'a') as outf:
                outf.write(f'{metrics}\n')

        with open(f'results/{args.out_num}_{args.task_name}/{args.out_num}_train_attr.json', 'w') as json_file:
            json.dump(attribute_cache, json_file)
        with open(f'results/{args.out_num}_{args.task_name}/{args.out_num}_test_attr.json', 'w') as json_file:
            json.dump(test_attr_cache, json_file)

    print("DONE!")
