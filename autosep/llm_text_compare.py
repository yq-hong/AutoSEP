import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import concurrent.futures
import json
import random
import argparse
import generator
import api_utils as utils


def select_k_from_n_excluding_i(n, k, i):
    numbers = list(range(n))
    if i in numbers:
        numbers.remove(i)

    if k > len(numbers):
        raise ValueError("Cannot select k numbers from the remaining numbers.")
    selected_numbers = random.sample(numbers, k)

    return selected_numbers


def predict_with_compare(true_ex, false_ex, prompt, attrs, model_name='gemini'):
    random_bit = random.randint(0, 1)

    if random_bit == 0:
        pred_prompt = f"Text 1:\n{attrs[f'{true_ex}']}\n\nText 2:\n{attrs[f'{false_ex}']}\n\nWhich description correctly describes the image? The first text or the second text?"
        if 'gemini' in model_name:
            response = utils.google_gemini(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0)[0]
        elif 'gpt4o' in model_name:
            response = utils.gpt4o(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0)[0]
        elif 'sglang' in model_name:
            response = utils.sglang_model(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0,
                                          model_name=model_name)[0]
        else:
            raise Exception(f"Unsupported model: {model_name}")
        if 'first' in response.lower() and 'second' not in response.lower():
            answer = 1
        else:
            answer = 0
    else:
        pred_prompt = f"Text 1:\n{attrs[f'{false_ex}']}\n\nText 2:\n{attrs[f'{true_ex}']}\n\nWhich description correctly describes the image? The first text or the second text?"
        if 'gemini' in model_name:
            response = utils.google_gemini(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0)[0]
        elif 'gpt4o' in model_name:
            response = utils.gpt4o(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0)[0]
        elif 'sglang' in model_name:
            response = utils.sglang_model(pred_prompt, [true_ex['img_path']], max_tokens=6, temperature=0,
                                          model_name=model_name)[0]
        else:
            raise Exception(f"Unsupported model: {model_name}")
        if 'second' in response.lower() and 'first' not in response.lower():
            answer = 1
        else:
            answer = 0

    return answer, true_ex, false_ex, prompt


def run_evaluate(true_exs, false_exs, prompt, attrs, model_name='gemini'):
    trues, falses, preds = [], [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(predict_with_compare, true_ex, false_ex, prompt, attrs, model_name)
                   for true_ex in true_exs for false_ex in false_exs[f'{true_ex}']]

        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                              total=len(futures), desc='running evaluation on all exs combinations'):
            answer, true_ex, false_ex, prompt = future.result()
            trues.append(true_ex)
            falses.append(false_ex)
            preds.append(answer)

    return trues, falses, preds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='CUB_cuckoo',
                        choices=['iNat_butterfly', 'iNat_lupine', 'Stanford_terrier',
                                 'CUB_cuckoo', 'CUB_oriole', 'CUB_vireo', 'vegfru_greens', 'vegfru_allium'])
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--max_threads', default=8, type=int)
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--result_folder', default='prompt_optimization')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--exp', default=14, type=int)
    parser.add_argument('--n_test', default=30, type=int)
    parser.add_argument('--n_compare', default=20, type=int)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    score_address = f"compare/text_compare/prompt_scores_{args.result_folder}_{args.exp}_{args.task_name}_{args.model}_{args.mode}_{args.n_test}_{args.n_compare}.json"
    os.makedirs(os.path.dirname(score_address), exist_ok=True)
    prompt_test_address = f"../{args.result_folder}/results/{args.exp}_{args.task_name}/{args.exp}_test_attr.json"
    prompt_address = f"../{args.result_folder}/results/{args.exp}_{args.task_name}/{args.exp}_{args.mode}_attr.json"

    with open(prompt_test_address, 'r') as json_file:
        attr_all = json.load(json_file)
    prompt_keys = list(attr_all.keys())
    with open(prompt_address, 'r') as json_file:
        attr_all = json.load(json_file)

    if not args.evaluate:
        with open(score_address, 'r') as json_file:
            prompt_scores = json.load(json_file)
    else:
        import main

        configs = vars(args)
        task = main.get_task_class(args)
        gpt4 = main.get_predictor(configs)
        gpt_generator = generator.AttrGredictor(configs)

        exs = task.get_even_exs(args.mode, args.n_test)
        if args.generate:
            attr_all = {}
            for prompt in prompt_keys:
                attr_all[f'{prompt}'] = {}
                attr_all = generator.parallel_generate(gpt_generator, prompt, exs, attr_all, args.max_threads)

        if args.parallel:
            false_exs = {}
            for i in range(len(exs)):
                false_idx = select_k_from_n_excluding_i(len(exs), args.n_compare, i)
                false_exs[f'{exs[i]}'] = [exs[idx] for idx in false_idx]
            prompt_scores = {}
            for prompt in prompt_keys:
                prompt_scores[f'{prompt}'] = {}
                for i in range(len(exs)):
                    prompt_scores[f'{prompt}'][f'{exs[i]}'] = {}
                trues, falses, preds = run_evaluate(exs, false_exs, prompt, attr_all[f'{prompt}'], args.model)
                for i in range(len(trues)):
                    prompt_scores[f'{prompt}'][f'{trues[i]}'][f'{falses[i]}'] = preds[i]
        else:
            prompt_scores = {}
            for prompt in prompt_keys:
                prompt_scores[f'{prompt}'] = {}
                for i in tqdm(range(len(exs)), desc=f"evaluating one prompt (Single)"):
                    prompt_scores[f'{prompt}'][f'{exs[i]}'] = {}
                    selected_idx = select_k_from_n_excluding_i(len(exs), args.n_compare, i)
                    for j in range(len(selected_idx)):
                        (prompt_scores[f'{prompt}'][f'{exs[i]}'][f'{exs[selected_idx[j]]}'],
                         _, _, _) = predict_with_compare(exs[i], exs[selected_idx[j]], prompt, attr_all[f'{prompt}'],
                                                         args.model)

        with open(score_address, 'w') as json_file:
            json.dump(prompt_scores, json_file)

    all_scores = {}
    for prompt in prompt_keys:
        all_scores[f'{prompt}'] = 0
        for ex, values in prompt_scores[f'{prompt}'].items():
            all_scores[f'{prompt}'] += sum(values.values())

    for s in all_scores.values():
        print(s)

    print('Done!')
