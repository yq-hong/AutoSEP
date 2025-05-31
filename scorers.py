from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures
from autosep.llm_text_compare import select_k_from_n_excluding_i, predict_with_compare
from spo.llm_eval import select_k_from_n_excluding_i, prompt_spo_compare


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    pred = predictor.inference(prompt, [ex['img_path']])
    return prompt, ex, pred


def predict_on_example_attr(inputs):
    ex, predictor, pred_prompt, prompt, attr = inputs
    img_path = ex['img_path']
    pred = predictor.inference(pred_prompt, [img_path], attr)
    return prompt, ex, pred


class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores = {}
            if model_name == 'gemini' or model_name == 'gpt4o' or 'sglang' in model_name:
                if attribute_cache != None:
                    inputs = [(ex, predictor, pred_prompts[f'{prompt}'], prompt, attribute_cache[f'{prompt}'][f'{ex}'])
                              for prompt, ex in prompts_exs]
                else:
                    inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                    if attribute_cache != None:
                        futures = [executor.submit(predict_on_example_attr, ex) for ex in inputs]
                    else:
                        futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                    for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                          desc='01 scorer parallel'):
                        prompt, ex, pred = future.result()
                        if pred == ex['label']:
                            out_scores[f'{ex}-{prompt}'] = 1
                        else:
                            out_scores[f'{ex}-{prompt}'] = 0
            else:
                for prompt, ex in tqdm(prompts_exs, desc='01 scorer single'):
                    prompt, ex, pred = predict_on_example_attr((ex, predictor, pred_prompts[f'{prompt}'], prompt,
                                                                attribute_cache[f'{prompt}'][f'{ex}']))
                    if pred == ex['label']:
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0

            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))

        computed_scores = compute_scores(prompts_exs_to_compute)

        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)


class CachedCompareScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores, prompt_scores = {}, {}
            for prompt, ex in prompts_exs:
                prompt_scores[f'{ex}-{prompt}'] = {}
                out_scores[f'{ex}-{prompt}'] = 0

            if model_name == 'gemini' or model_name == 'gpt4o' or 'sglang' in model_name:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                    futures = [executor.submit(predict_with_compare, true_ex, false_ex, prompt,
                                               attribute_cache[f'{prompt}'], model_name)
                               for prompt, true_ex in prompts_exs for false_ex in false_exs[f'{true_ex}']]
                    for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                          desc='compare scorer parallel'):
                        answer, true_ex, false_ex, prompt = future.result()
                        prompt_scores[f'{true_ex}-{prompt}'][f'{false_ex}'] = answer
                        out_scores[f'{true_ex}-{prompt}'] += answer
            else:
                for prompt, true_ex in tqdm(prompts_exs, desc='compare scorer single'):
                    for false_ex in false_exs[f'{true_ex}']:
                        answer, true_ex, false_ex, prompt = predict_with_compare(true_ex, false_ex, prompt,
                                                                                 attribute_cache[f'{prompt}'],
                                                                                 model_name)
                        prompt_scores[f'{true_ex}-{prompt}'][f'{false_ex}'] = answer
                        out_scores[f'{true_ex}-{prompt}'] += answer

            return out_scores

        false_exs = {}
        for i in range(len(data)):
            false_idx = select_k_from_n_excluding_i(len(data), 2, i)  # compare
            false_exs[f'{data[i]}'] = [data[idx] for idx in false_idx]

        cached_scores = defaultdict(list)
        for prompt in prompts:
            prompts_exs_to_compute = []
            for ex in data:  # for ex, prompt in [(ex, prompt) for ex in data]:
                if f'{ex}-{prompt}' in self.cache:
                    cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
                else:
                    prompts_exs_to_compute.append((prompt, ex))

            computed_scores = compute_scores(prompts_exs_to_compute)

            for prompt, ex in prompts_exs_to_compute:
                self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
                cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)


class CachedSPOScorer:

    def __init__(self, args):
        self.opt = args
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores, prompt_scores = {}, {}
            for pi, ex, idx in prompts_exs:
                out_scores[f'{ex}-{pi}-{idx}'] = 0

            if model_name == 'gemini' or model_name == 'gpt4o' or 'sglang' in model_name:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                    futures = [executor.submit(prompt_spo_compare, ex, pos_idx, neg_idx,
                                               attribute_cache[f'{prompts[pos_idx]}'],
                                               attribute_cache[f'{prompts[neg_idx]}'],
                                               self.opt['task_name'], model_name)
                               for pos_idx, ex, neg_idx in prompts_exs]
                    for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                          desc='compare SPO scorer parallel'):
                        answer, ex, pos, neg = future.result()
                        out_scores[f'{ex}-{pos}-{neg}'] += answer
            else:
                for pos_idx, ex, neg_idx in tqdm(prompts_exs, desc='compare SPO scorer single'):
                    answer, ex, pos, neg = prompt_spo_compare(ex, pos_idx, neg_idx,
                                                              attribute_cache[f'{prompts[pos_idx]}'],
                                                              attribute_cache[f'{prompts[neg_idx]}'],
                                                              self.opt['task_name'], model_name)
                    out_scores[f'{ex}-{pos}-{neg}'] += answer

            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex in data:
            prompts_exs_to_compute.append((0, ex, 1))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for p, ex, idx in prompts_exs_to_compute:
            cached_scores[prompts[0]].append(computed_scores[f'{ex}-{p}-{idx}'])
            cached_scores[prompts[1]].append(1 - computed_scores[f'{ex}-{p}-{idx}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)
