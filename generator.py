from abc import ABC, abstractmethod
from liquid import Template
from tqdm import tqdm
import concurrent.futures
import api_utils as utils


class GPT4Generator(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def generate(self, ex, prompt):
        pass


class AttrGredictor(GPT4Generator):
    categories = ['No', 'Yes']

    def generate(self, prompt, ex=None):
        prompt = Template(prompt).render()
        if 'gpt4o' in self.opt['model']:
            response = utils.gpt4o(prompt, [ex['img_path']])[0]
        elif 'sglang' in self.opt['model']:
            response = utils.sglang_model(prompt, [ex['img_path']], model_name=self.opt['model'])[0]
        else:
            response = utils.google_gemini(prompt, [ex['img_path']])[0]

        if response is None:
            print(f"No attributes generated for {ex['id']}\t{ex['img_path']}")
            with open(self.opt['out'], 'a') as outf:
                outf.write(f"No attributes generated for {ex['id']}\t{ex['img_path']}\n")
        return response

    def generate_category(self, prompt, exs=None):
        prompt = Template(prompt).render()
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, [ex['img_path'] for ex in exs])[0]
        else:
            response = utils.gpt4o(prompt, [ex['img_path'] for ex in exs])[0]
        if response is None:
            print(f"No attributes generated category")
        return response


def generate_on_example(inputs):
    generator, prompt, ex = inputs
    pred = generator.generate(prompt, ex)
    return prompt, pred, ex


def parallel_generate(generator, prompt, examples, attr_cache, max_threads):
    inputs = [(generator, prompt, ex) for ex in examples]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(generate_on_example, ex) for ex in inputs]
    for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='Generating (parallel)'):
        prompt, pred, ex = future.result()
        attr_cache[f'{prompt}'][f'{ex}'] = pred
    return attr_cache
