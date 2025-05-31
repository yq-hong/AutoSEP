from abc import ABC, abstractmethod
from liquid import Template
from collections import Counter
import api_utils as utils


class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class TwoClassPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, attr, prompt, img_paths=None):
        prompt = Template(prompt).render(text=attr)
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=6, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=6, n=1, temperature=self.opt['temperature'])[0]
        if response is None:
            print("No response received from the model.")
            return 1
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

    def zero_shot_inference(self, prompt, img_paths=None):
        prompt = Template(prompt).render()
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=6, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=6, n=1, temperature=self.opt['temperature'])[0]
        if response is None:
            print("No response received from the model.")
            return 1
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

class ThreeClassPredictor(GPT4Predictor):

    def inference(self, prompt, img_paths=None, attr=None, few_shot_files=None, content=None):
        prompt = Template(prompt).render(text=attr)

        if 'gemini' in self.opt['model']:
            if few_shot_files != None:
                response = utils.google_gemini(prompt, img_paths, few_shot_files=few_shot_files,
                                               max_tokens=16, temperature=self.opt['temperature'])[0]
            else:
                response = utils.google_gemini(prompt, img_paths, max_tokens=16, temperature=self.opt['temperature'])[0]
        elif 'sglang' in self.opt['model']:
            response = utils.sglang_model(prompt, img_paths, max_tokens=16, temperature=self.opt['temperature'],
                                          model_name=self.opt['model'])[0]
        elif 'gpt4o' in self.opt['model']:
            response = utils.gpt4o(prompt, img_paths, max_tokens=16, n=1, temperature=self.opt['temperature'])[0]
        else:
            raise Exception(f"Unsupported model: {self.opt['model']}")
        if response is None:
            print("No response received from the model.")
            return None

        if 'A.' in response or '**A' in response or 'A\n' in response or 'A \n' in response or '(A)' in response or ': A' in response or response == 'A':
            pred = 0
        elif 'B.' in response or '**B' in response or 'B\n' in response or 'B \n' in response or '(B)' in response or ': B' in response or response == 'B':
            pred = 1
        elif 'C.' in response or '**C' in response or 'C\n' in response or 'C \n' in response or '(C)' in response or ': C' in response or response == 'C':
            pred = 2
        else:
            print(f"No valid response. {response}")
            return None

        return pred

    def inference_majority_vote(self, prompt, img_paths=None, attr=None, content=None, n_votes=5):
        preds = []
        for i in range(n_votes):
            single_pred = self.inference(prompt, img_paths, attr, content)
            if single_pred != None:
                preds.append(single_pred)
        if len(preds) > 0:
            vote_pred = Counter(preds).most_common(1)[0][0]
        else:
            vote_pred = None

        return vote_pred
