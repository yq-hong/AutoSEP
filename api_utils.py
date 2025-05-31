import os
import sys

sys.path.append("./sglang/python")
sys.path.append("./../sglang/python")
import re
import time
import base64
import config
import string
import PIL.Image
import openai
from openai import OpenAI, BadRequestError
import pathlib

openai.organization = config.openai_organization
media = pathlib.Path(__file__).parents[1] / "third_party"


def clean_text(text):
    text = text.replace('-', ' ')
    # Remove non-letter and non-space characters using regex, then convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return cleaned_text


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


def parse_sectioned_prompt(s):
    '''
    Have sections separated by headers (lines starting with # ).
    The function parses the string into a dictionary, where each section header becomes a key,
    and the corresponding content under that header becomes the associated value.
    '''
    result = {}
    current_header = None

    for line in s.split('\n'):
        # line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def gpt4o(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024,
          presence_penalty=0, frequency_penalty=0, logit_bias={}):
    client = OpenAI(api_key=config.openai_api_key,
                    project=config.openai_project_id)

    if img_paths != None:
        # imgs_url = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_paths[i])}"}}
        #             for i in range(len(img_paths))]
        imgs_url = []
        for i in range(len(img_paths)):
            base64_image, mime_type = encode_image(img_paths[i])
            imgs_url.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
        messages = [{"role": "user",
                     "content": imgs_url + [{"type": "text", "text": prompt}], }]
    else:
        messages = [{"role": "user", "content": prompt}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                      messages=messages,
                                                      temperature=temperature,
                                                      n=n,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      presence_penalty=presence_penalty,
                                                      frequency_penalty=frequency_penalty,
                                                      logit_bias=logit_bias
                                                      )
            num_attempts = 5
            return [response.choices[i].message.content for i in range(n)]

        except BadRequestError as be:
            print(f"BadRequestError: {be}")
            continue
        except openai.RateLimitError as e:
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"OpenAI server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)  # Wait for 5 seconds before the next attempt
            continue


def google_gemini(prompt, img_paths=None, few_shot_files=None, temperature=0.7, n=1, top_p=1, max_tokens=1024):
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    from google.generativeai.types.generation_types import StopCandidateException
    from google.generativeai import protos

    genai.configure(api_key=config.gemini_api_key)

    safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": 'BLOCK_NONE'}]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    if img_paths != None:
        # imgs = [genai.upload_file(media / image) for image in img_paths]
        imgs = [PIL.Image.open(image) for image in img_paths]
        if few_shot_files != None:
            messages = few_shot_files + imgs + [prompt]
        else:
            messages = imgs + [prompt]
    else:
        messages = prompt

    num_attempts = 0
    while num_attempts < 10:
        num_attempts += 1
        try:
            response = model.generate_content(messages,
                                              generation_config=genai.GenerationConfig(temperature=temperature,
                                                                                       top_p=top_p,
                                                                                       max_output_tokens=max_tokens),
                                              safety_settings=safety_settings
                                              )
            FinishReason = protos.Candidate.FinishReason
            if response.candidates:
                if (response.candidates[0].finish_reason == FinishReason.STOP
                        or response.candidates[0].finish_reason == FinishReason.MAX_TOKENS):
                    out = response.text
                    num_attempts = 10
                    return [out]
                else:
                    if not response.candidates:
                        print("Generate issue: No candidates returned in response.")
                    else:
                        print(f"Generate issue {response.candidates[0].finish_reason}")
                    time.sleep(1)

        except StopCandidateException as e:
            if e.args[0].finish_reason == 3:  # Block reason is safety
                print('Blocked for Safety Reasons')
                time.sleep(1)
        except ResourceExhausted as e:  # Too many requests, wait for a minute
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
        except Exception as e:
            print(f"Other issue: {e}")
            time.sleep(1)
    return [None]


def clear_gemini_img_files(verbose=False):
    import google.generativeai as genai

    genai.configure(api_key=config.gemini_api_key)
    for f in genai.list_files():
        myfile = genai.get_file(f.name)
        myfile.delete()
        if verbose:
            print("Deleted", f.name)


def get_gemini_upload_file(img_paths):
    import google.generativeai as genai

    genai.configure(api_key=config.gemini_api_key)

    files, file_names = [], []
    for image in img_paths:
        file = genai.upload_file(media / image)
        file_name = file.name
        files.append(file)
        file_names.append(file_name)

    return files, file_names


def sglang_model(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024, model_name='sglang_qwen'):
    import sglang as sgl
    from sglang import function
    from sglang.lang.chat_template import get_chat_template

    if 'qwen' in model_name:
        host = "http://localhost:30000"
        template = "qwen2-vl"
    else:
        raise Exception(f'Unsupported task: {model_name}')

    # sgl.set_default_backend(sgl.RuntimeEndpoint(host))
    backend = sgl.RuntimeEndpoint(host)

    chat_template = get_chat_template(template)
    backend.chat_template = chat_template

    @function
    def image_qa(s, question, image_path=None):
        # s += sgl.user(sgl.image(image_path) + question)
        if image_path != None:
            s += sgl.user(sgl.image(image_path))
        s += sgl.user(question)
        s += sgl.assistant(sgl.gen("answer"))

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            if img_paths == None:
                state = image_qa.run(question=prompt,
                                     image_path=None,
                                     backend=backend,
                                     max_new_tokens=max_tokens,
                                     temperature=temperature,
                                     top_p=top_p)
            else:
                state = image_qa.run(question=prompt,
                                     image_path=img_paths[0],
                                     backend=backend,
                                     max_new_tokens=max_tokens,
                                     temperature=temperature,
                                     top_p=top_p)
            num_attempts = 5
            return [state["answer"].strip()]

        except Exception as e:
            print(f"SGLang server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)
            continue


def sglang_model1(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024, model_name='sglang_qwen'):
    if 'qwen' in model_name:
        client = OpenAI(base_url=f"http://localhost:30000/v1", api_key="None")
    else:
        raise Exception(f'Unsupported task: {model_name}')

    if img_paths != None:
        imgs_url = []
        for i in range(len(img_paths)):
            imgs_url.append({"type": "image_url", "image_url": {"url": img_paths[i]}})
        messages = [{"role": "user",
                     "content": imgs_url + [{"type": "text", "text": prompt}], }]
    else:
        messages = [{"role": "user", "content": prompt}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model="Qwen/Qwen2-VL-72B-Instruct",
                                                      messages=messages,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      )
            num_attempts = 5
            return [response.choices[0].message.content]

        except Exception as e:
            print(f"SGLang server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)
            continue
