import json
import numpy as np
from openai import OpenAI
from llama_cpp import Llama
from llama_cpp import LlamaGrammar
from typing import List
from math_pars import get_result
from yello import Yello, Point
from saygment import Saygment


def convert_to_template(user_prompt: str, system_prompt: str, template: str) -> str:
    if template == 'Orca-Vicuna':
        # SYSTEM: {system_message}
        # USER: {prompt}
        # ASSISTANT:
        return f'SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:'
    elif template == 'Llama-2-Chat':
        # [INST] <<SYS>>
        # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        # <</SYS>>
        # {prompt}[/INST]
        return f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]'
    elif template == 'Orca-Hashes':
        # ### System:
        # {system_message}

        # ### User:
        # {prompt}

        # ### Assistant:
        return f'### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:'
    elif template == 'Alpaca':
        # Below is an instruction that describes a task. Write a response that appropriately completes the request.

        # ### Instruction:
        # {prompt}

        # ### Response:
        return f'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{system_prompt}{user_prompt}\n\n### Response:'
    elif template == "ChatML":
        # <|im_start|>system
        # {system_message}<|im_end|>
        # <|im_start|>user
        # {prompt}<|im_end|>
        # <|im_start|>assistant
        return f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant'
    elif template == "Vicuna":
        # A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:
        return f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER:{system_prompt}\n {user_prompt} ASSISTANT:'
    elif template == "Llama-3":
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        # {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        # {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

    elif template == "phi":
        return f'<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>'

    else:
        return f'{system_prompt}\nUser:\n{user_prompt}'


def covert_to_template_with_examples(user_prompt: str, system_prompt: str, examples: dict, template: str) -> str:
    if template == "ChatML":
        # <|im_start|>system
        # {system_message}<|im_end|>
        # <|im_start|>user
        # {prompt}<|im_end|>
        # <|im_start|>assistant

        prompt = ""

        # let's start by adding the system prompt
        prompt += f'<|im_start|>system\n{system_prompt}<|im_end|>\n'

        # now we add every example
        for ex in examples:
            prompt += f'<|im_start|>user\n{ex["user"]}<|im_end|>\n<|im_start|>assistant\n{ex["assistant"]}<|im_end|>\n'

        # finally we add the user prompt
        prompt += f'<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant'
        return prompt

    elif template == "Orca-Vicuna":
        # SYSTEM: {system_message}
        # USER: {prompt}
        # ASSISTANT:
        prompt = ""
        prompt += f'SYSTEM: {system_prompt}\n'
        for ex in examples:
            prompt += f'USER: {ex["user"]}\nASSISTANT: {ex["assistant"]}\n'
        prompt += f'USER: {user_prompt}\nASSISTANT:'
        return prompt

    elif template == "Llama-3":
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        #
        # {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        #
        # {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        prompt = ""
        prompt += f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>'
        for ex in examples:
            prompt += f'<|start_header_id|>user<|end_header_id|>\n{ex["user"]}\n<|start_header_id|>assistant<|end_header_id|>\n{ex["assistant"]}\n'
        prompt += f'<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        return prompt

    elif template == "phi":
        # f'<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>'
        prompt = ""
        prompt += f'<|system|>\n{system_prompt}<|end|>\n'
        for ex in examples:
            prompt += f'<|user|>\n{ex["user"]}<|end|>\n<|assistant|>\n{ex["assistant"]}<|end|>\n'
        prompt += f'<|user|>\n{user_prompt}<|end|>\n<|assistant|>'
        return prompt
    else:
        prompt = ""
        prompt += f'{system_prompt}\n'
        for ex in examples:
            prompt += f'User: {ex["user"]}\nAssistant: {ex["assistant"]}\n'
        prompt += f'User: {user_prompt}'
        return prompt


class LPVCS:
    def __init__(self, use_gpt=False, use_phi=False, llm_path='models/LLMs/Tess/tess-10.7b-v1.5b.Q6_K.gguf', yello_vlm="GroundingDINO", saygment_vlm="CLIP_Surgery", chat_template="Orca-Vicuna"):
        self.use_gpt = use_gpt
        if self.use_gpt:
            from openai_key import openai_key
            if openai_key == "":
                raise Exception("OpenAI key not set")
            self.client = OpenAI(api_key=openai_key)
        else:
            print("Loading LLM...")
            self.model = self.load_llama(llm_path)

        self.template = chat_template
        print("Loading Yello... (Object Detection)")
        self.yello = Yello(vlm=yello_vlm, debug=False)
        print("Loading Saygment... (Object Segmentation)")
        self.saygment = Saygment(vlm=saygment_vlm, debug=True)
        self.history = []
        self.log = []

    def load_llama(self, llm_path) -> Llama:
        """
            Loads the local LLM model using the llama_cpp library

            Args:
            llm_path: The path to the LLM model

            Returns:
            Llama: The LLM model
        """
        return Llama(llm_path, n_gpu_layers=-1, verbose=False, n_ctx=1024*4)

    def classify(self, prompt: str) -> str:
        """
            Classifies the prompt into one of the following categories:
            - prediction
            - correction
            - confirmation

            Args:
            prompt: The prompt to be classified

            Returns:
            str: The category of the prompt
        """
        prompts_json = json.load(open('prompts/prompts.json'))["prompts"]
        system_prompt = prompts_json["classifier"]["system_prompt"]

        if self.use_gpt:
            # let's build the mesasges using the example prompts in the json file
            messages = []

            # first we need the system prompt

            messages.append({"role": "system", "content": system_prompt})

            for example in prompts_json["classifier"]["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                logit_bias={70031: 5,
                            6133: 5,
                            31466: 5,
                            66481: 5
                            },
                max_tokens=20
            )

            return completion.choices[0].message.content

        else:
            # first we nee the classification grammar
            grammar = LlamaGrammar.from_file("grammar/classifier.gbnf", verbose=False)

            final_prompt = covert_to_template_with_examples(
                prompt, system_prompt, prompts_json["prediction"]["examples"], self.template)

            category = self.model(final_prompt,
                                  max_tokens=5, temperature=0, top_p=0.9, grammar=grammar)['choices'][0]['text'].strip()

            return category

    def get_objects_in_prompt(self, prompt: str) -> List[str]:
        """
            Gets the objects in the prompt

            Args:
            prompt: The prompt to get the objects from

            Returns:
            List[str]: The objects in the prompt
        """

        prompts_json = json.load(open('prompts/prompts.json'))["prompts"]
        system_prompt = prompts_json["object_detection"]["system_prompt"]
        if self.use_gpt:

            messages = []
            messages.append({"role": "system", "content": system_prompt})
            for example in prompts_json["object_detection"]["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.6,
                max_tokens=1024
            )
            try:
                response = completion.choices[0].message.content
                response = json.loads(response)

                return response["objects"]
            except:
                return []

        else:
            grammar = LlamaGrammar.from_file("grammar/text_object_detector.gbnf", verbose=False)

            final_prompt = covert_to_template_with_examples(
                prompt, system_prompt, prompts_json["object_detection"]["examples"], self.template)
            output = self.model(final_prompt,
                                max_tokens=1024*3, temperature=0, grammar=grammar)['choices'][0]['text']

            output = json.loads(output)

            try:
                return response["objects"]
            except:
                return []

    def predict(self, prompt: str, img: np.array, objects=None):
        self.history = []
        self.log.append(prompt)
        # let's extract the objects from the prompt
        if objects is None:
            objects = self.get_objects_in_prompt(prompt)

        # get the bounding boxes
        if len(objects) == 0:
            bbs = []
        else:
            bbs = self.yello.predict(img, objects)

        # Construct the prompt containg the bounding boxes
        objects_prompt = ""
        for bb in bbs:
            objects_prompt += bb.get_sys_prompt() + ". "

        user_prompt = objects_prompt + prompt

        # get the system prompt from the json file
        prompts_json = json.load(open('prompts/prompts.json'))["prompts"]
        system_prompt = prompts_json["prediction"]["system_prompt"]

        if self.use_gpt:

            # let's build the mesasges using the example prompts in the json file
            messages = []

            # first we need the system prompt
            messages.append({"role": "system", "content": system_prompt})

            for example in prompts_json["prediction"]["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

            # now we add the user prompt
            messages.append({"role": "user", "content": user_prompt})

            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=messages,
                temperature=1,
                max_tokens=1024
            )
            response = completion.choices[0].message.content

        else:

            final_prompt = covert_to_template_with_examples(
                user_prompt, system_prompt, prompts_json["prediction"]["examples"], self.template)

            # use llama cpp
            grammar = LlamaGrammar.from_file("grammar/new_grammar.gbnf", verbose=False)
            response = self.model(final_prompt, max_tokens=1024*4, temperature=0.6,
                                  grammar=grammar, repeat_penalty=1.1)['choices'][0]['text']
        try:
            response = json.loads(response)

            print("="*10)
            math_x = response["math_expression_x"]
            math_y = response["math_expression_y"]
            x = int(get_result(math_x))
            y = int(get_result(math_y))

            # clip the point within the image
            x = min(max(0, x), img.shape[1])
            y = min(max(0, y), img.shape[0])

        except:
            x = 0
            y = 0

        cot = response["chain_of_thought"]

        return Point(x, y), None, bbs, cot, response

    def correct(self, prompt: str, target: np.array, img: np.array) -> Point:
        self.log.append(prompt)

        objects = self.get_objects_in_prompt(prompt)
        bbs = []
        if len(objects) > 0:
            bbs = self.yello.predict(img, objects)
        target_prompt = "Target is at [{}, {}]. ".format(target[1], target[0])

        objects_prompt = ""

        for bb in bbs:
            objects_prompt += bb.get_sys_prompt() + ". "

        user_prompt = target_prompt + objects_prompt + prompt

        with open('prompts/corrections/system_prompt.txt', 'r') as f:
            system_prompt = f.read()

        prompts_json = json.load(open('prompts/prompts.json'))["prompts"]
        system_prompt = prompts_json["correction"]["system_prompt"]
        if self.use_gpt:

            # let's build the mesasges using the example prompts in the json file
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            for example in prompts_json["correction"]["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

            messages.append({"role": "user", "content": user_prompt})

            print(messages)

            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.4,
                max_tokens=512,

            )
            response = completion.choices[0].message.content
            response = json.loads(response)
            self.history.append([user_prompt, response])
            print(response)
            try:
                math_x = response["math_expression_x"]
                math_y = response["math_expression_y"]

                x = int(get_result(math_x))
                y = int(get_result(math_y))

                # clip the point within the image
                x = min(max(0, x), img.shape[1])
                y = min(max(0, y), img.shape[0])

            except:
                x = target[1]
                y = target[0]
            print(x, y)
            cot = response["chain_of_thought"]

            return Point(x, y), bbs, cot, response
        else:
            # use llama cpp
            grammar = LlamaGrammar.from_file("grammar/grammar.gbnf", verbose=False)
            output = self.model(convert_to_template(user_prompt, system_prompt, self.template),
                                max_tokens=1024*3, temperature=0.2, grammar=grammar, repeat_penalty=1.1)['choices'][0]['text']

            response = json.loads(output)
            x = int(response["correct_x"])
            y = int(response["correct_y"])
            cot = response["chain_of_thought"]
            return Point(x, y), bbs, cot, response

    def rel_or_abs(self, prompt, img):

        prompts_json = json.load(open('prompts/prompts.json'))["prompts"]
        system_prompt = prompts_json["rel_or_abs"]["system_prompt"]

        if self.use_gpt:
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            for example in prompts_json["rel_or_abs"]["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=messages,
                temperature=1.,
                max_tokens=1024
            )
            response = completion.choices[0].message.content
            response = json.loads(response)

        else:
            grammar = LlamaGrammar.from_file("grammar/rel_or_abs.gbnf", verbose=False)

            final_prompt = covert_to_template_with_examples(
                prompt, system_prompt, prompts_json["rel_or_abs"]["examples"], self.template)

            output = self.model(final_prompt, max_tokens=1024*3, temperature=0.7, grammar=grammar)['choices'][0]['text']

            response = json.loads(output)

        if response["position"] == "absolute":
            point, heatmap = self.predict_abs(img, response["objects"])
            return point, heatmap, None, None, response
        else:

            return self.predict_rel(prompt, img, response["objects"])

    def predict_abs(self, img, objects):
        return self.saygment.predict(img, objects=objects)

    def predict_rel(self, prompt, img, objects):
        return self.predict(prompt, img, False, objects)
