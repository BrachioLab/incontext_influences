import os
import copy
import json
import torch
import random
import numpy as np
from typing import List, Dict
from collections import defaultdict
from templates import TEMPLATES, SEP, apply_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig, \
    OPTForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM, GPTNeoXConfig, \
    GPTJForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM


def load_configs():
    with open("task_config.json", "r") as f:
        config = json.load(f)
    return config


TASK_CONFIG = load_configs()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def clear_gpu_resources():
    import gc, time
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def get_model(args):
    """
    Initialize tokenizer and model
    """

    model_name = args.model_name_or_path

    if "opt" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m", cache_dir=args.cache_dir)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.padding_size = "left"
        config = AutoConfig.from_pretrained(model_name)
        config.vocab_size = tokenizer.vocab_size
        config.eos_token_id = tokenizer.eos_token_id
        config.max_seq_length = 2048
        model = OPTForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=args.cache_dir,
            # max_memory={0: "55GiB", 1: "55GiB", "cpu": "50GiB"}
        )
        model.eval()
    elif "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.config.max_seq_length = 1024
        model.eval()
    elif "gpt-neox" in model_name.lower():
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name, cache_dir=args.cache_dir)
        config = GPTNeoXConfig.from_pretrained(model_name)
        config.is_decoder = True
        config.max_seq_length = 2048
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            # max_memory={0: "55GiB", 1: "55GiB", "cpu": "50GiB"},
            cache_dir=args.cache_dir
        )
        model.eval()
    elif "gpt-j" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_name)
        config.pad_token_id = tokenizer.eos_token_id
        config.max_seq_length = 2048
        model = GPTJForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            revision="float16",  # half precision
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=args.cache_dir
        )
        model.eval()
    elif "llama" in model_name.lower():
        weight_dir, _, _ = model_name.rpartition("/")
        tokenizer = LlamaTokenizer.from_pretrained(os.path.join(weight_dir, "tokenizer"))
        tokenizer.padding_side = "left"
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).cuda()
        model.config.pad_token_id = model.config.eos_token_id
        model.config.max_seq_length = 2048
    else:
        raise NotImplementedError

    return tokenizer, model


def encode_subset(subset, task, tokenizer):
    """
    subset: object with "data" and "subset_idx" keys, where "data" is list of dps
    Return dict with (index: encoded ids) key-value pairs
    """
    dem_ids = {}
    for dp in subset["data"][::-1]:
        if task == "piqa" or TASK_CONFIG[task] == "multi-choice":
            ans = TEMPLATES[task][1].format(dp["options"][dp["output"]])  # expand option
        else:
            ans = dp["output"]
        demonstration = "{}{}{}".format(dp["input"], ans, SEP)
        ids = tokenizer.encode(
            text=demonstration,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        dem_ids[dp["index"]] = ids
    return dem_ids


def inference(demonstration_ids: dict, test_ex, task, max_seq_len, model, tokenizer, debug=False):
    """
    demonstration_ids: encoded subsets which we load into context window from right to left
    test_ex: query, last line that follows all demonstrations
    """
    # query
    query_end = TEMPLATES[task][2]
    query = "{}{}".format(test_ex["input"], query_end)
    query_ids = tokenizer.encode(
        text=query,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).cuda()

    max_option_len = max(
        [tokenizer.encode(" " + opt, return_tensors="pt", padding=False, add_special_tokens=False).shape[-1] for opt in
         test_ex["options"]])

    # add demonstrations within context window limit
    input_ids = query_ids
    subset_actual = []
    for index, ids in demonstration_ids.items():
        curr_len = input_ids.shape[1]
        demonstration_len = ids.shape[1]
        if curr_len + demonstration_len + max_option_len <= max_seq_len:
            input_ids = torch.cat((ids.cuda(), input_ids), dim=1)
            subset_actual.insert(0, index)  # store actual index
        else:
            break

    option_count = len(test_ex["options"])
    ip_ids = torch.zeros((option_count, max_seq_len)).long()
    # pad option lavel with -100 so CE ignores it
    label_ids = -100 * torch.ones((option_count, max_seq_len)).long()
    s = input_ids[0].long()  # s = source (input prompt)

    if debug:
        context = tokenizer.decode(s.squeeze())
        print(f"[CONTEXT]\n{context}")

    for i, option in enumerate(test_ex["options"]):
        # t = target (an option)
        t = tokenizer.encode(" " + option, padding=False, return_tensors="pt", add_special_tokens=False)
        t = t[0].long()
        ip_ids[i, :len(s)] = s
        ip_ids[i, len(s):len(s) + len(t)] = t  # ip_ids[i] = [demonstration + query + options[i]...0]
        label_ids[i, len(s):len(s) + len(t)] = t  # label_ids[i] = [-100...options[i]...-100]

    # Get logits from model
    with torch.no_grad():
        logits = model(
            input_ids=ip_ids.cuda(),
            return_dict=True
        ).logits.cpu()[:, :-1].contiguous()

    # Get cross entropies given logits
    logit_shape = logits.shape
    logits = logits.view(-1, logit_shape[-1])
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    ce_list = loss_fct(logits.float(), label_ids[:, 1:].contiguous().view(-1))  # CrossEntropy[(C), (N, C)]
    ce_list = ce_list.view(option_count, max_seq_len - 1).sum(dim=1).squeeze().tolist()

    # Get answer
    min_index = np.argmin(ce_list)
    if task == "piqa" or TASK_CONFIG[task] == "multi-choice":
        prediction = min_index
    else:
        prediction = test_ex["options"][min_index]
        prediction = TEMPLATES[task][2] + " " + prediction  # "Answer: " + pred

    return prediction, subset_actual


class Loader:
    def __init__(self, data_dir: str, task: str):
        self.data_dir = data_dir
        self.task = task
        self.data_template = DataSplit(self.task, "train", self.data_dir)
        # apply template
        for dp in self.data_template:
            apply_template(dp, self.task)
        # get label example group once, for sampling calls later
        self.data_by_label = defaultdict(list)
        for dp in self.data_template:
            self.data_by_label[dp["output"]].append(dp)

    def get_subset_for_inference(self, k_shot: int) -> Dict:
        """Get random subset k demonstrations, ensure even number of samples by class label"""
        n = len(self.data_template)
        # sample a balanced subset
        if self.data_template.task_type == "classification":
            # get roughly equal examples for each label
            remainder = k_shot % len(self.data_by_label)
            subset = []
            for label in self.data_by_label:
                k_sub = k_shot // len(self.data_by_label)
                if remainder > 0:
                    k_sub += 1
                    remainder -= 1
                # random - sample with label group
                sample = random.sample(self.data_by_label[label], k_sub)
                subset.extend(sample)

            # random - shuffle subset to alternate label examples
            random.shuffle(subset)
            subset_idx = [dp["index"] for dp in subset]
        else:
            idx = np.random.choice(n, k_shot, replace=False).astype(int)
            subset = [self.data_template[i] for i in idx]
            subset_idx = [dp["index"] for dp in subset]

        return {
            "data": subset,
            "subset_idx": subset_idx,
        }

    def get_examples_oneshot(self) -> List[Dict]:
        """For 1-shot, we pick all train data points for inference instead of random sampling"""
        ret = []
        data = DataSplit(self.task, "train", self.data_dir)
        for dp in data:
            apply_template(dp, self.task)
            ret.append({
                "data": [dp],
                "subset_idx": [dp["index"]],
            })
        return ret

    def get_dev_examples(self) -> List[Dict]:
        split = "dev"
        data = DataSplit(self.task, split, self.data_dir)
        for dp in data:
            apply_template(dp, self.task)
        return [dp for dp in data]

    def get_test_examples(self) -> List[Dict]:
        split = "test"
        data = DataSplit(self.task, split, self.data_dir)
        for dp in data:
            apply_template(dp, self.task)
        return [dp for dp in data]

    def get_subset_by_indices(self, indices: list) -> List[Dict]:
        """Get subset by index list"""
        indices = [int(i) for i in indices]  # convert to int
        dps = [dp for dp in self.data_template if dp["index"] in indices]
        # get it in the same order
        order = {key: i for i, key in enumerate(indices)}
        dps_sorted = sorted(dps, key=lambda dp: order[dp["index"]])
        return dps_sorted

    def get_dp(self, index) -> Dict:
        index = int(index)
        for dp in self.data_template:
            if index == dp["index"]:
                return dp
        print(f"Could not find dp with index {index}")

    def __iter__(self):
        for dp in self.data_template:
            yield dp

    def __len__(self):
        return len(self.data_template)


class DataSplit:
    def __init__(self, task, split, data_dir):
        self.task = task
        self.task_type = TASK_CONFIG[self.task]
        self.split = split
        self.data_dir = data_dir
        self.data = self.read_data_split(self.task, split, self.data_dir)
        # Additionally, load templated versions of data
        self.data_template = copy.deepcopy(self.data)
        for dp in self.data_template:
            apply_template(dp, self.task)

    def get_indices(self) -> List[str]:
        return [str(dp["index"]) for dp in self.data]

    @staticmethod
    def read_data_split(task: str, split: str, data_dir) -> List[Dict]:
        """Load full data split. Currently get all examples across all seed files."""
        task_dir = os.path.join(data_dir, task)
        if not os.path.exists(task_dir):
            print("Data not found at '%s'" % task_dir)
            raise
        split_files = [f for f in os.listdir(task_dir) if split in f]
        data = []
        for file in split_files:
            with open(os.path.join(task_dir, file), 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return json.dumps(self.data, indent=2)

    def __iter__(self):
        for dp in self.data:
            yield dp
