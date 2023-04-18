import os
import csv
import argparse
import json
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils import Loader, get_model, set_seed, \
    clear_gpu_resources, encode_subset, inference

# maximal number of shots in context window
SHOT_MAP = {
    "superglue-rte": 12,
    "superglue-boolq": 10,
    "hellaswag": 18,
    "piqa": 38,
    "superglue-wic": 32,
    "superglue-wsc": 32,
    "ai2_arc_challenge": 46,
    "ai2_arc_easy": 52,
    "openbookqa": 52
}

BASELINES = [
    "incontext_influence_positive",
    "incontext_influence_negative",
    "incontext_influence_neutral",
    "datamodel_influence",
    "oneshot_influence",
    "best_set",
    "random"
]


def get_top_dict(args, max_seq_len):
    """
    Nested dict to get top train points per [task][test]
    + add items in inverse order from worse -> better
    """
    task_shot = SHOT_MAP[args.task] // int(2048 / max_seq_len)
    ret = {}
    # Perplexity
    if "perplexity" in args.resource_file:
        data = pd.read_csv(args.resource_file)
        for task, d in data.groupby("task"):
            ret[task] = {}
            for model, d2 in d.groupby("model"):
                ret[task][model] = []
                for _, row in d2[:task_shot].iterrows():
                    # add from smallest similarity to biggest
                    ret[task][model].insert(0, row["index"])
        for task in ret:
            assert (len(ret[task]) == len(MODELS_ALL))
            for model in ret[task]:
                assert (len(ret[task][model]) == SHOT_MAP[task])
    # Influence
    elif "influence" in args.resource_file:
        with open(args.resource_file, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                if line["method"] not in args.method:
                    continue
                task = line["task"]
                model = line["model"]
                scores = dict(sorted(line["scores"].items(), key=lambda item: item[1]))
                top = list(scores.keys())
                if "negative" in args.method:
                    top = top[:task_shot]
                    top = top[::-1]  # inverse so worst examples come last
                elif "neutral" in args.method:
                    top = top[int((len(top) - task_shot) / 2): int((len(top) + task_shot) / 2)]  # select middle k shot
                else:  # default
                    top = top[-task_shot:]  # select top k shot at the end, don't need to inverse
                if task not in ret:
                    ret[task] = {}
                ret[task][model] = top
    elif "random" == args.method:
        return ret, task_shot
    elif "best_set" == args.method:
        with open(args.resource_file, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                if line["method"] not in args.method:
                    continue
                task = line["task"]
                model = line["model"]
                top = line["scores"][-task_shot:]
                if task not in ret:
                    ret[task] = {}
                ret[task][model] = top
    # Average Validation Distance
    elif args.method in ["dev_avg_roberta-large", "dev_avg_mpnet_avg"]:
        with open(args.resource_file, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                task = line["task"]
                scores = dict(sorted(line["rank_average_dev"].items(), key=lambda item: item[1]))
                top = list(scores.keys())
                top = top[-task_shot:]  # select top k shot at the end, don't need to inverse
                if task not in ret:
                    ret[task] = {}
                ret[task] = top
    else:
        raise NotImplementedError
    return ret, task_shot


def check_line_exists(out_csv, data):
    with open(out_csv, 'r') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if all(str(row[key]) == str(value) for key, value in data.items()):
                return True
    return False


def main(args):
    if args.method not in BASELINES:
        print("Method not valid")
        return

    set_seed(args.seed)

    model_abbr = args.model_name_or_path.split('/')[-1]
    fieldnames = ["method", "model", "task", "subset_idx", "k_shot", "val_acc"]
    out_csv = os.path.join(args.out_dir, "baseline.csv")
    # check if out folder exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # check if csv exists
    if not os.path.exists(out_csv):
        with open(out_csv, 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()

    # load tokenizer + model + exemplars
    tokenizer, model = get_model(args)
    max_seq_len = model.config.max_seq_length
    top_dict, task_shot = get_top_dict(args, max_seq_len)

    # Loader with original data dir and helpers
    loader = Loader(args.data_dir, args.task)

    if args.split == "test":
        val_examples = loader.get_test_examples()
    elif args.split == "dev":
        val_examples = loader.get_dev_examples()
    else:
        raise

    train_indices = [dp["index"] for dp in loader.data_template]

    # iterate to get subsets
    test_indices = [dp["index"] for dp in val_examples]
    pred = []
    true = []
    for val_ex in tqdm(val_examples):
        test_idx = val_ex["index"]
        if args.method in ["perplexity", "incontext_influence_positive", "incontext_influence_negative",
                           "incontext_influence_neutral", "datamodel_influence", "oneshot_influence", "best_set"]:
            top_k = top_dict[args.task][model_abbr]
        # randomly choose demonstrations
        elif args.method == "random":
            top_k = random.choices(train_indices, k=task_shot)
        elif args.method in ["dev_avg_roberta-large", "dev_avg_mpnet_avg"]:
            top_k = top_dict[args.task]
        else:
            raise

        # 1) Encode subset from right to left (do this once)
        subset = {
            "data": loader.get_subset_by_indices(top_k),
            "subset_idx": top_k,
        }

        # 2) Evaluate
        demonstration_ids = encode_subset(subset, args.task, tokenizer)
        prediction, subset_actual = inference(demonstration_ids, val_ex, args.task, max_seq_len, model, tokenizer)

        true.append(val_ex["output"])
        pred.append(prediction)

    # Write
    acc = accuracy_score(true, pred)
    data = {
        "method": args.method,
        "model": model_abbr,
        "task": args.task,
        "subset_idx": ",".join(str(x_i) for x_i in top_k),
        "k_shot": task_shot,
        "val_acc": acc
    }
    if not check_line_exists(out_csv, data):
        with open(out_csv, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writerow(data)
        print(f"[{args.method}-{model_abbr}-{args.task}] Accuracy: {acc}")
        print(f"Wrote eval results for ({args.method}, {args.task}, {model_abbr}) at '{out_csv}'")
    else:
        print(f"(Line exists) [{args.method}-{model_abbr}-{args.task}] Accuracy: {acc}")

    clear_gpu_resources()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="superglue-rte",
                        help="List of tasks to run counterfactual for")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-6.7b",
                        help="HF model identifier. For LLaMA, please specify directory of model weights")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to do evaluation on")
    parser.add_argument("--resource_file", type=str, default="influence_scores.jsonl",
                        help="Influence score output file")
    parser.add_argument("--data_dir", type=str, default="data-train400-dev200",
                        help="Data directory")
    parser.add_argument("--out_dir", type=str, default="baseline",
                        help="Output directory")
    parser.add_argument("--method", type=str, default="incontext_influence_positive",
                        help="Baseline type")
    parser.add_argument("--cache_dir", type=str, default="/scratch/taing",
                        help="HF cache directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
