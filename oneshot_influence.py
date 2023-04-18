import os
import csv
import json
import argparse
import pandas as pd
import torch
import logging
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils import Loader, DataSplit, get_model, set_seed, encode_subset, \
            inference, clear_gpu_resources


def collect_training_runs(args):
    if not torch.cuda.is_available():
        print("Inference requires GPU!")
        return
    if not os.path.exists(args.data_dir):
        print(f"Data not found in '{args.data_dir}'")
        return
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    set_seed(args.seed)

    # load tokenizer + model
    tokenizer, model = get_model(args)
    max_seq_len = model.config.max_seq_length
    model_abbr = args.model_name_or_path.split('/')[-1]
    k_shot = 1
    loader = Loader(args.data_dir, args.task)

    # prepare to write data
    start = 0
    csv_name = f"{args.task}_shot{k_shot}_model{model_abbr}_dir{args.data_dir}_seed{args.seed}.csv"
    csv_name = os.path.join(args.out_dir, csv_name)
    fieldnames = ["iter", "subset_idx", "k_shot", "val_acc"]

    # init log
    log_name = csv_name.replace(f"{args.out_dir}/", "log/").replace(".csv", ".log")
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(args)

    # get checkpoint of results
    start = 0
    seen = []
    if os.path.exists(csv_name) and not pd.read_csv(csv_name)["iter"].empty:
        logger.info(f"Output for task '{args.task}' already exists, appending to it...")
        seen = set(seen + (pd.read_csv(csv_name)["subset_idx"].tolist()))
        start = len(seen)
    else:
        with open(csv_name, 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()

    # 1. Get all subsets we are doing inference on
    dev_examples = loader.get_dev_examples()
    dev_examples = random.sample(dev_examples, min(len(dev_examples), 100))  # randomly choose less than 200 validations for speed

    subsets = loader.get_examples_oneshot()
    subsets = [subset for subset in subsets if subset["subset_idx"][0] not in seen]

    logger.info(f"One-shot influences calculated for {start} train examples")
    logger.info(f"Remaining subsets: {len(subsets)}")

    # 2. Inference on dev
    logger.info(f"Train size: {len(loader)} - Dev size: {len(dev_examples)}")
    for subset_i in tqdm(range(len(subsets)), desc=f"[{args.task}] Subset"):
        # a. Encode subset from right to left (once)
        demonstration_ids = encode_subset(subsets[subset_i], args.task, tokenizer)

        # b. Evaluate
        pred = []
        true = []
        for dev_i, dev_ex in enumerate(dev_examples):
            debug = True if subset_i == 0 and dev_i == 0 else False

            prediction, subset_actual = inference(demonstration_ids, dev_ex,
                                args.task, max_seq_len, model, tokenizer, debug)

            if debug: print(f"[Iter {start} - dev {dev_i}] Pred:{prediction}\tTruth:{dev_ex['output']}")

            true.append(dev_ex["output"])
            pred.append(prediction)

        # write results
        with open(csv_name, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            acc = accuracy_score(true, pred)
            writer.writerow({
                "iter": start,
                "subset_idx": ",".join(str(v) for v in subsets[subset_i]["subset_idx"]),
                "k_shot": k_shot,
                "val_acc": acc
            })

            start += 1  # increment iter

        print(f"[{args.task}-Model {model_abbr}-Iter {start}] Accuracy: {acc}")
        # end of subset

    logger.info(f"Finished writing results for ({args.task}, {model_abbr}) at '{csv_name}'")
    # end of task

    clear_gpu_resources()

    return csv_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default="superglue-rte",
                        help="Task of interest")
    parser.add_argument("--model_name_or_path", type=str,
                        default="facebook/opt-6.7b",
                        help="HF model identifier. For LLaMA, please specify directory of model weights")
    parser.add_argument("--data_dir", type=str, default="data_new-train400-test200")
    parser.add_argument("--out_dir", type=str, default="out_oneshot")
    parser.add_argument("--cache_dir", type=str, default="/scratch/taing",
                        help="HF cache dir")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--result_file", type=str, default="influence_scores.jsonl")
    args = parser.parse_args()

    # Step 1: Collect subset training runs
    collection_csv = collect_training_runs(args)

    # Step 2: Store One-shot influences
    df = pd.read_csv(collection_csv)
    mean_acc = df['val_acc'].mean()
    model = collection_csv.split("_model")[1].split("_dir")[0]
    # init train idx influence dict
    train = DataSplit(args.task, "train", args.data_dir)
    oneshot_dict = {str(x): 0.0 for x in train.get_indices()}

    read_dict = dict(zip(df['subset_idx'], df['val_acc']))
    for k, v in read_dict.items():
        oneshot_dict[str(k)] = v
    # Check current results
    # if file is created
    if not os.path.exists(args.result_file):
        with open(args.result_file, 'w') as f:
            pass
    # if duplicate line
    found = False
    with open(args.result_file, 'r') as fp:
        for line in fp:
            line = json.loads(line)
            if line["task"] == args.task and line["model"] == model and line["shot"] == 1:
                found = True
    # write!
    logger = logging.getLogger(__name__)
    if found:
        print(f"Oneshot influences already calculated for ({args.task}, {model})")
    else:
        with open(args.result_file, "a+") as fp:
            fp.write(json.dumps({
                "method": "oneshot_influence",
                "model": model,
                "task": args.task,
                "shot": 1,
                "coverage": 1,
                "acc": mean_acc,
                "data_dir": args.data_dir,
                "scores": dict(sorted(oneshot_dict.items(), key=lambda item: item[1]))
            }))
            fp.write("\n")
        logger.info(f"Coverage: 1; Acc: {mean_acc}")
        logger.info(f"Wrote oneshot influences to '{args.result_file}'")

    # Step 3: Evaluate one shot along with other baselines
    import subprocess
    for method in ["oneshot_influence", "best_set", "random"]:
        eval_args = [
            "python3",
            "evaluate.py",
            f"--task={args.task}",
            f"--model_name_or_path={args.model_name_or_path}",
            "--split=test",
            f"--data_dir={args.data_dir}",
            f"--method={method}",
            f"--cache_dir={args.cache_dir}",
        ]

    result = subprocess.run(eval_args, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        logger.info(result.stdout)
    else:
        print(f"Command failed with return code {result.returncode}: {result.stderr}")
