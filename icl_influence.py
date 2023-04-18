import os
import csv
import argparse
import pandas as pd
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from icl_datamodel import ICLDatamodel
from utils import Loader, get_model, set_seed, encode_subset, \
            inference, clear_gpu_resources


def collect_training_runs(args):
    if not torch.cuda.is_available():
        print("Inference requires GPU!")
        return
    if args.shot == 1:
        print("Refer to other file one-shot influences")
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
    k_shot = args.shot // int(2048 / max_seq_len)
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
    if os.path.exists(csv_name) and not pd.read_csv(csv_name)["iter"].empty:
        logger.info(f"Output for task '{args.task}' already exists, appending to it...")
        start = pd.read_csv(csv_name)["iter"].max()
        start += 1
    else:
        with open(csv_name, 'w') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
    logger.info(f"Starting at iter {start}/{args.iterations} for task: {args.task}, k_shot: {k_shot}")

    # 1. Get all subsets we are doing inference on
    dev_examples = loader.get_dev_examples()
    subsets = [loader.get_subset_for_inference(k_shot) for _ in range(args.iterations)]
    subsets = subsets[start:args.iterations]
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
    parser.add_argument("--shot", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of subset combinations we try")
    parser.add_argument("--data_dir", type=str, default="data-train400-dev200")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--cache_dir", type=str, default="/scratch/taing",
                        help="HF cache dir")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--result_file", type=str, default="influence_scores.jsonl")
    args = parser.parse_args()

    # Step 1: Collect subset training runs
    collection_csv = collect_training_runs(args)

    # Step 2: Compute In-context influences and Datamodel influences
    # (we conveniently write data for the Best-set seen from dev, which is a baseline)
    model = ICLDatamodel(collection_csv, args.data_dir)
    model.write_results(args)

    logger = logging.getLogger(__name__)
    logger.info(f"Coverage: {model.get_coverage()}; Acc: {model.get_acc_overall()}")
