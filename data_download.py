import json
import os
import datasets
import random
from collections import defaultdict
from utils import load_configs, set_seed

out_dir = "data"
set_seed(42)
TASK_CONFIG = load_configs()
splits = ["train", "validation"]  # HF Datasets don't usually publish test set answers
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def download_piqa():
    hf_identifier = "piqa"
    print(f"----------------------------Download: {hf_identifier}---------------------")
    hf_dataset = datasets.load_dataset(hf_identifier)
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(json.dumps(
                    {"index": index, "input": datapoint["goal"], "options": [datapoint["sol1"], datapoint["sol2"]],
                     "output": datapoint["label"]}))
            for line in lines:
                fout.write(line)
                fout.write("\n")

            print(len(lines))


def download_wic():
    hf_identifier = "superglue-wic"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("super_glue", "wic")
    optionMap = {
        "0": "false",
        "1": "true"
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(json.dumps({"index": index, "input": datapoint["sentence1"] + " [SEP] " + datapoint[
                    "sentence2"] + " [SEP] " + datapoint["word"],
                                         "options": list(optionMap.values()),
                                         "output": optionMap[str(datapoint["label"])]
                                         }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_boolq():
    hf_identifier = "superglue-boolq"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("super_glue", "boolq")
    optionMap = {
        "0": "no",
        "1": "yes"
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(
                    json.dumps({"index": index, "input": datapoint["passage"] + " [SEP] " + datapoint["question"],
                                "options": list(optionMap.values()),
                                "output": optionMap[str(datapoint["label"])]
                                }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_wsc():
    hf_identifier = "superglue-wsc"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("super_glue", "wsc.fixed")  # Fixed version
    optionMap = {
        "0": "false",
        "1": "true",
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(json.dumps({"index": index,
                                         "input": datapoint["text"] + " [SEP] " + datapoint["span1_text"] + " [SEP] " +
                                                  datapoint["span2_text"],
                                         "options": list(optionMap.values()),
                                         "output": optionMap[str(datapoint["label"])]
                                         }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_rte():
    hf_identifier = "superglue-rte"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("super_glue", "rte")
    optionMap = {
        "0": "true",
        "1": "false",
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(
                    json.dumps({"index": index, "input": datapoint["premise"] + " [SEP] " + datapoint["hypothesis"],
                                "options": list(optionMap.values()),
                                "output": optionMap[str(datapoint["label"])]
                                }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_ai2_arc_easy():
    hf_identifier = "ai2_arc_easy"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("ai2_arc", "ARC-Easy")
    optionMap = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                if datapoint["answerKey"] != "E":  # skip over q's with 5 choices
                    lines.append(json.dumps({
                        "index": index,
                        "input": datapoint["question"],
                        "options": datapoint["choices"]["text"],
                        "output": optionMap[str(datapoint["answerKey"])]
                    }))
            # write all lines
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_hellaswag():
    hf_identifier = "hellaswag"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("AlekseyKorshuk/hellaswag")
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(json.dumps({
                    "index": index,
                    "input": datapoint["activity_label"] + ": " + datapoint["ctx"],
                    "options": datapoint["endings"],
                    "output": int(datapoint["label"])
                }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_ai2_arc_challenge():
    hf_identifier = "ai2_arc_challenge"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    optionMap = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                if datapoint["answerKey"] == "E":
                    print(datapoint)
                else:
                    lines.append(json.dumps({
                        "index": index,
                        "input": datapoint["question"],
                        "options": datapoint["choices"]["text"],
                        "output": optionMap[str(datapoint["answerKey"])]
                    }))
            # write all lines
            for line in lines:
                fout.write(line)
                fout.write("\n")


def download_openbookqa():
    hf_identifier = "openbookqa"
    print(f"--------------------Download: {hf_identifier}-------------")
    hf_dataset = datasets.load_dataset("openbookqa", "main")
    optionMap = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    task_dir = os.path.join(out_dir, hf_identifier)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    else:
        print(f"Already exists: '{task_dir}'")
        return
    for split in splits:
        out_file = os.path.join(task_dir, f"{split if split != 'validation' else 'dev'}.jsonl")
        with open(out_file, "w") as fout:
            lines = []
            for index, datapoint in enumerate(hf_dataset[split]):
                lines.append(json.dumps({
                    "index": index,
                    "input": datapoint["question_stem"],
                    "options": datapoint["choices"]["text"],
                    "output": optionMap[datapoint["answerKey"]]
                }))
            for line in lines:
                fout.write(line)
                fout.write("\n")


def sample_train_and_dev(train_size=400, dev_size=200, data_dir="data"):
    """
    [Data sampling utility]
    For each task, read whole data split, get a sample by specified size, and write new files
    """
    out_dir = f"{data_dir}-train{train_size}-dev{dev_size}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    task_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # for each tasks
    for task_dir in task_dirs:
        print(f"--------------------Sample train and dev: {task_dir}-----------------")
        if not os.path.exists(os.path.join(out_dir, task_dir)):
            os.mkdir(os.path.join(out_dir, task_dir))

        for split in ["train", "dev"]:
            out_file = os.path.join(out_dir, task_dir, f"{split}.jsonl")
            if os.path.exists(out_file):
                print(f"Already exists: '{out_file}'")
                continue

            data = []
            split_files = [f for f in os.listdir(os.path.join(data_dir, task_dir)) if
                           f.endswith(".jsonl") and split in f]
            for file in split_files:
                with open(os.path.join(data_dir, task_dir, file), 'r') as f:
                    data.extend([json.loads(line) for line in f])
            # sample data
            # if classification, make sure get balanced number of examples per class
            k = min(dev_size, len(data)) if split == "dev" else min(train_size, len(data))
            if TASK_CONFIG[task_dir] == "classification":
                # get examples by label
                data_by_label = defaultdict(list)
                for dp in data:
                    data_by_label[dp["output"]].append(dp)
                sizes = [len(data_by_label[label]) for label in data_by_label]
                # sample balance
                remainder = k % len(data_by_label)
                sample = []
                for label in data_by_label:
                    # sometimes, number of label examples is fewer than k_sub
                    k_sub = k // len(data_by_label)
                    size = len(data_by_label[label])
                    # first array gets all remainder + number minor class can't get
                    if remainder > 0 or min(sizes) < k_sub:
                        k_sub += remainder
                        k_sub += k_sub - min(sizes)
                        remainder = 0
                    if size < k_sub:
                        sub_sample = random.sample(data_by_label[label], size)
                    else:
                        sub_sample = random.sample(data_by_label[label], k_sub)
                    sample.extend(sub_sample)
            else:
                sample = [data[i] for i in sorted(random.sample(range(len(data)), k))]
            random.shuffle(sample)
            out_writer = open(out_file, 'w')
            for d in sample:
                json.dump(d, out_writer)
                out_writer.write("\n")
            print(f"Successfully wrote '{out_file}': {len(sample)} examples out of {len(data)}")


def sample_test(data_dir: str, out_dir: str, test_size=500):
    """
    [Data sampling utility]
    Arguments:
        - data_dir: Original source
        - out_dir: Current source (same directory as train/dev)
    Sample a test set of certain size on unseen dev examples.
    If dev set is not enough, sample from unseen train.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    task_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    test_size = 500

    # for each tasks
    for task_dir in task_dirs:
        print(f"--------------------Sample test: {task_dir}-----------------------")
        # source data
        split = "dev"
        fsplit = os.path.join(data_dir, task_dir, f"{split}.jsonl")
        with open(fsplit, 'r') as f:
            data = [json.loads(line) for line in f]

        # read existing data in out_dir
        fsplit_seen = os.path.join(out_dir, task_dir, f"{split}.jsonl")
        data_seen = []
        with open(fsplit_seen, 'r') as f:
            data_seen.extend([json.loads(line) for line in f])

        data_ids = [d["index"] for d in data]
        data_seen_ids = [d["index"] for d in data_seen]

        # pool = [d for d in data if d["index"] in (set(data_ids) - set(data_seen_ids))]
        pool = []
        for d in data:
            if d["index"] in (set(data_ids) - set(data_seen_ids)):
                d["index"] = f"dev_{d['index']}"
                pool.append(d)
        print(f"Original dev: {len(data)}")
        print(f"Dev pool: {len(pool)}")

        # if dev does not meet test_size, sample unseen train
        if len(pool) < test_size:
            fsplit = os.path.join(data_dir, task_dir, f"train.jsonl")
            with open(fsplit, 'r') as f:
                data = [json.loads(line) for line in f]
            # read existing data in out_dir
            fsplit_seen = os.path.join(out_dir, task_dir, f"train.jsonl")
            data_seen = []
            with open(fsplit_seen, 'r') as f:
                data_seen.extend([json.loads(line) for line in f])
            data_ids = [d["index"] for d in data]
            data_seen_ids = [d["index"] for d in data_seen]
            for d in data:
                if d["index"] in (set(data_ids) - set(data_seen_ids)):
                    d["index"] = f"train_{d['index']}"
                    pool.append(d)
            print(f"Original train: {len(data)} - Add extra train: {len(set(data_ids) - set(data_seen_ids))}")

        print(f"Full pool: {len(pool)}")
        if (len(pool)) == 0:
            print("No data for test - SKIP")
            continue
        # sample data
        # if classification, make sure get balanced number of examples per class
        k = min(test_size, len(pool))
        if TASK_CONFIG[task_dir] == "classification":
            # get examples by label
            data_by_label = defaultdict(list)
            for dp in pool:
                data_by_label[dp["output"]].append(dp)
            sizes = [len(data_by_label[label]) for label in data_by_label]
            # sample balance
            remainder = k % len(data_by_label)
            sample = []
            for label in data_by_label:
                # sometimes, number of label examples is fewer than k_sub
                k_sub = k // len(data_by_label)
                size = len(data_by_label[label])
                # first array gets all remainder + number minor class can't get
                if remainder > 0 or min(sizes) < k_sub:
                    k_sub += remainder
                    k_sub += k_sub - min(sizes)
                    remainder = 0
                if size < k_sub:
                    sub_sample = random.sample(data_by_label[label], size)
                else:
                    sub_sample = random.sample(data_by_label[label], k_sub)
                sample.extend(sub_sample)
        else:
            sample = [pool[i] for i in sorted(random.sample(range(len(pool)), k))]

        random.shuffle(sample)
        fname = "test.jsonl"
        out_file = os.path.join(out_dir, task_dir, fname)

        # delete existed file
        if os.path.exists(out_file):
            print(f"Already exists: '{out_file}'")
            continue
        # write
        with open(out_file, 'w') as writer:
            for d in sample:
                json.dump(d, writer)
                writer.write("\n")
        print(f"Successfully wrote '{out_file}': {len(sample)} examples in pool of {len(pool)}")

        # check that test has no overlapps with dev/train
        with open(os.path.join(out_dir, task_dir, "dev.jsonl")) as f:
            dev_ids = set([f"dev_{json.loads(line)['index']}" for line in f])
        with open(os.path.join(out_dir, task_dir, "train.jsonl")) as f:
            train_ids = set([f"train_{json.loads(line)['index']}" for line in f])
        with open(os.path.join(out_dir, task_dir, "test.jsonl")) as f:
            test_ids = set([json.loads(line)["index"] for line in f])
        assert dev_ids.isdisjoint(test_ids), f"Overlapped examples: {dev_ids & test_ids}"
        assert train_ids.isdisjoint(test_ids), f"Overlapped examples: {train_ids & test_ids}"


def get_data():
    # Write the following datasets to out_dir
    download_piqa()
    # download_boolq()
    # download_wic()
    # download_wsc()
    # download_rte()
    # download_ai2_arc_easy()
    # download_hellaswag()
    # download_ai2_arc_challenge()
    # download_openbookqa()

    train_size = 400
    dev_size = 200
    sample_train_and_dev(train_size=train_size, dev_size=dev_size, data_dir="data")
    sample_test(data_dir="data", out_dir=f"data-train{train_size}-dev{dev_size}")


if __name__ == '__main__':
    get_data()
