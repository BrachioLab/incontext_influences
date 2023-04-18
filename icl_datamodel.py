import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from utils import DataSplit


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def subset_idx_to_array(text: str) -> List[str]:
    """Convert string '1,2,3' to list [1, 2, 3]"""
    text = str(text)
    if "," in text:
        return text.split(",")
    else:
        return [text]


def append_to_jsonl(filename: str, data: dict) -> bool:
    """
    Appends the given data to a JSONL file, if the same data doesn't already exist in the file.
    """
    with open(filename, "r") as fp:
        # Check if the data already exists in the file
        for line in fp:
            existing_data = json.loads(line)
            if existing_data == data:
                return False

    # Append the data to the file
    with open(filename, "a+") as fp:
        fp.write(json.dumps(data))
        fp.write("\n")

    return True


class ICLDatamodel:
    def __init__(self, out_file, data_dir):
        self.data_dir = data_dir
        self.task = out_file.split("/")[-1].split("_shot")[0]
        self.model = out_file.split("_model")[1].split("_dir")[0]
        self.shot = int(out_file.split("_model")[0].split("_shot")[1])
        self.train = DataSplit(self.task, "train", data_dir)
        self.df = pd.read_csv(out_file)

    def get_incontext_scores(self) -> Dict[str, float]:
        """Get train example influence scores based on Accuracy"""
        X = self.train.get_indices()
        ans = {x: 0.0 for x in X}
        non_coverage = []

        # get f(S) 's
        infl_x_i = defaultdict(list)
        infl_x_i_complement = defaultdict(list)
        for _, row in self.df.iterrows():
            S_i = subset_idx_to_array(row["subset_idx"])
            S_i_complement = set(X) - set(S_i)
            assert (len(S_i_complement) < len(X))
            for x in S_i:
                infl_x_i[x].append(row["val_acc"])
            for x_c in S_i_complement:
                infl_x_i_complement[x_c].append(row["val_acc"])

        # compute influences
        for x in X:
            score = np.mean(infl_x_i[x]) - np.mean(infl_x_i_complement[x])
            if not np.isnan(score):
                ans[x] = np.round(score, 7)
            else:
                non_coverage.append(x)

        if len(non_coverage) > 0:
            print(f"[{self.task} - {self.model}] Empty influence scores for: {non_coverage}")
        return ans

    def get_datamodel_scores(self, alpha=0.0001) -> Dict[str, float]:
        """Linear datamodel with l1 regression"""
        indices = self.train.get_indices()
        subsets = [ss.split(",") for ss in self.df["subset_idx"].tolist()]
        test_accs = self.df["val_acc"].tolist()

        # Create df of idx columns
        df_train = pd.DataFrame(0, index=range(len(subsets)), columns=indices)
        for i, subset in enumerate(subsets):
            for idx in subset:
                df_train.loc[i, str(idx)] = 1
        df_train = df_train.reindex(sorted(df_train.columns), axis=1)  # sort

        # Get split
        test_size = 0.2  # keep most train data for influence estimation
        X_train, X_test, y_train, y_pred = train_test_split(df_train, test_accs,
                                                            test_size=test_size, random_state=42)

        # Train linear model
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        coefs = model.coef_
        coef_dict = dict(zip(df_train.columns, coefs))
        # y_pred = model.predict(X_test)  # get predictions
        return coef_dict

    def get_coverage(self) -> float:
        """Coverage score as average of x_i appearance"""
        X = self.train.get_indices()
        coverages = {x: 0 for x in X}
        for _, row in self.df.iterrows():
            for x_i in subset_idx_to_array(row["subset_idx"]):
                coverages[x_i] += 1
        return np.mean(list(coverages.values()))

    def get_acc_overall(self) -> float:
        """Average accuracy over all subsets"""
        return np.round(self.df["val_acc"].mean(), 5)

    def write_results(self, args) -> None:
        if not os.path.exists(args.result_file):
            with open(args.result_file, 'w') as f:
                pass

        coverage = self.get_coverage()
        accuracy = self.get_acc_overall()

        # store in-context influences
        ic_dict = self.get_incontext_scores()
        ic_dict = dict(sorted(ic_dict.items(), key=lambda item: item[1]))
        ic_dict = {k: round(v, 6) for k, v in ic_dict.items()}
        line = {
            "method": "incontext_influence",
            "model": self.model,
            "task": self.task,
            "shot": self.shot,
            "coverage": coverage,
            "acc": accuracy,
            "data_dir": self.data_dir,
            "scores": ic_dict
        }
        if append_to_jsonl(args.result_file, line):
            print(f"Wrote in-context influences to '{args.result_file}'")
        else:
            print(f"In-context influences already written")

        # store datamodel influences
        dm_dict = self.get_datamodel_scores()
        dm_dict = dict(sorted(dm_dict.items(), key=lambda item: item[1]))
        dm_dict = {k: round(v, 6) for k, v in dm_dict.items()}
        line = {
            "method": "datamodel_influence",
            "model": self.model,
            "task": self.task,
            "shot": self.shot,
            "coverage": coverage,
            "acc": accuracy,
            "data_dir": self.data_dir,
            "scores": dm_dict
        }
        if append_to_jsonl(args.result_file, line):
            print(f"Wrote datamodel influences to '{args.result_file}'")
        else:
            print(f"Datamodel influences already written")
