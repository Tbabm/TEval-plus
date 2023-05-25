# encoding=utf-8

"""
Collect and compare the performance differences when generating from buggy versions and fixed versions.
"""

import os
from collections import OrderedDict, defaultdict
from typing import Dict

import fire

from .metrics import *


def read_result_df(data_dir: str, src: str, idx: int, system: str, result_dir: str):
    result_file = os.path.join(data_dir, f"evosuite_{src}_tests/{idx}/{system}_generated/{result_dir}/full_test_data.csv")
    print(result_file)
    result_df = pd.read_csv(result_file)
    # deduplicate
    result_df.drop_duplicates(subset=["project", "bug_num", "focal_method", "docstring", "test_prefix"])
    return result_df


def cal_one_result(data_dir: str, src: str, idx: int, system: str, result_dir: str, scorers: Dict[str, Scorer]):
    result_df = read_result_df(data_dir, src, idx, system, result_dir)
    result = OrderedDict()
    for name, scorer in scorers.items():
        result[name] = scorer.score(result_df)
    return result


def cal_result(data_dir="data"):
    scorers = OrderedDict({
        "BugFound": BugFound(),
        "FPRate": FPR(),
        "Precision": Precision(),
        "TPs": TPs(),
        "FPs": FPs()
    })
    src_names = ["fixed", "buggy"]
    dfs = []
    all_dfs = defaultdict(dict)
    for src in src_names:
        for result_dir in ["results", "merged_results"]:
            exp_results = []
            for i in range(1, 11):
                result = cal_one_result(data_dir, src, i, "toga", result_dir, scorers)
                exp_results.append(result)
            assert len(exp_results) == 10
            src_df = pd.DataFrame(exp_results)
            all_dfs[src][result_dir] = src_df
            dfs.append(src_df.mean())
    indexes = ["TEval@fixed_overfiltering", "TEval@fixed", "TEval@buggy_overfiltering", "TEval@buggy"]
    df = pd.DataFrame(dfs, index=indexes)
    print(df)
    print("dump result")
    df.to_csv('results/rq1_2.csv')


if __name__ == "__main__":
    fire.Fire({
        "cal_result": cal_result,
    })
