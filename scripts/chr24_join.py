# %%
import os, pickle, tqdm

from scipy.stats import beta
import pandas as pd
import numpy as np
from scipy.spatial import distance


# %%
def ci(f, n):
    k = np.round(n * f).astype(int)
    alpha = 0.05
    p_u, p_o = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])
    return p_u, p_o


# %%
def get_angles(attributions: pd.DataFrame):

    ncol = len(attributions.columns)
    angles = pd.DataFrame(index=attributions.index, columns=np.arange(ncol))
    mr = angles.index.get_level_values(3) == "MR"
    fnn = np.logical_not(mr)
    for i in range(ncol - 1):
        alone = ncol - i - 1
        angles.loc[fnn, i] = np.arctan2(
            np.sqrt(np.sum(attributions.iloc[fnn, slice(alone)] ** 2, 1)),
            attributions.iloc[fnn, alone],
        )
    angles.loc[fnn, ncol - 1] = np.pi / 2 - angles.loc[fnn, ncol - 2]
    angles.iloc[mr, :] = attributions.loc[mr, :]
    return angles


def get_candidates(attributions: pd.DataFrame):
    return (
        attributions.loc[(attributions.reset_index().A == 0).values]
        .groupby("att")
        .apply(lambda df: df.idxmax(1).value_counts())
    )


def applyCA(df: pd.DataFrame):
    unkind = np.where(df.index.get_level_values(5) == 0)
    distances = distance.pdist(df.values, "cosine")
    disunk = distance.squareform(distances)[unkind][0]
    assert np.isclose(disunk[unkind], 0)
    dest = np.argsort(disunk)
    pick = dest[df.index.get_level_values(5)[dest] != 0][0]
    if df.index.get_level_values(5)[pick] == 0:
        print(np.argsort(disunk)[:5], sorted(disunk)[:5])
    return df.index.get_level_values(5)[pick]


def get_closeAngle(attributions: pd.DataFrame):
    unk_atts = attributions.groupby(["P0", "F", "N", "delta", "att"]).apply(applyCA)
    return unk_atts.groupby("att").value_counts()


# %%
def run_corpus_from_partial(path):
    books = {}
    with open(os.path.join("../data", path, "bookNames.dat")) as fp:
        for line in fp:
            A, B, title = line.split()
            try:
                books[int(A)].append(int(B))
            except:
                books[int(A)] = [int(B)]
    tasks = []
    for A in sorted(books):
        for B in sorted(books[A]):
            tasks.append((A, B))
    candidates = {}
    for task in tqdm.tqdm(tasks, desc=path):

        attributions = pd.concat(
            [
                pd.read_csv(
                    f"../results/individual_results/{path}_A{task[0]}B{task[1]}_P{P0}.csv"
                )
                for P0 in [0, 1]
            ]
        )
        attributions["A"] = attributions["A"].astype(int)
        attributions.set_index(["P0", "F", "N", "delta", "att", "A", "B"], inplace=True)
        attributions.rename(columns=lambda x: int(x), inplace=True)
        attributions.sort_index(inplace=True)

        scores = pd.concat(
            [
                pd.read_csv(
                    f"../results/individual_results/scores_{path}_A{task[0]}B{task[1]}_P{P0}.csv"
                )
                for P0 in [0, 1]
            ]
        )
        scores.set_index(["P0", "F", "N", "delta", "att"], inplace=True)
        # scores.rename(columns=lambda x: int(x), inplace=True)
        # print(attributions.head())
        tot_books = len(attributions.loc[attributions.index[0][:-2]]) - 1
        min_int, _ = ci(scores["mR"].max(), tot_books)
        int_df = scores[scores["mR"] >= min_int]
        int_df.to_csv(
            f"../results/joined_results/scores_{path}_A{task[0]}B{task[1]}.csv"
        )
        attributions = (
            attributions.reset_index(["A", "B"])
            .loc[int_df.index]
            .reset_index()
            .set_index(["P0", "F", "N", "delta", "att", "A", "B"])
        )
        attributions.to_csv(
            f"../results/joined_results/{path}_A{task[0]}B{task[1]}.csv"
        )
        max_df = scores[scores["mR"] >= scores["mR"].max()]
        naive_att = (
            attributions.reset_index(["A", "B"])
            .loc[max_df.index]
            .reset_index()
            .set_index(["P0", "F", "N", "delta", "att", "A", "B"])
        )

        candidates_plain = get_candidates(attributions)
        candidates_naive = get_candidates(naive_att)
        angles = get_angles(attributions)
        candidates_ca = get_closeAngle(angles)
        candidates[task] = [
            candidates_plain,
            candidates_ca,
            candidates_naive,
        ]

    return candidates


# %%
for corpus in ["lemmas", "forms"]:
    with open(f"../results/{corpus}_results.pkl", "wb") as fp:
        pickle.dump(run_corpus_from_partial(corpus), fp)  #
