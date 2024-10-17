# %%
import os, glob, pickle, re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_theme("paper", "ticks")
sns.set_palette("colorblind")

# %%
with open("../results/forms_results.pkl", "rb") as fp:
    original = pickle.load(fp)

with open("../results/lemmas_results.pkl", "rb") as fp:
    lemmatised = pickle.load(fp)


# %%
def get_tmp(tmp1, name):
    if isinstance(tmp1, pd.DataFrame):
        return (
            tmp1.reset_index()
            .melt(id_vars="att", value_vars=tmp1.columns, var_name="a", value_name=name)
            .set_index(["att", "a"])
        )
    else:
        return (
            tmp1.rename(name)
            .reset_index()
            .rename(columns={"level_1": "a"})
            .set_index(["att", "a"])
        )


def print_num(what):
    tmp = what.sum()
    if isinstance(tmp, np.ndarray):
        return tmp[0]
    elif isinstance(tmp, np.integer):
        return tmp
    else:
        return tmp.values[0]


# %% [markdown]
# # paramatt

# %%
numSets = []
numSetsCP2D = []
numAuths = []
fract = []
correct = np.zeros(6)
correct2 = np.zeros(2)
for book in original:
    print(book)
    numSets.append(["Forms", print_num(original[book][0])])
    print(*numSets[-1], sep=": ")
    numSets.append(["Lemmas", print_num(lemmatised[book][0])])
    print(*numSets[-1], sep=": ")
    numSetsCP2D.append(["Forms", print_num(original[book][2])])
    numSetsCP2D.append(["Lemmas", print_num(lemmatised[book][2])])
    tmp = [
        get_tmp(tmp1, name)
        for tmp1, name in zip(original[book], ["ori_plain", "ori_ca", "ori_naive"])
    ] + [
        get_tmp(tmp1, name)
        for tmp1, name in zip(lemmatised[book], ["lem_plain", "lem_ca", "lem_naive"])
    ]
    counts = (
        pd.concat(tmp[:], axis=1).groupby(["att", "a"]).apply(lambda df: df.sum(axis=0))
    )

    for column in counts.columns:
        numAuths.append(
            [
                "Forms" if "ori" in column else "Lemmas",
                column.split("_")[1],
                np.sum(counts[column].groupby("a").sum().values > 0),
            ]
        )
        fract.append(
            [
                "Forms" if "ori" in column else "Lemmas",
                column.split("_")[1],
                counts.groupby("a").sum().loc[:, column].max() / counts[column].sum(),
            ]
        )
    correct += counts.groupby("a").sum().idxmax(0).values == book[0]
    for i, naive in enumerate(["ori_naive", "lem_naive"]):
        cand = counts[naive][counts[naive] != 0].index.get_level_values("a").unique()
        correct2[i] += len(cand == 1) and cand[0] == book[0]

numAuths_df = pd.DataFrame(numAuths, columns=["Tokens", "attr", "authors"])
fract_df = pd.DataFrame(fract, columns=["Tokens", "attr", "fraction"])

# %%
for attr in ["plain", "ca", "naive"]:
    print(attr)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    plt.sca(ax[0])
    sns.boxplot(
        pd.DataFrame(
            numSetsCP2D if attr == "naive" else numSets, columns=["Tokens", "Sets"]
        ),
        x="Tokens",
        y="Sets",
        ax=ax[0],
        hue="Tokens",
    )
    plt.legend([], [], frameon=False)
    plt.ylabel("#comnparable parameter sets")
    plt.text(0.9, 0.2, "A", transform=ax[0].transAxes, fontweight="bold", fontsize=14)

    plt.sca(ax[1])
    sns.histplot(
        numAuths_df[numAuths_df.attr == attr],
        x="authors",
        hue="Tokens",
        multiple="dodge",
        bins=np.arange(5) + 0.5,
        shrink=0.8,
        ax=ax[1],
    )
    plt.xticks(1 + np.arange(4))
    plt.yticks(np.arange(plt.ylim()[1] // 5 + 1) * 5)
    plt.xlabel("Number of proposed authors")
    plt.ylabel("Number of documents")
    plt.text(0.9, 0.2, "B", transform=ax[1].transAxes, fontweight="bold", fontsize=14)

    plt.sca(ax[2])
    sns.boxplot(
        fract_df[fract_df.attr == attr],
        x="Tokens",
        y="fraction",
        ax=ax[2],
        hue="Tokens",
    )
    plt.legend([], [], frameon=False)
    plt.ylabel("Fraction of attributions to the preferred author")
    plt.xlim(plt.xlim())
    plt.hlines(0.5, *plt.xlim(), "k", "dotted")
    plt.text(0.9, 0.2, "C", transform=ax[2].transAxes, fontweight="bold", fontsize=14)
    plt.savefig(
        f"../results/paramatt_stats_{attr}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "chr24_results.py"},
    )
    plt.show()

# %%
print(
    "Top quartile lemmas:",
    pd.DataFrame(numSetsCP2D, columns=["Tokens", "Sets"])[
        (np.arange(72) % 2).astype(bool)
    ].Sets.quantile(0.75),
)
print(
    "Top quartile forms :",
    pd.DataFrame(numSetsCP2D, columns=["Tokens", "Sets"])[
        ~(np.arange(72) % 2).astype(bool)
    ].Sets.quantile(0.75),
)

# %%
print(
    "Correct attributions:\nforms plain\t forms ca\t lemmas plain\t lemmas ca",
    correct[[0, 1, 3, 4]],
)
print("Correct attributions naive:\nforms \t lemmas ", correct2)

# %% [markdown]
# # maps

# %%
colors = {}
for A, color in zip(range(7), sns.palettes.color_palette("colorblind")):
    colors[A] = color
letter = np.array([["A", "B", "C", "D"], ["E", "F", "G", "H"]]).T

# %%
for tokens in ["lemmas", "forms"]:
    for i in range(5):
        print(tokens)
    fig, ax = plt.subplots(4, 2, figsize=(8, 12))
    for row, book in enumerate([(3, 14), (3, 15), (3, 18), (4, 4)]):
        print("Now showing: ", book)
        with open(os.path.join("../data", tokens, "bookNames.dat")) as fp:
            book_names = {}
            for line in fp:
                A, B, tit = line.split()
                A, B = int(A), int(B)
                if A == book[0] and B >= book[1]:
                    if B == book[1]:
                        book_names[(0, 1)] = tit
                    else:
                        book_names[(A, B - 1)] = tit
                else:
                    book_names[(A, B)] = tit
        with open(os.path.join("../data", tokens, "authorNames.dat")) as fp:
            author_names = {}
            for line in fp:
                A, name = line.split()
                author_names[int(A)] = name
        author_names[0] = book_names[(0, 1)]
        attrib = pd.read_csv(
            f"../results/joined_results/{tokens}_A{book[0]}B{book[1]}.csv",
            index_col=np.arange(0, 7),
        )
        selector = (
            attrib.loc[(attrib.reset_index().A == 0).values]
            .idxmax(axis=1)
            .rename("idxmax")
            .reset_index(["A", "B"])
            .drop(columns=["A", "B"])
        )
        scores = pd.read_csv(
            f"../results/joined_results/scores_{tokens}_A{book[0]}B{book[1]}.csv",
            index_col=np.arange(0, 5),
        )
        params = (
            pd.concat([scores, selector], axis=1)
            .loc[
                (slice(None), slice(None), slice(None), slice(None), "FNN"),
            ]  # (attrib.reset_index().att == "FNN").values
            .groupby("idxmax")["mR"]
            .idxmax()
        )
        if len(params.index) > 2:
            params = params.iloc[:2]
        elif len(params.index) < 2:
            params = pd.concat(
                [
                    params,
                    pd.concat([scores, selector], axis=1)
                    .loc[
                        (slice(None), slice(None), slice(None), slice(None), "FNN"),
                    ]  # (attrib.reset_index().att == "FNN").values
                    .groupby("idxmax")["mR"]
                    .idxmin(),
                ],
                axis=0,
            )

        for col, param in enumerate(params.values):
            plt.sca(ax[row, col])
            mR = scores.loc[param, "mR"]
            data = attrib.loc[param]
            columns = sorted(np.argsort(data.loc[(0, 1)].values)[-3:] + 1)
            palette = [colors[A] for A in [0] + columns]
            print(columns)
            shaved = data.loc[
                ([(a in columns) or a == 0 for a in data.index.get_level_values(0)],),
                map(str, columns),
            ].copy()
            tmp = shaved.values / np.sqrt(np.sum(shaved.values**2, 1))[:, np.newaxis]
            shaved.reset_index(inplace=True)
            shaved["phi1"] = np.arctan2(np.sqrt(np.sum(tmp[:, :2] ** 2, 1)), tmp[:, 2])
            shaved["phi2"] = np.arctan2(tmp[:, 1], tmp[:, 0])
            shaved["A"] = pd.Categorical(shaved["A"])
            shaved["Author"] = shaved["A"].apply(lambda x: author_names[x])

            hue_order = [author_names[A] for A in [0] + columns]
            sns.scatterplot(
                shaved,
                x="phi2",
                y="phi1",
                hue="Author",
                hue_order=hue_order,
                palette=palette,
                ax=ax[row, col],
            )
            for i, [A, B, phi1, phi2] in shaved.loc[
                slice(None), ["A", "B", "phi1", "phi2"]
            ].iterrows():
                ax[row, col].text(
                    phi2,
                    phi1,
                    book_names[(A, B)],
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    color="red" if (A, B) == (0, 1) else "black",
                )
            ax[row, col].set_xlim(ax[row, col].get_xlim())
            ax[row, col].set_ylim(ax[row, col].get_ylim())
            xyz = np.full([3, 1000], 3.0)
            xyz[1] = 3 * np.arange(1000) / 999
            order = np.arange(3)
            for i in range(3):
                neword = np.roll(order, i)
                phi1 = np.arctan2(
                    np.sqrt(xyz[neword[0], :] ** 2 + xyz[neword[1], :] ** 2),
                    xyz[neword[2], :],
                )
                phi2 = np.arctan2(xyz[neword[1], :], xyz[neword[0], :])
                ax[row, col].plot(phi2, phi1, ":k")

            ax[row, col].set_title(
                f"$P_0$: {'fixed' if param[0] else 'auth.dep.'}, {f'F: {param[1]}' if param[1] else 'f.d.'}, {f'N: {param[2]}' if param[2] else 'f.l.'}, $\\delta$: {param[3]:.3} (M.R.: {mR:.4})",
                pad=0,
            )
            if row == 3:
                ax[row, col].set_xlabel(r"$\phi_1$")
            else:
                ax[row, col].set_xlabel("")
            ax[row, col].set_ylabel(
                f"Document: {book_names[(0,1)]}\n" + r"$\phi_2$" if col == 0 else ""
            )
            yspan = np.diff(plt.ylim())
            yspace = 0.001
            while yspan // yspace > 6:
                yspace += 0.001

            nticks = int(yspan // yspace) + 1
            plt.yticks((plt.ylim()[0] // yspace + 1 + np.arange(nticks)) * yspace)
            xspan = np.diff(plt.xlim())
            xspace = 0.001
            while xspan // xspace > 5 and xspace < 0.01:
                xspace += 0.001

            nticks = int(xspan // xspace) + 1
            plt.xticks((plt.xlim()[0] // xspace + 1 + np.arange(nticks)) * xspace)
            x0, y0 = (
                ax[row, col]
                .transData.inverted()
                .transform_point(ax[row, col].transAxes.transform_point([0.1, 0.1]))
            )
            plt.plot(
                [x0, x0, x0 + 0.001],
                [y0 + 0.001, y0, y0],
                "-k",
                lw=2,
                solid_capstyle="butt",
                solid_joinstyle="bevel",
            )
            plt.legend([], [], frameon=False)
            plt.text(
                0.92,
                0.08,
                letter.T.reshape([4, 2])[row, col],
                transform=ax[row, col].transAxes,
                fontweight="bold",
                fontsize=14,
            )
    ha = [Line2D([], [], lw=0, ls="-", color=colors[A], marker="o") for A in colors]
    la = ["Anonymous"] + [author_names[A] for A in author_names]
    plt.sca(ax[2, 1])
    plt.legend(
        ha,
        la,
        title="Author",
        bbox_to_anchor=(1.1, 1.1),
        loc=6,
        borderaxespad=0.0,
        frameon=False,
    )
    # fig.tight_layout()
    plt.savefig(
        f"../results/maps_{'Lemmas' if 'le' in tokens else 'Forms'}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "chr24_results.py"},
    )
    plt.show()
