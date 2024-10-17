# %%
import sys, os, shutil, tempfile, glob

sys.path.append("../InnovationProcessesInference")
import cp2d
from scipy.stats import beta
import pandas as pd
import numpy as np

BASEPATH = "../data"
CHUNKS = int(sys.argv[3])
CHUNK = int(sys.argv[1])
P0 = int(sys.argv[2])

DELTAS = [
    0.01,
    0.016,
    0.025,
    0.04,
    0.063,
    0.1,
    0.16,
    0.25,
    0.4,
    0.63,
    1,
    1.6,
    2.5,
    4.0,
    6.3,
    10,
    16,
    25,
    40,
    63,
    100,
]
# P0s = [0, 1]
Fs = [0, 50, 100, 150, 300]
Ns = [0, 3, 4, 5, 6]


# %%
def create_temp(path, task):
    global P0
    there = glob.glob(f"/tmp/{os.path.basename(path)}_*_A{task[0]}B{task[1]}")
    if len(there):
        for fold in there:
            if os.path.isdir(
                "../res/DICT-WOR-L1O-" + os.path.basename(fold) + f"_P{P0}"
            ):
                return fold

    temp_folder = tempfile.mkdtemp(
        prefix=os.path.basename(path) + "_", suffix=f"_A{task[0]}B{task[1]}"
    )
    try:
        for file in os.listdir(path):
            if file.endswith("txt"):
                A = int(file[1:].split(".")[0])
                if A == task[0]:
                    BN = 1
                    unkFlag = False
                    with open(os.path.join(path, file)) as fin, open(
                        os.path.join(temp_folder, file), "w"
                    ) as fout:
                        for line in fin:
                            if line.startswith("#"):
                                if int(line.split()[1]) == task[1]:
                                    unkFlag = True
                            else:
                                if unkFlag:
                                    with open(
                                        os.path.join(temp_folder, "A0.txt"), "w"
                                    ) as funk:
                                        print("# 1", file=funk)
                                        print(line.strip(), file=funk)
                                    unkFlag = False
                                else:
                                    print(f"# {BN}", file=fout)
                                    BN += 1
                                    print(line.strip(), file=fout)
                elif A != 0:
                    shutil.copy(os.path.join(path, file), temp_folder)
            else:
                if file == "bookNames.dat":
                    BN = 1
                    with open(os.path.join(path, file)) as fin, open(
                        os.path.join(temp_folder, file), "w"
                    ) as fout:
                        for line in fin:
                            A, B, title = line.split()
                            if int(A) == task[0]:
                                if int(B) == task[1]:
                                    continue
                                print(" ".join(map(str, [A, BN, title])), file=fout)
                                BN += 1
                            else:
                                print(line.strip(), file=fout)
                else:
                    shutil.copy(os.path.join(path, file), temp_folder)
    except:
        shutil.rmtree(temp_folder)
        raise

    return temp_folder


def clear_temp(temp_folder):
    shutil.rmtree(temp_folder)
    path = f"../res/*{os.path.basename(temp_folder)}_P*"
    for folder in glob.glob(path):
        shutil.rmtree(folder)


# %%
def ci(f, n):
    k = np.round(n * f).astype(int)
    alpha = 0.05
    p_u, p_o = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])
    return p_u, p_o


def get_attributions(temp_folder, variant="mR"):
    global P0
    results = {}
    attributions = {}
    for F in Fs:
        for N in Ns:
            try:
                exp = cp2d.elaboration.cp2dExperiment(
                    LZ77=False,
                    fragment=F,
                    window=-1,
                    suffix=os.path.basename(temp_folder) + f"_P{P0}",
                    configFile="config.ini",
                    database=temp_folder,
                    ngramSize=N,
                    mantainSlicing=True,
                    retrieve=True,
                    keepTemporary=True,
                    overwriteOutputs=False,
                    leaveNout=1,
                    authorFiles=True,
                )
                exp.run()
                results[(P0, F, N)] = exp.results(
                    delta=DELTAS, machine=True, PANStyle="strict", topres=None
                )
            except:
                print("FAILED!!! ", temp_folder, F, N)
                raise
            attributions[(P0, F, N)] = exp.attributions
            del exp

    tret = []
    for F in Fs:
        for N in Ns:
            for delta in results[(P0, F, N)]:
                tret.append(
                    [
                        P0,
                        F,
                        N,
                        delta,
                        "FNN",
                        results[(P0, F, N)][delta]["all"]["FNN"]["weigh"]["R"],
                        results[(P0, F, N)][delta]["all"]["FNN"]["macro"]["R"],
                    ]
                )
                tret.append(
                    [
                        P0,
                        F,
                        N,
                        delta,
                        "MR",
                        results[(P0, F, N)][delta]["all"]["MR"]["weigh"]["R"],
                        results[(P0, F, N)][delta]["all"]["MR"]["macro"]["R"],
                    ]
                )
    res_df = (
        pd.DataFrame(tret, columns=["P0", "F", "N", "delta", "att", "wR", "mR"])
        .set_index(["P0", "F", "N", "delta", "att"])
        .sort_values(variant)
    )

    tot_books = len(attributions[(P0, 0, 0)]) - 1
    min_int, _ = ci(res_df[variant].max(), tot_books)
    int_df = res_df[res_df[variant] >= min_int]
    parts = os.path.basename(temp_folder).split("_")
    int_df.to_csv(
        os.path.join(
            "../results/individual_results",
            "_".join(["scores", parts[0], parts[-1] + f"_P{P0}.csv"]),
        )
    )

    tb = list(attributions[(P0, F, N)].keys())[0]
    td = list(attributions[(P0, F, N)][tb].keys())[0]
    nAuths = len(attributions[(P0, F, N)][tb][td]["FNN"])
    selected = []
    for P0, F, N, delta, att in int_df.index:
        for book in attributions[(P0, F, N)]:
            if att == "FNN":
                selected.append(
                    [
                        P0,
                        F,
                        N,
                        delta,
                        att,
                        *book,
                        *[
                            -1 / a[1]
                            for a in sorted(
                                attributions[(P0, F, N)][book][delta]["FNN"],
                                key=lambda x: x[0][0],
                            )
                        ],
                    ]
                )
            else:
                mratt = {i: 0 for i in range(1, nAuths + 1)}
                for fragA in attributions[(P0, F, N)][book][delta]["MR"]:
                    mratt[fragA[0][0]] = fragA[1]
                selected.append(
                    [
                        P0,
                        F,
                        N,
                        delta,
                        att,
                        *book,
                        *[mratt[A] for A in sorted(mratt.keys())],
                    ]
                )
    attributions_df = pd.DataFrame(
        selected,
        columns=[
            "P0",
            "F",
            "N",
            "delta",
            "att",
            "A",
            "B",
            *[i for i in range(1, nAuths + 1)],
        ],
    ).set_index(["P0", "F", "N", "delta", "att", "A", "B"])

    return attributions_df


# %%
def run_corpus(path):
    books = {}
    with open(os.path.join(BASEPATH, path, "bookNames.dat")) as fp:
        for line in fp:
            A, B, __ = line.split()
            try:
                books[int(A)].append(int(B))
            except:
                books[int(A)] = [int(B)]
    tasks = []
    for A in sorted(books):
        for B in sorted(books[A]):
            tasks.append((A, B))
    for i, task in enumerate(tasks):
        if i % CHUNKS != CHUNK or os.path.isfile(
            f"../results/individual_results/{path}_A{task[0]}B{task[1]}_P{P0}.csv"
        ):
            continue
        temp_folder = create_temp(os.path.join(BASEPATH, path), task)
        attributions = get_attributions(temp_folder)
        attributions.to_csv(
            f"../results/individual_results/{path}_A{task[0]}B{task[1]}_P{P0}.csv"
        )
        clear_temp(temp_folder)


# %%
for corpus in ["lemmas", "forms"]:
    run_corpus(corpus)
