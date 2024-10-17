import glob, os

if not os.path.isdir("../data/lemmas"):
    os.mkdir("../data/lemmas")
if not os.path.isdir("../data/forms"):
    os.mkdir("../data/forms")
for folder in ["lemmas", "forms"]:
    with open(os.path.join("../data", folder, "encoding.dat"), "+a") as fo:
        print("ASCII", file=fo)

old_aut = ""
NA = 0
NB = 0
for file in sorted(glob.glob("../data/apn/*APN")):
    fname = os.path.basename(file)
    fname = os.path.splitext(fname)[0]
    author, book, short = fname.split("_")
    if author != old_aut:
        NA += 1
        NB = 1
        for folder in ["lemmas", "forms"]:
            with open(os.path.join("../data", folder, "authorNames.dat"), "+a") as fo:
                print(NA, author, file=fo)
        old_aut = author
    else:
        NB += 1
    for folder in ["lemmas", "forms"]:
        with open(os.path.join("../data", folder, "bookNames.dat"), "+a") as fo:
            print(NA, NB, short, file=fo)

    lemmas = []
    forms = []
    with open(file, "r") as fp:
        for line in fp:
            if not len(line.strip()):
                continue
            lemma = line[8:30].split()[0]
            form = line[30:55].strip()
            lemmas.append(lemma)
            forms.append(form)
    for folder, sequence in [("lemmas", lemmas), ("forms", forms)]:
        with open(
            os.path.join("../data", folder, f"A{NA}.txt"), "+a", encoding="ASCII"
        ) as fo:
            if author == "Catullus":
                NB = 1
                # Catullus is manually split in three pieces
                for n, part in enumerate(
                    [slice(None, 3294), slice(3294, 11530), slice(11530, None)]
                ):
                    print("#", NB + n, file=fo)
                    print(" ".join(sequence[part]), file=fo)
                with open(
                    os.path.join("../data", folder, "bookNames.dat"), "+a"
                ) as fbn:
                    for i in range(2):
                        NB += 1
                        print(NA, NB, short + str(NB), file=fbn)
            else:
                print("#", NB, file=fo)
                print(" ".join(sequence), file=fo)
