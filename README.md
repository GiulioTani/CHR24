# CHR24

This repository contains the code underlying submission 121 to CHR24 conference: **"Promises from an Inferential Approach in Classical Latin Authorship Attribution"**

After downloading the APN files from Hyperbase into `data/apn`, running `scripts/launch_chr24.sh` and waiting long enough will produce the results in the paper.

Since most of the time is spent in computing the likelihoods using 3-grams, it is possible to reproduce most of the results in much shorter time (under half an hour in total with 8 x 2.4 GHz CPU, 16 GiB RAM) deleting the element "3" from the list at line 40 in `scripts/chr24_compute.py`
