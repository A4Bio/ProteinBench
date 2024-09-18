#!/usr/bin/env bash

# run this script on the root
mkdir data/cath
cd data/cath

# down cath and place them in `data/cath/`
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_L100.json
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_sc.json


# download and arrange them in the following structure:
# OpenCPD
# └── data
#     ├── cath
#     │   ├── chain_set_splits.json
#     │   ├── chain_set.jsonl
#     │   ├── test_split_L100.json
#     │   ├── test_split_sc.json
#     │   ├── ts50remove.txt
