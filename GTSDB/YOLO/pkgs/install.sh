#!/bin/bash

conda install --file $(dirname $0)/conda_packages.txt && \
pip install -r $(dirname $0)/pip_packages.txt
