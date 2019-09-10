#!/usr/bin/env bash
git clone https://github.com/mattloper/opendr
cd opendr && python setup.py install
cd ../ && rm -fr opendr