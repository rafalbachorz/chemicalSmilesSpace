#!/bin/bash
python trainSmiles.py -dataFile ../data/processed/molDataGroupedFinal.pckl -modelWeightsFile test.h5 -completeModel test.pckl -nSample 300000 -nBatch 196 -nEpoch 1
