#!/bin/bash
python trainSmiles.py -dataFile ../data/processed/molDataGroupedFinal.pckl -modelWeightsFile model_20191126.h5 -completeModel predictiveModel_20191126.pckl -nSample 300000 -nBatch 196 -nEpoch 256
