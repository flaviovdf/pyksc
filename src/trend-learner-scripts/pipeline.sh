#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Please provide me with a time series file and an output folder"
    exit 1
fi

IN=$1
BASE_FOLD=$2
K=4

#Creates output folder
mkdir -p $BASE_FOLD 2> /dev/null

#Generate cross-val
python generate_cross_vals.py $IN $BASE_FOLD

#Cluster dataset
for fold in $BASE_FOLD/*/; do
    mkdir -p $fold/ksc 2> /dev/null
    python cluster.py $IN $fold/ksc $K
done

#Precompute probabilities train
for fold in $BASE_FOLD/*/; do
    mkdir -p $fold/probs/ 2> /dev/null
    python classify_pts.py $IN $fold/train.dat $fold/ksc/cents.dat \
        $fold/ksc/assign.dat $fold/probs/
done

#Precompute probabilities test
for fold in $BASE_FOLD/*/; do
    mkdir -p $fold/probs-test/ 2> /dev/null
    python classify_pts_test.py $IN $fold/ksc/cents.dat $fold/test.dat \
        $fold/ksc/assign.dat $fold/probs-test/
done

#Create the assign for the test
for fold in $BASE_FOLD/*/; do
    python create_test_assign.py $IN $fold/test.dat \
        $fold/ksc/cents.dat > $fold/ksc/test_assign.dat
done

#Learn parameters train
for fold in $BASE_FOLD/*/; do
    mkdir -p $fold/cls-res-fitted-50-train 2> /dev/null
done
python classify_theta_train.py $IN $BASE_FOLD

#Learn parameters test
for fold in $BASE_FOLD/*/; do
    mkdir -p $fold/cls-res-fitted-50 2> /dev/null
done
python classify_theta.py $IN $BASE_FOLD
