#!/bin/bash

if [ -z $4 ]
then
    echo 'Enter data folder'
    echo 'Enter attention type'
    echo 'Enter target language'
    echo 'Enter GPU id'
    exit 1
fi

F=$1
A=$2
L=$3
G=$4

echo $F
echo $A
echo $L
echo $G

D=/media/dimitarsh1/barracuda4tb/dimitarsh1/Projects/WMT2019/QETask/EN-${L}/${F}/model-bert-${F}-${A}
rm ${D}/*
mkdir ${D}
touch ${D}/init.time
python train.py -a ${A} -d ../EN-DE/${F}/data -m ${D} -g ${G}; touch ${D}/end.time
touch ${D}/end.time
