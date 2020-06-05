#!/bin/bash
if [ ! -z $ENGINEDIR ]
then
        echo "ENGINEDIR set to: "
else
        echo "Please export ENGINEDIR!"
        exit 1
fi
echo $ENGINEDIR

DATADIR=$ENGINEDIR/data
MODELDIR=$ENGINEDIR/model

MTTools=$( dirname $0 )
SUBWORDTools="subword-nmt"

NUMSYM=5000

# train BPE
$SUBWORDTools/learn_bpe.py --input $DATADIR/train.src-mt --output $DATADIR/bpe.src-mt --symbols $NUMSYM

# apply BPE
for FILE in 'train' 'test' 'dev'
do
    $SUBWORDTools/apply_bpe.py -c $DATADIR/bpe.src-mt < $DATADIR/$FILE.src > $DATADIR/${FILE}.bpe_src
    $SUBWORDTools/apply_bpe.py -c $DATADIR/bpe.src-mt < $DATADIR/$FILE.mt > $DATADIR/${FILE}.bpe_mt
done

cat $DATADIR/${FILE}.bpe_src $DATADIR/${FILE}.bpe_mt > $DATADIR/${FILE}.bpe_src-bpe_mt
python create_vocabulary_labels.py -d $DATADIR/${FILE}.bpe_src-bpe_mt > $DATADIR/data.dict


