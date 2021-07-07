#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
#git clone https://github.com/rsennrich/subword-nmt.git
SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=../subword-nmt
#BPE_TOKENS=$2 #40000


CORPORA=(
        'es'
        'pt-br'
        'fr'
        'ru'
        'he'
        'ar'
        'ko'
        'zh-cn'
        'it'
        'ja'
        'zh-tw'
        'nl'
        'ro'
        'tr'
        'de'
        'vi'
        'pl'
        'pt'
        'bg'
        'el'
        'fa'
        'sr'
        'hu'
        'hr'
        'uk'
        'cs'
        'id'
        'th'
        'sv'
        'sk'
        'sq'
        'lt'
        'da'
        'my'
        'sl'
        'mk'
        'fr-ca'
        'fi'
        'hy'
        'hi'
        'nb'
        'ka'
        'mn'
        'et'
        'ku'
        'gl'
        'mr'
        'zh'
        'ur'
        'eo'
        'ms'
        'az'
        'ta'
        'bn'
        'kk'
        'be'
        'eu'
        'bs'
)
OUTDIR=$1
tgt=en
prep=$OUTDIR
tmp=$prep/tmp

mkdir -p $prep $tmp
tmp1=ted

echo "pre-processing train data..."
for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    for l in $cp $tgt; do
        for f in train test; do 
            rm $tmp/bpe.$lang.$f.$l
            cat $tmp1/data/${cp}_en/$f.$l  >> $tmp/bpe.$lang.$f.$l
        done 
    done
done


for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    for l in $cp $tgt; do
        for f in dev; do        
            rm $tmp/bpe.$lang.valid.$l
            cat $tmp1/data/${cp}_en/$f.$l  >> $tmp/bpe.$lang.valid.$l
        done
    done
done

TRAIN=$tmp/train.all-en
TRAIN_EN=$tmp/train.allen
BPE_CODE=$prep/code
BPE_CODE1=$prep/code1
rm -f $TRAIN 
rm -f $TRAIN_EN

BPE_INITIAL=$2
for cp in "${CORPORA[@]}"; do
    for l in $tgt; do
        lang=$cp-en
        cat $tmp/bpe.$lang.train.$l >> $TRAIN_EN
    done
done

for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    shuf -r -n 100000 $tmp/bpe.$lang.train.$cp >> $TRAIN
done

python3 ../ot_run.py $TRAIN_EN $TRAIN $BPE_INITIAL/code $prep/multilingual.vocab 100000 10000 400

#shuf -r -n 100000 $TRAIN_EN >> $TRAIN


#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s multilingual.vocab  < $TRAIN > $BPE_CODE
echo "#version: 0.2" > $prep/multilingual.bpe
cat $prep/multilingual.vocab >> $prep/multilingual.bpe


mkdir -p $prep/processed_data
for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    for l in $cp $tgt; do
        for f in train valid test; do
            if [ $f == train ]
            then
               echo $cp
               #FN=$(wc -l < $tmp/bpe.$lang.$f.$l)
               #float=`expr $FN / 10`
               #head -${float%.*} $tmp/bpe.$lang.$f.$l >  $tmp/bpe.$lang.$f.$l.small
               python3 $BPEROOT/apply_bpe.py -c multilingual.vocab < $tmp/bpe.$lang.$f.$l > $prep/processed_data/$lang.$f.$l
            else
               #echo $cp
               python3 $BPEROOT/apply_bpe.py -c multilingual.vocab < $tmp/bpe.$lang.$f.$l > $prep/processed_data/$lang.$f.$l
            fi
        done
    done
done
            

