#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


cd ../POT
python3 -m pip install --force-reinstall --editable ./ -i https://pypi.doubanio.com/simple --user


cd ../examples




SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=../subword-nmt
BPE_INITIAL=$2 #40000


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
tmp1=$3

echo "pre-processing train data..."
for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    for l in $cp $tgt; do
        for f in train test; do
            rm $tmp/bpe.$lang.$f.$l
            cat $tmp1/${cp}_en/$f.$l >> $tmp/bpe.$lang.$f.$l
        done
    done
done


for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    for l in $cp $tgt; do
        for f in dev; do
            rm $tmp/bpe.$lang.valid.$l
            cat $tmp1/${cp}_en/$f.$l  >> $tmp/bpe.$lang.valid.$l
        done
    done
done

TRAIN=$tmp/train.all-en
TRAIN_EN=$tmp/train.allen
BPE_CODE=$prep/code
BPE_CODE1=$prep/code1
rm -f $TRAIN


for cp in "${CORPORA[@]}"; do
    lang=$cp-en

    #shuf -r -n 100000 $tmp/bpe.$lang.train.$cp >> $TRAIN
    cat $tmp/bpe.$lang.train.$cp > $TRAIN$lang
    cat $tmp/bpe.$lang.train.en >> $TRAIN$lang
    python3 ../ot_run.py --source_file $BPE_INITIAL/processed_data/$lang.train.$cp --target_file $BPE_INITIAL/processed_data/$lang.train.en \
	    --token_candidate_file $BPE_INITIAL/$lang.vocab \
	    --vocab_file $prep/en$cp.vocab --max_number 10000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file $prep/en$cp.size

    best_size=$(cat $prep/en$cp.size)
    python3  spm/spm_train.py --input=$TRAIN$lang --model_prefix=$prep/$lang --vocab_size=$best_size --character_coverage=1.0 --model_type=bpe
    #size=$(cat $prep/size.$cp)
    mkdir -p $prep/processed_data
    
    for l in $cp $tgt; do
            for f in train valid test; do
                if [ $f == train ]
                then
                    echo $cp
		    python3 spm/spm_encoder.py --model $prep/$lang.model --inputs $tmp/bpe.$lang.$f.$l --outputs $prep/processed_data/$lang.$f.$l --output_format piece
               else
                    python3 spm/spm_encoder.py --model $prep/$lang.model --inputs $tmp/bpe.$lang.$f.$l --outputs $prep/processed_data/$lang.$f.$l --output_format piece
	       fi  
            done
        
    done
done

