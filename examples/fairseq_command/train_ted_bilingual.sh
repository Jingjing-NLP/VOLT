#!/bin/bash



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

#the number of gpus: 4
local_dataset_path=$1 # the generated dataset
update_freq=$2 #2
max_epoch=$3  #40
max_tokens=$4 #2400

mkdir -p ${local_dataset_path}/idx

for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    python3 fairseq_cli/preprocess.py \
        --source-lang $cp --target-lang en \
        --trainpref ${local_dataset_path}/processed_data/$lang.train \
        --validpref ${local_dataset_path}/processed_data/$lang.valid \
        --testpref ${local_dataset_path}/processed_data/$lang.test \
        --destdir ${local_dataset_path}/idx/$cp-idx \
        --nwordssrc 1000000 --nwordstgt 1000000 \
        --joined-dictionary \
        --workers 20
done

for cp in "${CORPORA[@]}"; do

mkdir checkpoints
mkdir checkpoints/multilingual_transformer
python3  fairseq_cli/train.py --num-workers 0 ${local_dataset_path}/idx/$cp-idx \
    --single-node --task multilingual_translation --lang-pairs $cp-en \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens $max_tokens \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --update-freq $update_freq  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --fp16  --max-epoch $max_epoch --keep-last-epochs 5

python3 scripts/average_checkpoints.py \
    --inputs ./checkpoints/multilingual_transformer \
    --num-epoch-checkpoints 5 \
    --output ./checkpoints/checkpoint.avg5.pt

mkdir -p ./checkpoints/output
echo $cp
python3 fairseq_cli/generate.py ${local_dataset_path}/idx/$cp-idx \
            --task multilingual_translation --lang-pairs $cp-en \
           --source-lang ${cp} --target-lang en \
           --path checkpoints/checkpoint.avg5.pt \
            --batch-size 128 \
           --beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > ./checkpoints/output/gen$cp.out
bash scripts/compound_split_bleu.sh ./checkpoints/output/gen$cp.out

