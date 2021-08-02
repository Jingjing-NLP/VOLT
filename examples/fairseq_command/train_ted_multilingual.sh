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
#GPU numbers: 16
local_dataset_path=$1 #the processed datasets
update_freq=$2 #2
max_epoch=$3 #20
max_tokens=$4 #2400

mkdir -p  ${local_dataset_path}/merge/
for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    cat  ${local_dataset_path}/processed_data/$lang.train.en  >> ${local_dataset_path}/merge/all.train.target
    cat  ${local_dataset_path}/processed_data/$lang.train.$cp  >> ${local_dataset_path}/merge/all.train.source
    cat  ${local_dataset_path}/processed_data/$lang.valid.$cp >> ${local_dataset_path}/merge/all.valid.source
    cat  ${local_dataset_path}/processed_data/$lang.valid.en  >> ${local_dataset_path}/merge/all.valid.target
    cat  ${local_dataset_path}/processed_data/$lang.test.$cp  >> ${local_dataset_path}/merge/all.test.source
    cat  ${local_dataset_path}/processed_data/$lang.test.en  >> ${local_dataset_path}/merge/all.test.target
done

python3 fairseq_cli/preprocess.py \
    --source-lang source --target-lang target \
    --trainpref ${local_dataset_path}/merge/all.train \
    --validpref ${local_dataset_path}/merge/all.valid \
    --testpref ${local_dataset_path}/merge/all.test \
    --destdir ${local_dataset_path}mergealldata \
    --joined-dictionary \ #    --nwordssrc 1000000 --nwordstgt 1000000 \
    --workers 20



for cp in "${CORPORA[@]}"; do
    lang=$cp-en
    python3 fairseq_cli/preprocess.py \
        --source-lang $cp --target-lang en \
        --trainpref ${local_dataset_path}/processed_data/$lang.train \
        --validpref ${local_dataset_path}/processed_data/$lang.valid \
        --testpref ${local_dataset_path}/processed_data/$lang.test \
        --tgtdict ${local_dataset_path}mergealldata/dict.target.txt \
        --destdir ${local_dataset_path}_idx \
        --joined-dictionary \
        --workers 20
done

lang_pair=""
for cp in "${CORPORA[@]}"; do
    lang_pair=$lang_pair,$cp-en
done
lang_pair=${lang_pair#?}


python3 fairseq_cli/train.py --num-workers 0 ${local_dataset_path}_idx \
    --task multilingual_translation --lang-pairs $lang_pair \
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
    --max-epoch $max_epoch --keep-last-epochs 10

python3 scripts/average_checkpoints.py \
    --inputs ./checkpoints/multilingual_transformer \
    --num-epoch-checkpoints 1  --checkpoint-upper-bound 4 \
    --output ./checkpoints/checkpoint.avg5.pt

mkdir -p ./checkpoints/output

#for some languagues requiring segmentation, you can calculate char-level BLEU. You need to write another script the process the output texts.  

for cp in "${CORPORA[@]}"; do
     echo $cp
     python3 fairseq_cli/generate.py ${local_dataset_path}$2_idx \
           --task multilingual_translation --lang-pairs $lang_pair \
           --source-lang ${cp} --target-lang en \
           --path checkpoints/checkpoint.avg5.pt \
            --batch-size 128 \
           --beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > ./checkpoints/output/gen$cp.out

