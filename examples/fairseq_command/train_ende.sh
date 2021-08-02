#!/bin/bash


output_path=${local_root}/output
mkdir $output_path

#number gpu = 4
data=$1
max_token=9600
max_epoch=100
update_freq=8
lr=0.0005
total_word=100000

python3 fairseq_cli/preprocess.py \
    --source-lang en --target-lang de \
    --trainpref $data/train \
    --validpref $data/valid \
    --testpreef $data/test \
    --destdir en_de \
    --nwordssrc $total_word --nwordstgt $total_word \
    --joined-dictionary \
    --workers 20


python3 fairseq_cli/train.py --num-workers 8 en_de \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 --update-freq $update_freq \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $max_token  \
    --fp16  --max-epoch $max_epoch --keep-last-epochs 5

python3 scripts/average_checkpoints.py \
    --inputs ./checkpoints \
    --num-epoch-checkpoints 5 \
    --output checkpoint.avg5.pt

python3 fairseq_cli/generate.py en_de \
    --path checkpoint.avg5.pt \
    --beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > gen1.out

bash scripts/compound_split_bleu.sh gen1.out



