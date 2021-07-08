bash prepare-wmt14en2de.sh  bpe30000 30000 
python3 ../ot_run.py --source_file bpe30000/train.de --target_file bpe30000/train.en --token_candidate_file bpe30000/code --vocab_file wmtende.vocab \
       --max_number 30000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file ende.size ##source file, target file, language code, output vocabulary 
bash prepare-wmt14en2de-volt.sh OTdata wmtende.vocab 


