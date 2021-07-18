bash prepare-wmt14en2de-withoutjoint.sh  bpe30000 30000 
python3 ../ot_run.py --source_file bpe30000/train.de  --token_candidate_file bpe30000/codede --vocab_file wmtde.vocab \
       --max_number 30000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file de.size ##source file, target file, language code, output vocabulary 
python3 ../ot_run.py --source_file bpe30000/train.en  --token_candidate_file bpe30000/codeen --vocab_file wmten.vocab \
       --max_number 30000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file en.size ##source file, target file, language code, output vocabulary 

bash prepare-wmt14en2de-volt-withoutjoint.sh OTdatawithoutjoint wmten.vocab wmtde.vocab 


