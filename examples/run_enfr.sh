bash prepare-wmt14en2fr.sh  enfrbpe30000 30000 
python3 ../ot_run.py --source_file enfrbpe30000/train.fr --target_file enfrbpe30000/train.en --token_candidate_file enfrbpe30000/code --vocab_file wmtenfr.vocab \
       --max_number 30000 --interval 1000  --loop_in_ot 500 --tokenizer subword-nmt --size_file enfr.size ##source file, target file, language code, output vocabulary 
bash prepare-wmt14en2fr-volt.sh enfrOTdata wmtenfr.vocab 


