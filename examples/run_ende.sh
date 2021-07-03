# get token candidates
#bash prepare-wmt14en2de.sh  bpe30000 30000 #$1: output file, $2: bpe-size (here we adopt a very large one)
python3 ../ot_run.py bpe30000/train.de bpe30000/tran.en bpe30000/code ende.vocab 10000 2000 ##source file, target file, language code, output vocabulary 
bash prepare-wmt14en2de-volt.sh OT90000 ende.vocab 


