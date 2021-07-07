bash prepare-wmt14en2de.sh  bpe30000 30000 #$1: output file, $2: bpe-size (here we adopt a very large one)
python3 ../ot_run.py bpe30000/train.de bpe30000/train.en bpe30000/code wmtende.vocab 30000 1000 500 ##source file, target file, language code, output vocabulary 
bash prepare-wmt14en2de-volt.sh OTdata wmtende.vocab 


