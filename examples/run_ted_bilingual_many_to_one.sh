# get token candidates
bash prepare-ted-bilingual-manytoone.sh tedbpe30000 30000 #$1: output file, $2: bpe-size (here we adopt a very large one)
bash prepare-ted-bilingual-manytoone-volt.sh tedbilingualmanytoonevolt tedbpe30000


