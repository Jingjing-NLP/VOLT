import os
import math
import sys
import pandas as pd
from collections import OrderedDict

def read_file(path):
    tokens = {}
    total_number = 0
    lengths = []
    #de_lines
    with open(path, "r") as sr:
        lines = sr.readlines()#[:10000]
    return lines

def read(lines):
    tokens = {}
    total_number = 0
    lengths = []
    for line in lines:
            line_tokens = line.strip().split()#.replace("@@ ", "").replace(" ", "")
            total_number += len(line_tokens)
            for token in line_tokens:
                if token.strip() == "":
                   continue
                if token not in tokens:
                   tokens[token] = 0
                tokens[token] += 1
                lengths.append(len(token))

    entropy_sum, entropy_sum1 = 0, 0
    for token in tokens:
        prob = tokens[token]*1.0/total_number
        entropy_sum += -prob * math.log(prob)
    entropy_sum1 = entropy_sum / (sum(lengths)/len(lengths))
    return entropy_sum1,len(tokens)




dicts = {}
previous_entropy = {}
previous_size = 0
size_list = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000]
#lines = read_file("wmt_en_debpe"+str)
for size in size_list:
  files = ["train.de", "train.en"]
  dicts_entropy = OrderedDict()
  lines_all = []
  for file_name in files:
    lines = read_file("bpe"+str(size)+"/"+file_name)
    lines_all += lines

  entropy,size_token = read(lines_all)

  if "muv" in dicts:
       dicts["muv"].append(-(entropy-previous_entropy["muv"][-1])/(size_token-previous_size))
       previous_entropy["muv"].append(entropy)
       print(size, dicts["muv"][-1])
  else:
       if size != size_list[0]: continue
       dicts["muv"] = []
       previous_entropy["muv"] = [entropy]

  previous_size = size_token

print(dicts)

