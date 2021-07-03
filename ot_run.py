import math
import os
import numpy as np
from collections import OrderedDict
import ot
import ot.plot
import sys
from tqdm import tqdm
import get_tokens#.get_tokens 
import get_chars#.get_chars

def read(path, max_read_line=1000):
    words = OrderedDict()
    with open(path, "r") as sr:
        lines = sr.readlines()
        for line in lines:
            item = line.strip().split(" ")
            if len(item) == 3:
               words[" ".join([item[0], item[1]])] = int(item[2])
            else:
                words[item[0]] = int(item[1])
    return words


def build_d_matrix(chars, tokens, total_tokens):
    matrix = np.zeros((len(chars), len(tokens)))
    rows = len(chars)
    cols = len(tokens)
    for i in range(rows):
        for j in range(cols):
            if chars[i][0] in tokens[j][0]:
               matrix[i][j] =  0.0001*j #/len(tokens[j][0])#0.00001*tokens[j][1]#-math.log(tokens[j][1]*1.0/total_tokens)*1.0/len(tokens[j][0]) + 0.1/
            else:
               matrix[i][j] = 10#0
    return matrix


def get_total_tokens(tokens):
    sum1 = 0
    for token in tokens:
        #print(token)
        sum1 += token[1]
    return sum1

def get_average_len(tokens):
    avg = []
    for token in tokens:
        #print(token)
        avg.append(len(token[0]))
    return sum(avg)*0.1/len(avg)

def get_r(tokens, total_words,lens=1):
    a = []

    for token in tokens:
        if lens != 1:

           mul = len(token[0])
        else:
           mul = 1
        a.append(token[1]/total_words*mul)
    return a

def get_total_tokens(tokens):
    sum1 = 0
    for token in tokens:
        sum1 += token[1]
    return sum1

def get_total_tokens_dict(tokens):
    sum1 = 0
    for token in tokens:
        sum1 += tokens[token]
    return sum1
def write_vocab(tokens, pmatrix, chars,write_file_name):
  total_tokens = get_total_tokens_dict(tokens)
  tokens = list(tokens.items())
  chars  = list(chars.items())
  new_tokens = {}
  for j in tqdm(range(len(pmatrix[0]))):
      new_tokens[tokens[j][0]] = {}
      for i in range(len(pmatrix)):
          if pmatrix[i][j] != 0 and pmatrix[i][j] * total_tokens  > 0.0001 * 0.1 * tokens[j][1]:
              new_tokens[tokens[j][0]][chars[i][0]] = pmatrix[i][j] * total_tokens# * len(tokens[j][0])


  new_new_tokens = []
  for token in new_tokens:
      minm = 0
      itemlist = []
      if len(new_tokens[token]) == 0:
          print(token, origin_tokens[token])
          continue
      for item in new_tokens[token]:
          itemlist.append(new_tokens[token][item])
          minm += itemlist[-1]
      minm /= len(new_tokens[token])
      if  token.strip() != "" :
         new_new_tokens.append((token, minm))


  with open(write_file_name, 'w') as f:
        for item in new_new_tokens:
            f.write(item[0] +"\n")


def run_ot(oldtokens, chars, max_number=30000, interval=1000):
    scores = {}
    #max_number = 10000
    
    #iteration_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 100000, 20000, 30000]
    previous_entropy = 0
    for iter_number in range(1000, max_number, interval):#iteration_numbers:
        tokens = list(oldtokens.items())[:iter_number]
      
        total_tokens = get_total_tokens(tokens)
        total_chars = get_total_tokens(chars.items())
        average_len = get_average_len(tokens)
        l = [len(item[0])+1 for item in tokens]
        d_matrix = build_d_matrix(list(chars.items()), tokens, total_tokens)
        a = get_r(chars.items(), total_chars)
        b = get_r(tokens, total_tokens, 0)
        print("finish building")
        epsilon = 0.1  # entropy parameter
        alpha = 1.  # Unbalanced KL relaxation parameter
        Gs,_ = ot.sinkhorn(a,b,d_matrix,1.0,method='sinkhorn',numItermax=400)
        if iter_number <= 1000:
            previous_entropy = Gs
            continue#print("finish reading", iter_number, Gs, (Gs-previous_entropy)/2)
        if iter_number != 1000:
           print("finish reading", iter_number, Gs, Gs-previous_entropy)
           scores[iter_number] = Gs-previous_entropy
        previous_entropy = Gs
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    print("best size: ", str(sorted_scores[0][0]))
    return sorted_scores[0][0]
     

def run_ot_write(oldtokens, chars, optimal_size): 
    previous_entropy = 0
    scores = {}
    tokens = list(oldtokens.items())[:optimal_size]
    total_tokens = get_total_tokens(tokens)
    total_chars = get_total_tokens(chars.items())
    average_len = get_average_len(tokens)
    l = [len(item[0])+1 for item in tokens]
    d_matrix = build_d_matrix(list(chars.items()), tokens, total_tokens)
    a = get_r(chars.items(), total_chars)
    b = get_r(tokens, total_tokens, 0)
    print("finish building")
    epsilon = 0.1  # entropy parameter
    alpha = 1.  # Unbalanced KL relaxation parameter
    _,Gs = ot.sinkhorn(a,b,d_matrix,1.0,method='sinkhorn',numItermax=319)
    previous_entropy = Gs
    return Gs




if __name__ == "__main__":
    source_file = str(sys.argv[1])
    target_file = str(sys.argv[2])
    token_candidate_file = str(sys.argv[3])
    vocab_file = str(sys.argv[4])
    max_number = str(sys.argv[5])
    interval = str(sys.argv[6])

    oldtokens = get_tokens.get_tokens(source_file, target_file, token_candidate_file, "tokens")
    chars = get_chars.get_chars(source_file, target_file)
    optimal_size = run_ot(oldtokens, chars, int(max_number),int(interval))
    Gs = run_ot_write(oldtokens, chars, optimal_size)
    write_vocab(oldtokens, Gs, chars, vocab_file)

