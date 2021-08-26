import math
import os
import numpy as np
from collections import OrderedDict
import ot
import ot.plot
import sys
from tqdm import tqdm
import get_tokens 
import get_chars
import argparse



def build_d_matrix(chars, tokens):
    """
    Initialize distance matrix in optimal transport. if the i-th char in j-th token, their distance is set to be a very small value, otherwize a large value.
    Arguments:
        chars (dict): charaters and their frequencies.
        tokens (dict): tokens and their frequencies.
    Returns:
        matrix: a 2-dimension distance matrix. 
    """
    matrix = np.zeros((len(chars), len(tokens)))
    rows = len(chars)
    cols = len(tokens)
    for i in range(rows):
        for j in range(cols):
            if chars[i][0] in tokens[j][0]:
               matrix[i][j] =  0.001*j #/len(tokens[j][0])#0.00001*tokens[j][1]#-math.log(tokens[j][1]*1.0/total_tokens)*1.0/len(tokens[j][0]) + 0.1/
            else:
               matrix[i][j] = 100#0
    return matrix




def get_r(items, total_number, chars=True):
    
    """
    Initialize character distribution and token distribution. For charater distributions, we directly adopt charaters associated with frequencies. For token distributions, we set the number of characters they require as the multiplication between their frequency and their lengths. For example, give a token 'cat' with frequency 500, it requires 500 'c', 500 'a', and 500 't'. Therefore, it requires 1,500 characteres in total.  
    
    Arguments:
       tokens (dict): characters associated with their frequencies / tokens with their frequencies.
       total_words (int): the number of all characters (or tokens). 
       chars: flags to distinguish character distribution and token distribution.

    Returns:
       a (list): character distribution or token distribution.
    """

    a = []

    for token in items:
        if chars == False:
           mul = len(token[0])
        else:
           mul = 1
        a.append(token[1]/total_number*mul + 1e-4)
    return a

def get_total_tokens(tokens):
    sum1 = 0
    for token in tokens:
        sum1 += token[1]
    return sum1



def write_vocab(tokens, pmatrix, chars, write_file_name, threshold=0.0001):
  """
  Generated the vocabulary via optimal matrix.
  
  Arguments:
     
     tokens: candidate distribution.
     pmatrix: the generated optimal transportation matrix. 
     chars: character distribution.
     write_file_name: the file storing the vocabulary.
     threshold: the filter ratio from the optimal transportation matrix to the real vocabulary. Here we set a small value. higher threshold menas that more tokens are removed from the token candidates. 
  """

  #itotal_tokens = get_total_tokens_dict(tokens)
  tokens = list(tokens.items())
  chars  = list(chars.items())
  total_tokens = len(tokens)
  new_tokens = {}
  for j in tqdm(range(len(pmatrix[0]))):
      new_tokens[tokens[j][0]] = {}
      for i in range(len(pmatrix)):
          if pmatrix[i][j] != 0 and pmatrix[i][j] * total_tokens  > threshold * tokens[j][1]:
              new_tokens[tokens[j][0]][chars[i][0]] = pmatrix[i][j] * total_tokens# * len(tokens[j][0])


  vocab_tokens = []
  for token in new_tokens:
      minm = 0
      itemlist = []
      if len(new_tokens[token]) == 0:
          #print(token, new_tokens[token])
          continue
      if  token.strip() != "" :
         vocab_tokens.append(token)


  with open(write_file_name, 'w') as f:
        for item in vocab_tokens:
            f.write(item +"\n")
  print("The vocabulary has been written into the given file. You can use this vocabulary to segment tokens in subword-nmt")


def run_ot(oldtokens, chars, max_number=30000, interval=1000, numItermax=300):
    scores = {}
    #max_number = 10000
    
    previous_entropy = 0
    chars = list(chars.items())
    
    for iter_number in range(interval, max_number, interval):#iteration_numbers:
        tokens = list(oldtokens.items())[:iter_number]
        #chars = list(chars.items)
      
        total_tokens = get_total_tokens(tokens)
        total_chars = get_total_tokens(chars)
        #average_len = get_average_len(tokens)
        l = [len(item[0])+1 for item in tokens]
        d_matrix = build_d_matrix(chars, tokens)
        a = get_r(chars, total_chars)
        #print(min(a), min(b))
        b = get_r(tokens, total_tokens, False)
        print(min(a), min(b))
        print("finish building")
        epsilon = 0.1  # entropy parameter
        alpha = 1.  # Unbalanced KL relaxation parameter
        current_entropy,_ = ot.sinkhorn(a,b,d_matrix,1.0,method='sinkhorn',numItermax=numItermax, epsilon0=1e-6)
        if iter_number <= interval:
            previous_entropy = current_entropy
            continue#print("finish reading", iter_number, Gs, (Gs-previous_entropy)/2)
        if iter_number > interval:
           print("finish running", iter_number, current_entropy, Gs-previous_entropy)
           scores[iter_number] = current_entropy-previous_entropy
        previous_entropy = current_entropy
    sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    print("best size: ", str(sorted_scores[0][0]))
    print("One optional solution is that you can use this size to generated vocabulary in subword-nmt or sentencepiece")
    return sorted_scores[0][0]
     

def run_ot_write(oldtokens, chars, optimal_size, numItermax=300): 
    previous_entropy = 0
    scores = {}
    tokens = list(oldtokens.items())[:optimal_size]
    total_tokens = get_total_tokens(tokens)
    total_chars = get_total_tokens(chars.items())
    #average_len = get_average_len(tokens)
    l = [len(item[0])+1 for item in tokens]
    d_matrix = build_d_matrix(list(chars.items()), tokens)
    a = get_r(chars.items(), total_chars)
    b = get_r(tokens, total_tokens, False)
    print("finish building")
    epsilon = 0.1  # entropy parameter
    alpha = 1.  # Unbalanced KL relaxation parameter
    _,Gs = ot.sinkhorn(a,b,d_matrix,1.0,method='sinkhorn',numItermax=numItermax, epsilon0=1e-6)
    previous_entropy = Gs
    return Gs




if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Process some input flags.')
    parser.add_argument('--source_file', default=None,
                        help='path to a source file for translation')
    parser.add_argument('--target_file', default=None,
                        help='path to a target file for translation')
    parser.add_argument('--token_candidate_file', default=None,
                        help='path to token candidates. In this implementation, we take BPE-generated code segmentation as token candidates.')
    parser.add_argument('--vocab_file', default=None,
                       help='path to the file storing the generated tokens.')
    parser.add_argument('--max_number', default=10000, type=int,
                       help='the maximum size of the generated vocabulary')
    parser.add_argument('--interval', default=1000, type=int,
                       help='the inverval size of S where S defines a sequence of size intergers. Please see papers for details. ')
    parser.add_argument('--loop_in_ot', default=500, type=int,
                       help = 'the total loop of optimal transation.')
    parser.add_argument('--threshold', default=0.00001, type=float,
                       help = 'the threshhold for generating the vocabulary based on the optimal matrix')
    parser.add_argument('--tokenizer', default='subword-nmt',choices=['sentencepiece', 'subword-nmt'])
    parser.add_argument('--size_file', default="size.txt", 
                       help='the size file stores the best size')
    
    args = parser.parse_args()
    source_file = args.source_file
    target_file = args.target_file
    token_candidate_file = args.token_candidate_file
    vocab_file = args.vocab_file
    max_number = args.max_number
    interval = args.interval
    num_iter_max = args.loop_in_ot
    threshold = args.threshold
    tokenizer = args.tokenizer
    size_file = args.size_file

    
    oldtokens = get_tokens.get_tokens(source_file, target_file, token_candidate_file, tokenizer=args.tokenizer) # get token candidates and their frequencies
    chars = get_chars.get_chars(source_file, target_file, tokenizer=args.tokenizer) # get chars and their frequencies
    optimal_size = run_ot(oldtokens, chars, max_number,interval, num_iter_max) # generate the best ot size
    Gs = run_ot_write(oldtokens, chars, optimal_size, num_iter_max) # generate the optimal matrix based on the ot size
    write_vocab(oldtokens, Gs, chars, vocab_file, threshold) #generate the vocabulary based on the optimal matrix
    with open(size_file, 'w') as sw:
         sw.write(str(optimal_size)+"\n")
    #return optimal_size
