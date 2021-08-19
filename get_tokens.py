import os
import sys
from tqdm import tqdm
from collections import OrderedDict


def read_tokens(path, tokens={}, max_number_line=1e7, tokenizer='subword-nmt'): #@@
    """
    Get all tokens and their frequencies. 
    Arguments:
       path (str): the path of the specific file
       tokens (dict): the target dict storing tokens and their frequency
       max_number_line (int): we set the maximum number of lines to 1e7
       special_tokens (str): the tokens added by tokenization approaches
    Returns:
       tokens: return all tokens and their frequencies
    """
    with open(path, 'r') as sr:
         lines = sr.readlines()
         total_lines = min(len(lines), max_number_line)
         total_lines = int(total_lines)
         for i in range(total_lines):
             line = lines[i]
             items = line.split()
             for item in items:
                 if tokenizer == 'subword-nmt':
                     special_tokens = "@@"
                     if not item.endswith(special_tokens):
                            if item+"</w>" not in tokens:
                                 tokens[item+"</w>"] = 1
                            else:
                                 tokens[item+"</w>"] += 1
                     else:
                          if item not in tokens:
                               tokens[item] = 1
                          else:
                               tokens[item] += 1
                 elif tokenizer == 'sentencepiece':
                    if item not in tokens:
                        tokens[item] =1 
                    else:
                        tokens[item] += 1
                 else:
                    print("Errors: we only support subword-nmt and sentencepiece!")
    return tokens


def read_merge_code_frequency(path, tokens, min_number=10, tokenizer='subword-nmt'):
    """
    Get all code segmentations and their frequencies. Here we take BPE-generated code segmentations as token candidates. We usually sample a large BPE size, e.g., 3,0000. 
    Arguments:
       path (str): the path to the generated code. We take the segment merged by the generated codes as candidates.
       tokens (str): the dict storing tokens and their frequency. It is used to count the code frequency. 
       min_number (int): the minimum number of code frequency.
    Returns:
       merge_dict (dict): the code candidates and their frequency.
    """
    with open(path, 'r') as sr:
         lines = sr.readlines()
         merge_dict = OrderedDict()
         # for bpe codes, the first line shows code version
         if tokenizer == 'subword-nmt':
             lines = lines[1:] # the first line is version number
         for line in tqdm(lines):
             merge = line.strip()
             items = merge.split(" ")
             token = "".join(items)
             merge_dict[merge] = min_number
             for split_token in tokens:
                 #merge_dict[merge] = min_number
                 if token in split_token:
                     merge_dict[merge] += tokens[split_token]

    return merge_dict


def get_tokens(source_file, target_file, token_candidate_file, tokenizer='subword-nmt'):
    """
    Get all token cadidates associated with their frequencies. Here we take BPE-generated code segmentation as token candidates. 
    Arguments: 
        source_file (str): the source file from machine translation
        target_file (str): the target file from machine translation
        token_candidate_file: the token candididate file. Here we take BPE-generated code segmentation as candidates.
    """

    tokens = read_tokens(source_file, tokenizer=tokenizer)
    if target_file != None:
        tokens = read_tokens(target_file, tokens=tokens, tokenizer=tokenizer)
    merge_code = read_merge_code_frequency(token_candidate_file, tokens, tokenizer=tokenizer)
    return merge_code
