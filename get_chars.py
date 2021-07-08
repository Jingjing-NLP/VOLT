import os
import sys
from tqdm import tqdm
from collections import OrderedDict

def read(path, dicts={}, max_line_number=1e7, tokenizer='subword-nmt'): #"@@ "
    """
    Read characters and their frequencies from the give file. We set the max_line_number=1,000,000 to sample charater distributions.
    Arguments:
        path (str): the file path 
        dicts (dict): the dict to save characters and their frequencies
        max_line_number (int): the maximum number of lines 
        special_tokens (str): chars removed from the final character distributions; here we remove the tokens added by tokenization methods 
    Returns: 
        dicts (dict): storing characters and their frequencies
    """
    with open(path, 'r') as sr:
        lines = sr.readlines()
        total_lines = min(len(lines), max_line_number)
        total_lines = int(total_lines)
        for line_number in tqdm(range(total_lines)):
            line = lines[line_number]
            if tokenizer == 'subword-nmt':
                line = line.replace("@@ ", "")
            words = line.split(" ")
            for word in words:
                length = 1    
                for i in range(0, len(word)):
                    if " ".join(word[i:i+length]).strip() != "":
                        if " ".join(word[i:i+length]).strip() not in dicts:
                            dicts[" ".join(word[i:i+length]).strip()] = 0
                        dicts[" ".join(word[i:i+length]).strip()] += 1
    return dicts



def get_chars(source_file, target_file, tokenizer='subword-nmt'):
  """
  Get charaters and associated frequencies from source file and target file in machine translation.
  Arguments:
      source_file (str): source file in machine translation. Each line contains one source sentence.
      target_file (str): target file in machine translation. Each line contains one target sentence.
  Returns:
      return_dicts (dict): sorted characters. 
  """
  dicts = read(source_file, tokenizer=tokenizer)
  dicts = read(target_file, dicts=dicts, tokenizer=tokenizer)
 
  # filter charaters with frequency less than 2
  dicts = {key:val for key, val in dicts.items() if val > 2}
  new_dicts = sorted(dicts.items(), key=lambda x:x[1], reverse=True)
  return_dicts = OrderedDict()
  for item in new_dicts:
      return_dicts[item[0]] = item[1]
  return return_dicts
    


