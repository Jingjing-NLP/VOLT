import os
import sys
from tqdm import tqdm
from collections import OrderedDict

def read(path, dicts={}):
    with open(path, 'r') as sr:
        #total_lines = 1000000
        #read_line = 0
        lines = sr.readlines()
        #total_lines = len(lines) 
        total_lines = min(len(lines), 1000000)
        for line_number in tqdm(range(total_lines)):
            #line = sr.readline()
            line = lines[line_number]
            new_line = line.replace("@@ ", "")
            words = new_line.split(" ")
            for word in words:
                #word = word.split("@@")
                for length in range(1, 2):
                    for i in range(0, len(word)-length+1):
                        if " ".join(word[i:i+length]).strip() != "":
                            if " ".join(word[i:i+length]).strip() not in dicts:
                                dicts[" ".join(word[i:i+length]).strip()] = 0
                            dicts[" ".join(word[i:i+length]).strip()] += 1
    #dicts = {key:val for key, val in dicts.items() if val > 100}
    return dicts

def write(path, new_dicts):
    with open(path, "w") as sw:
         for item in new_dicts:
             sw.write(item[0] +" "+ str(item[1]) + "\n")


def get_chars(source_file, target_file):

  dicts = read(source_file)
  dicts = read(target_file, dicts=dicts)
  dicts = {key:val for key, val in dicts.items() if val > 2}
  new_dicts = sorted(dicts.items(), key=lambda x:x[1], reverse=True)
  #write(char_file,new_dicts)
  return_dicts = OrderedDict()
  for item in new_dicts:
      return_dicts[item[0]] = item[1]
  return return_dicts
    


