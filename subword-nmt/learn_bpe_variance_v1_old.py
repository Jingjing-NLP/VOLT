#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import math
import os
import sys
import inspect
import codecs
import re
import copy
import argparse
import warnings
import tempfile
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('learn-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--total-symbols', '-t', action="store_true",
        help="subtract number of characters from the symbols to be generated (so that '--symbols' becomes an estimate for the total number of symbols needed to encode text).")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")
    parser.add_argument('--maxmatrix', type=str, default="variance", help="variance or entropy")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser

def get_vocabulary(fobj, is_dict=False, num_workers=1):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    if is_dict:
        for i, line in enumerate(fobj):
            try:
                word, count = line.strip('\r\n ').split(' ')
            except:
                print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                sys.exit(1)
            vocab[word] += int(count)
    elif num_workers == 1 or fobj.name == '<stdin>':
        if num_workers > 1:
            warnings.warn("In parallel mode, the input cannot be STDIN. Using 1 processor instead.")
        for i, line in enumerate(fobj):
            prev_word = None
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[" ".join([char if char.strip() != "" else "</w>" for char in list(word)])] += 1
                    #print(" ".join(list(word)))
                #if  prev_word:
                #    vocab[" ".join(list(prev_word)) + " </w> " +" ".join(list(word))] += 1
                prev_word = word
    elif num_workers > 1:

        if sys.version_info < (3, 0):
            print("Parallel mode is only supported in Python3.")
            sys.exit(1)

        with open(fobj.name, encoding="utf8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = int(size / num_workers)
            offsets = [0 for _ in range(num_workers + 1)]
            for i in range(1, num_workers):
                f.seek(chunk_size * i)
                pos = f.tell()
                while True:
                    try:
                        line = f.readline()
                        break
                    except UnicodeDecodeError:
                        pos -= 1
                        f.seek(pos)
                offsets[i] = f.tell()
                assert 0 <= offsets[i] < 1e20, "Bad new line separator, e.g. '\\r'"

        vocab_files = []
        pool = Pool(processes=num_workers)
        for i in range(num_workers):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            vocab_files.append(tmp)
            pool.apply_async(_get_vocabulary, (fobj.name, tmp.name, offsets[i], offsets[i + 1]))
        pool.close()
        pool.join()
        import pickle
        for i in range(num_workers):
            with open(vocab_files[i].name, 'rb') as f:
                vocab += pickle.load(f)
            os.remove(vocab_files[i].name)
    else:
        raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))
    #for word in vocab: print(word)
    #print(vocab[2])
    return vocab

def _get_vocabulary(infile, outfile, begin, end):
    import pickle
    vocab = Counter()
    with open(infile, encoding="utf8") as f:
        f.seek(begin)
        line = f.readline()
        while line:
            prev_word = None
            pos = f.tell()
            assert 0 <= pos < 1e20, "Bad new line separator, e.g. '\\r'"
            if end > 0 and pos > end:
                break
            for word in line.strip('\r\n ').split(' '):
                if word:
                    vocab[" ".join([char if char.strip() != "" else "</w>" for char in list(word)])] += 1
                #if prev_word:
                #    vocab[" ".join(list(prev_word)) + " </s> " +" ".join(list(word))] += 1


            line = f.readline()
    with open(outfile, 'wb') as f:
        pickle.dump(vocab, f)

def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i-1:i+1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1

 
def get_pair_statistics(vocab, update=False, first=None, second=None):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)
    astats = {}
    
    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))
    #print(vocab)                 
    for i, (word, freq) in enumerate(vocab):
        #word = item
        #print(item,vocab[item])
        #freq = vocab[item]
        #prev_char = word[0]
        
        #current_token = word[1]
        #after_token = word[2]
        if update == True:
           if first not in word and  second not in word: continue
        #word = [char for char in word if char.strip() != ""] 
        for j, char in enumerate(word):
            #print(word[j+1] == char)
            if j == 0 and j+1 < len(word):
                after_token = word[j+1]
                #stats[char, after_token] += freq
                if char not in astats:
                   astats[char] = {}
                if " ".join([char, after_token]) not in astats[char]:
                   astats[char][" ".join([char, after_token])] = 0
                astats[char][" ".join([char, after_token])] += 1
                indices[char, after_token][i] += 1
            elif j == len(word)-1 and j-1 >= 0:
                prev_char = word[j-1]
                #stats[prev_char, char] += freq
                if char not in astats: astats[char] = {}
                if " ".join([prev_char, char]) not in astats[char]:
                   astats[char][" ".join([prev_char, char])] = 0
                astats[char][" ".join([prev_char, char])] += 1
                indices[prev_char, char][i] += 1
            elif j > 0 and j < len(word):
                prev_char = word[j-1]
                after_token = word[j+1]
                #print(prev_char, char, after_token)
                #if prev_char == " ": prev_char = "</w>"
                #if char == " ": char = "</w>"
                #stats[prev_char, char] += freq
                #stats[char, after_token] += freq
                if char not in astats:
                    astats[char]={}
                if " ".join([prev_char, char]) not in astats[char]:
                    astats[char][" ".join([prev_char, char])] = 0
                if " ".join([char, after_token]) not in astats[char]:
                    astats[char][" ".join([char, after_token])] = 0
                astats[char][" ".join([prev_char, char])] += 1
                astats[char][" ".join([char, after_token])] += 1

            
                #stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                indices[char, after_token][i] += 1
                #prev_char = char
                #print(vocab[i], prev_char, char, indices[prev_char, char], i)
                #prev_char = char

    return  indices,astats

import numpy
def get_max_variance(astats, flag="variance"):
    char_variance = {}
    #char_entropy = {}
    
    for char in astats:
         
        list1 = []
        for bigram in astats[char]:
            list1.append(astats[char][bigram])
        if flag == "variance":
            char_variance[char] = numpy.std(list1)
        if "entropy" in flag:
            entropy = 0
            sumlist1 = sum(list1)*1.0
            for number in list1:
                entropy -= number/sumlist1 * math.log(number/sumlist1)
            if flag == "entropy": char_variance[char] = entropy
            else: char_variance[char] = -entropy
    #print(char_variance)
    max_value = None
    max_index = None
    i = 0
    #print(char_variance)
    for char in char_variance:
        if i == 0: 
           max_value = char_variance[char]
           max_index = char

        i += 1
        if char_variance[char] > max_value:
            max_value = char_variance[char]
            max_index = char
    #print(max_index) 
    target_char = max_index
    #char = sorted(char_variance.items(), key=lambda x:x[1], reverse=True)[1]
    max_value = 0
    max_index = None
    #if max_index == None: print(len(astats) == 0, target_char in astats)
    #print(target_char, astats[target_char], "target_char")
    for bigram in astats[target_char]:
        if astats[target_char][bigram] > max_value:
           max_value = astats[target_char][bigram]
           max_index = bigram
    max_index = max_index.split(" ")
    #print(target_char, max_index, max_value)
    return max_index
def replace_pair(pair, vocab, indices,  astats):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    #print(vocab)
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    #if pair_str in astats: print(pair_str)
    astats[pair_str] = {}
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    #print(indices[pair])
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        #print(vocab[j], j,  pair,  "pair")
        #word = word.split(" ")
        #print(pair, word)
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))
        
        #print(pair, new_word, "new word")
        
        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))
    del astats[first]
    if second in astats: del astats[second]
    for item in astats:
        if " ".join([first, item]) in astats[item]:  del astats[item][" ".join([first, item])]
        if " ".join([item, first]) in astats[item]:  del astats[item][" ".join([item, first])]
        if " ".join([second, item]) in astats[item]: del astats[item][" ".join([second, item])]
        if " ".join([item, second]) in astats[item]: del astats[item][" ".join([item, second])]
    del indices[pair]
    #for pair_iter in indices.keys():
    #    if first in pair_iter or second in pair_iter:
    #       indices[pair_iter]={}
  

    return changes

def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def learn_bpe(infile, outfile, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False, num_workers=1, flag="variance"):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    outfile.write('#version: 0.2\n')

    vocab = get_vocabulary(infile, is_dict, num_workers)
    vocab = dict([(tuple(x.split()[:-1])+(x.split()[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    #sorted_vocab = vocab
    #print("vocab size", len(vocab))
    #print(vocab)
    #stats, indices = get_pair_statistics(sorted_vocab)
    #big_stats = copy.deepcopy(stats)

    #if total_symbols:
    #    uniq_char_internal = set()
    #    uniq_char_final = set()
    #    for word in vocab:
    #        for char in word[:-1]:
    #            uniq_char_internal.add(char)
    #        uniq_char_final.add(word[-1])
    #    sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
    #    sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
    #    sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
    #    num_symbols -= len(uniq_char_internal) + len(uniq_char_final)

    # threshold is inspired by Zipfian assumption, but should only affect speed
    #threshold = max(stats.values()) / 10
    for i in range(num_symbols):
        if i == 0:
          indices,astats = get_pair_statistics(sorted_vocab)
        else:
          indices, astats = get_pair_statistics(sorted_vocab, True, pair[0], pair[1])
        #print(len(stats), len(astats))
        #print(sorted_vocab)
        
        pair = get_max_variance(astats,flag)
        #print(pair)
        #if stats[tuple(pair)] < min_frequency:
        #    sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
        #    break

        #if verbose:
        #    sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        if pair[0].strip()=="" or pair[1].strip()=="": continue
        outfile.write('{0} {1}\n'.format(pair[0], pair[1]))
        
        changes = replace_pair(tuple(pair), sorted_vocab, indices, astats)
        #update_pair_statistics(most_frequent, changes, stats, indices)


if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.simplefilter('default')
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()
    flag = args.maxmatrix
    if args.num_workers <= 0:
        args.num_workers = cpu_count()

    if sys.version_info < (3, 0) and args.num_workers > 1:
        args.num_workers = 1
        warnings.warn("Parallel mode is only supported in Python3. Using 1 processor instead.")

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    #print(flag)
    learn_bpe(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input, total_symbols=args.total_symbols, num_workers=args.num_workers, flag=flag)
