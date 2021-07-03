#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.
Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings
import random
import tempfile
from multiprocessing import Pool, cpu_count

# hack for python2/3 compatibility
from io import open
argparse.open = open

class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):

        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes.read().rstrip('\n').split('\n')) if (n < merges or merges == -1)]

        #for i, item in enumerate(self.bpe_codes):
        #    if len(item) != 2:
        #        sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
        #        sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
        #        sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

        self.cache = {}

    def process_lines(self, filename, outfile, dropout=0, num_workers=1):

        if sys.version_info < (3, 0):
            print("Parallel mode is only supported in Python3.")
            sys.exit(1)

        if num_workers == 1:
            _process_lines(self, filename, outfile, dropout, 0, 0)
        elif num_workers > 1:
            with open(filename, encoding="utf-8") as f:
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
            res_files = []
            pool = Pool(processes=num_workers)
            for i in range(num_workers):
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.close()
                res_files.append(tmp)
                pool.apply_async(_process_lines, (self, filename, tmp.name, dropout, offsets[i], offsets[i + 1]))
            pool.close()
            pool.join()
            for i in range(num_workers):
                with open(res_files[i].name, encoding="utf-8") as fi:
                    for line in fi:
                        outfile.write(line)
                os.remove(res_files[i].name)
        else:
            raise ValueError('`num_workers` is expected to be a positive number, but got {}.'.format(num_workers))

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line, dropout)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence, dropout=0):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def _process_lines(bpe, filename, outfile, dropout, begin, end):
    if isinstance(outfile, str):
        fo = open(outfile, "w", encoding="utf-8")
    else:
        fo = outfile
    with open(filename, encoding="utf-8") as f:
        f.seek(begin)
        line = f.readline()
        while line:
            pos = f.tell()
            assert 0 <= pos < 1e20, "Bad new line separator, e.g. '\\r'"
            if end > 0 and pos > end:
                break
            fo.write(bpe.process_line(line, dropout))
            line = f.readline()
    if isinstance(outfile, str):
        fo.close()

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('apply-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--dropout', type=float, default=0,
        metavar="P",
        help="Dropout BPE merge operations with probability P (Provilkov et al., 2019). Use this on training data only.")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")
    parser.add_argument(
        '--seed', type=int, default=None,
        metavar="S",
        help="Random seed for the random number generators (e.g. for BPE dropout with --dropout).")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")

    return parser

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if not dropout and orig in cache:
        return cache[orig]

    if glossaries_regex and glossaries_regex.match(orig):
        cache[orig] = (orig,)
        return (orig,)

    if len(orig) == 1:
        return orig

    if version == (0, 1):
        word = list(orig) + ['</w>']
    elif version == (0, 2): # more consistent handling of word-final segments
        word = list(orig[:-1]) + [orig[-1] + '</w>']
    else:
        raise NotImplementedError

    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram = ''.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    word = tuple(word)
    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    # regex equivalent of (if word == glossary or glossary not in word)
    if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
        return [word]
    else:
        segments = re.split(r'({})'.format(glossary), word)
        segments, ending = segments[:-1], segments[-1]
        segments = list(filter(None, segments)) # Remove empty strings in regex group.
        return segments + [ending.strip('\r\n ')] if ending != '' else segments

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
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    if args.num_workers <= 0:
        args.num_workers = cpu_count()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    if sys.version_info < (3, 0):
        args.separator = args.separator.decode('UTF-8')
        if args.glossaries:
            args.glossaries = [g.decode('UTF-8') for g in args.glossaries]
        if args.num_workers > 1:
            args.num_workers = 1
            warnings.warn("Parallel mode is only supported in Python3. Using 1 processor instead.")

    if args.seed is not None:
        random.seed(args.seed)

    bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries)

    if args.input.name == '<stdin>' or args.num_workers == 1:
        if args.num_workers > 1:
            warnings.warn("In parallel mode, the input cannot be STDIN. Using 1 processor instead.")
        for line in args.input:
            args.output.write(bpe.process_line(line, args.dropout))
    else:
        bpe.process_lines(args.input.name, args.output, args.dropout, args.num_workers)

