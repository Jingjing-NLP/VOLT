#!/usr/bin/env perl

# script for preprocessing language data prior to tokenization
# Start by Ulrich Germann, after noticing systematic preprocessing errors
# in some of the English Europarl data.
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;
use Getopt::Std;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

sub usage
{
  print "Script for preprocessing of raw language data prior to tokenization\n";
  print "Usage: $0 -l <language tag> [-b]\n";
  print "       -b: no buffering\n";
}

my %args;
getopt('l=s h b',\%args);
usage() && exit(0) if $args{'h'};
$|++ if $args{'b'};
if ($args{'l'} eq "en")
  {
    while (<>)
      {
	s/([[:alpha:]]\') s\b/$1s/g;
	print;
      }
  }
elsif ($args{'l'} eq "fr")
  {
    while (<>)
      {
	s/\b([[:alpha:]]\')\s+(?=[[:alpha:]])/$1/g;
	print;
      }
  }
else
  {
    print while <>;
  }
