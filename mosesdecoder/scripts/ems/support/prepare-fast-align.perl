#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;

my ($source_file,$target_file,$alignment_factors) = @ARGV;

# initialize data structures for factors
my (@SOURCE_FACTOR,@TARGET_FACTOR);
if (defined($alignment_factors)) {
  my ($source,$target) = split(/\-/,$alignment_factors);
  @SOURCE_FACTOR = split(/,/,$source);
  @TARGET_FACTOR = split(/,/,$target);
}

# loop through corpus file
open(SOURCE,$source_file);
open(TARGET,$target_file);
while(my $source = <SOURCE>) {
  my $target = <TARGET>;
  chop($source);
  chop($target);

  # remove markup
  foreach my $line (\$source,\$target) {
    $$line =~ s/\<[^\>]+\>/ /g;
    $$line =~ s/\s+/ /g;
    $$line =~ s/^ //;
    $$line =~ s/ $//;
  }

  # no factors
  if (!defined($alignment_factors)) {
    print "$source ||| $target\n";
    next;
  }

  foreach (split(/\s+/,$source)) {
    my @SOURCE_WORD = split(/\|/);
    for(my $i=0; $i<scalar(@SOURCE_FACTOR); $i++) {
      print "|" if $i;
      print "$SOURCE_WORD[$SOURCE_FACTOR[$i]]";
    }
    print " ";
  }
  print "|||";
  foreach (split(/\s+/,$target)) {
    print " ";
    my @TARGET_WORD = split(/\|/);
    for(my $i=0; $i<scalar(@TARGET_FACTOR); $i++) {
      print "|" if $i;
      print "$TARGET_WORD[$TARGET_FACTOR[$i]]";
    }
  }
  print "\n";
}
close(TARGET);
close(SOURCE);

