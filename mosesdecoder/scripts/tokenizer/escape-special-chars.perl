#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;
use strict;

while (@ARGV) {
    $_ = shift;
    /^-b$/ && ($| = 1, next); # not buffered (flush each line)
}

while(<STDIN>) {
  chop;

  # avoid general madness
  s/[\000-\037]//g;
  s/\s+/ /g;
	s/^ //g;
	s/ $//g;

  # special characters in moses
  s/\&/\&amp;/g;   # escape escape
  s/\|/\&#124;/g;  # factor separator
  s/\</\&lt;/g;    # xml
  s/\>/\&gt;/g;    # xml
  s/\'/\&apos;/g;  # xml
  s/\"/\&quot;/g;  # xml
  s/\[/\&#91;/g;   # syntax non-terminal
  s/\]/\&#93;/g;   # syntax non-terminal

  # restore xml instructions
  s/\&lt;(\S+) translation=&quot;(.+?)&quot;&gt; (.+?) &lt;\/(\S+)&gt;/\<$1 translation=\"$2\"> $3 <\/$4>/g;
  print $_."\n";
}
