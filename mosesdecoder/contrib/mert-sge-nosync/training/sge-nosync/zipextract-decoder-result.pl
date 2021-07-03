#!/usr/bin/perl -w 
# $Id$
# Usage:
# mert-moses.pl <foreign> <english> <decoder-executable> <decoder-config>
# For other options see below or run 'mert-moses.pl --help'

# Notes:
# <foreign> and <english> should be raw text files, one sentence per line
# <english> can be a prefix, in which case the files are <english>0, <english>1, etc. are used

# Excerpts from revision history

# Sept 2011   multi-threaded mert (Barry Haddow)
# 3 Aug 2011  Added random directions, historic best, pairwise ranked (PK)
# Jul 2011    simplifications (Ondrej Bojar)
#             -- rely on moses' -show-weights instead of parsing moses.ini 
#                ... so moses is also run once *before* mert starts, checking
#                    the model to some extent
#             -- got rid of the 'triples' mess;
#                use --range to supply bounds for random starting values:
#                --range tm:-3..3 --range lm:-3..3
# 5 Aug 2009  Handling with different reference length policies (shortest, average, closest) for BLEU 
#             and case-sensistive/insensitive evaluation (Nicola Bertoldi)
# 5 Jun 2008  Forked previous version to support new mert implementation.
# 13 Feb 2007 Better handling of default values for lambda, now works with multiple
#             models and lexicalized reordering
# 11 Oct 2006 Handle different input types through parameter --inputype=[0|1]
#             (0 for text, 1 for confusion network, default is 0) (Nicola Bertoldi)
# 10 Oct 2006 Allow skip of filtering of phrase tables (--no-filter-phrase-table)
#             useful if binary phrase tables are used (Nicola Bertoldi)
# 28 Aug 2006 Use either closest or average or shortest (default) reference
#             length as effective reference length
#             Use either normalization or not (default) of texts (Nicola Bertoldi)
# 31 Jul 2006 move gzip run*.out to avoid failure wit restartings
#             adding default paths
# 29 Jul 2006 run-filter, score-nbest and mert run on the queue (Nicola; Ondrej had to type it in again)
# 28 Jul 2006 attempt at foolproof usage, strong checking of input validity, merged the parallel and nonparallel version (Ondrej Bojar)
# 27 Jul 2006 adding the safesystem() function to handle with process failure
# 22 Jul 2006 fixed a bug about handling relative path of configuration file (Nicola Bertoldi) 
# 21 Jul 2006 adapted for Moses-in-parallel (Nicola Bertoldi) 
# 18 Jul 2006 adapted for Moses and cleaned up (PK)
# 21 Jan 2005 unified various versions, thorough cleanup (DWC)
#             now indexing accumulated n-best list solely by feature vectors
# 14 Dec 2004 reimplemented find_threshold_points in C (NMD)
# 25 Oct 2004 Use either average or shortest (default) reference
#             length as effective reference length (DWC)
# 13 Oct 2004 Use alternative decoders (DWC)
# Original version by Philipp Koehn

use strict;
use FindBin qw($Bin);
use File::Basename;
use File::Path;
use File::Spec;
use Cwd;

my $SCRIPTS_ROOTDIR = $Bin;
$SCRIPTS_ROOTDIR =~ s/\/training$//;
$SCRIPTS_ROOTDIR = $ENV{"SCRIPTS_ROOTDIR"} if defined($ENV{"SCRIPTS_ROOTDIR"});

## We preserve this bit of comments to keep the traditional weight ranges.
#     "w" => [ [ 0.0, -1.0, 1.0 ] ],  # word penalty
#     "d"  => [ [ 1.0, 0.0, 2.0 ] ],  # lexicalized reordering model
#     "lm" => [ [ 1.0, 0.0, 2.0 ] ],  # language model
#     "g"  => [ [ 1.0, 0.0, 2.0 ],    # generation model
# 	      [ 1.0, 0.0, 2.0 ] ],
#     "tm" => [ [ 0.3, 0.0, 0.5 ],    # translation model
# 	      [ 0.2, 0.0, 0.5 ],
# 	      [ 0.3, 0.0, 0.5 ],
# 	      [ 0.2, 0.0, 0.5 ],
# 	      [ 0.0,-1.0, 1.0 ] ],  # ... last weight is phrase penalty
#     "lex"=> [ [ 0.1, 0.0, 0.2 ] ],  # global lexical model
#     "I"  => [ [ 0.0,-1.0, 1.0 ] ],  # input lattice scores



# moses.ini file uses FULL names for lambdas, while this training script
# internally (and on the command line) uses ABBR names.
my @ABBR_FULL_MAP = qw(d=weight-d lm=weight-l tm=weight-t w=weight-w
  g=weight-generation lex=weight-lex I=weight-i);
my %ABBR2FULL = map {split/=/,$_,2} @ABBR_FULL_MAP;
my %FULL2ABBR = map {my ($a, $b) = split/=/,$_,2; ($b, $a);} @ABBR_FULL_MAP;

my $minimum_required_change_in_weights = 0.00001;
    # stop if no lambda changes more than this

my $verbose = 0;
my $usage = 0; # request for --help

# We assume that if you don't specify working directory,
# we set the default is set to `pwd`/mert-work
#@@# my $___WORKING_DIR = File::Spec->catfile(Cwd::getcwd(), "mert-work");
my $___WORKING_DIR = undef;
my $___DEV_F = undef; # required, input text to decode
my $___DEV_E = undef; # required, basename of files with references
my $___DECODER = undef; # required, pathname to the decoder executable
my $___CONFIG = undef; # required, pathname to startup ini file
my $___N_BEST_LIST_SIZE = 100;
my $___LATTICE_SAMPLES = 0;
my $submithost = "";
my $queue_flags = "-hard";  # extra parameters for parallelizer
      # the -l ws0ssmt was relevant only to JHU 2006 workshop
my $___JOBS = undef; # if parallel, number of jobs to use (undef or 0 -> serial)
my $___DECODER_FLAGS = ""; # additional parametrs to pass to the decoder
my $continue = 0; # should we try to continue from the last saved step?
my $skip_decoder = 0; # and should we skip the first decoder run (assuming we got interrupted during mert)
my $___FILTER_PHRASE_TABLE = 1; # filter phrase table
my $___PREDICTABLE_SEEDS = 0;
my $___START_WITH_HISTORIC_BESTS = 0; # use best settings from all previous iterations as starting points [Foster&Kuhn,2009]
my $___RANDOM_DIRECTIONS = 0; # search in random directions only
my $___NUM_RANDOM_DIRECTIONS = 0; # number of random directions, also works with default optimizer [Cer&al.,2008]
my $___PAIRWISE_RANKED_OPTIMIZER = 0; # use Hopkins&May[2011]
my $___PRO_STARTING_POINT = 0; # get a starting point from pairwise ranked optimizer
my $___RANDOM_RESTARTS = 20;
my $___HISTORIC_INTERPOLATION = 0; # interpolate optimize weights with previous iteration's weights [Hopkins&May,2011,5.4.3]
my $__THREADS = 0;
my $run = 0;

# Parameter for effective reference length when computing BLEU score
# Default is to use shortest reference
# Use "--shortest" to use shortest reference length
# Use "--average" to use average reference length
# Use "--closest" to use closest reference length
# Only one between --shortest, --average and --closest can be set
# If more than one choice the defualt (--shortest) is used
my $___SHORTEST = 0;
my $___AVERAGE = 0;
my $___CLOSEST = 0;

# Use "--nocase" to compute case-insensitive scores
my $___NOCASE = 0;

# Use "--nonorm" to non normalize translation before computing scores
my $___NONORM = 0;

# set 0 if input type is text, set 1 if input type is confusion network
my $___INPUTTYPE = 0; 


my $mertdir = undef; # path to new mert directory
my $mertargs = undef; # args to pass through to mert & extractor
my $mertmertargs = undef; # args to pass through to mert only
my $extractorargs = undef; # args to pass through to extractor only
my $filtercmd = undef; # path to filter-model-given-input.pl
my $filterfile = undef;
my $qsubwrapper = undef;
my $qsubwrapper_exit = undef;
my $moses_parallel_cmd = undef;
my $old_sge = 0; # assume sge<6.0
my $___CONFIG_ORIG = undef; # pathname to startup ini file before filtering
my $___ACTIVATE_FEATURES = undef; # comma-separated (or blank-separated) list of features to work on 
                                  # if undef work on all features
                                  # (others are fixed to the starting values)
my $___RANGES = undef;
my $prev_aggregate_nbl_size = -1; # number of previous step to consider when loading data (default =-1)
                                  # -1 means all previous, i.e. from iteration 1
                                  # 0 means no previous data, i.e. from actual iteration
                                  # 1 means 1 previous data , i.e. from the actual iteration and from the previous one
                                  # and so on
my $maximum_iterations = 25;

#####################
my $processfeatlistcmd = undef;
my $processfeatlistargs = undef;
my $createconfigcmd = undef;
my $createconfigargs = undef;
my $decoderargs = undef;
#####################

use Getopt::Long;
GetOptions(
  "working-dir=s" => \$___WORKING_DIR,
  "input=s" => \$___DEV_F,
  "inputtype=i" => \$___INPUTTYPE,
  "refs=s" => \$___DEV_E,
  "decoder=s" => \$___DECODER,
  "config=s" => \$___CONFIG,
  "nbest=i" => \$___N_BEST_LIST_SIZE,
  "lattice-samples=i" => \$___LATTICE_SAMPLES,
  "submithost=s" => \$submithost,
  "queue-flags=s" => \$queue_flags,
  "jobs=i" => \$___JOBS,
  "decoder-flags=s" => \$___DECODER_FLAGS,
  "continue" => \$continue,
  "skip-decoder" => \$skip_decoder,
  "shortest" => \$___SHORTEST,
  "average" => \$___AVERAGE,
  "closest" => \$___CLOSEST,
  "nocase" => \$___NOCASE,
  "nonorm" => \$___NONORM,
  "help" => \$usage,
  "verbose" => \$verbose,
  "mertdir=s" => \$mertdir,
  "mertargs=s" => \$mertargs,
  "extractorargs=s" => \$extractorargs,
  "mertmertargs=s" => \$mertmertargs,
  "rootdir=s" => \$SCRIPTS_ROOTDIR,
  "filtercmd=s" => \$filtercmd, # allow to override the default location
  "filterfile=s" => \$filterfile, # input to filtering script (useful for lattices/confnets)
  "qsubwrapper=s" => \$qsubwrapper, # allow to override the default location
  "mosesparallelcmd=s" => \$moses_parallel_cmd, # allow to override the default location
  "old-sge" => \$old_sge, #passed to moses-parallel
  "filter-phrase-table!" => \$___FILTER_PHRASE_TABLE, # (dis)allow of phrase tables
  "predictable-seeds" => \$___PREDICTABLE_SEEDS, # make random restarts deterministic
  "historic-bests" => \$___START_WITH_HISTORIC_BESTS, # use best settings from all previous iterations as starting points
  "random-directions" => \$___RANDOM_DIRECTIONS, # search only in random directions
  "run=i" => \$run,
  "number-of-random-directions=i" => \$___NUM_RANDOM_DIRECTIONS, # number of random directions
  "random-restarts=i" => \$___RANDOM_RESTARTS, # number of random restarts
  "activate-features=s" => \$___ACTIVATE_FEATURES, #comma-separated (or blank-separated) list of features to work on (others are fixed to the starting values)
  "range=s@" => \$___RANGES,
  "prev-aggregate-nbestlist=i" => \$prev_aggregate_nbl_size, #number of previous step to consider when loading data (default =-1, i.e. all previous)
  "maximum-iterations=i" => \$maximum_iterations,
  "pairwise-ranked" => \$___PAIRWISE_RANKED_OPTIMIZER,
  "pro-starting-point" => \$___PRO_STARTING_POINT,
  "historic-interpolation=f" => \$___HISTORIC_INTERPOLATION,
  "threads=i" => \$__THREADS
) or exit(1);

# the 4 required parameters can be supplied on the command line directly
# or using the --options
if (scalar @ARGV == 2) {
  # required parameters: input_file references_basename decoder_executable
#  $___DEV_F = shift;
  $___DEV_E = shift;
#  $___DECODER = shift;
  $___CONFIG = shift;
}

# if ($usage || !defined $___DEV_F || !defined $___DEV_E || !defined $___DECODER || !defined $___CONFIG) {
if ($usage || !defined $___CONFIG || !defined $___DEV_E ){
  print STDERR "usage: $0 reference decoder.ini
Options:
  --working-dir=mert-dir ... where all the files are created
  --nbest=100            ... how big nbestlist to generate
  --lattice-samples      ... how many lattice samples (Chatterjee & Cancedda, emnlp 2010)
  --jobs=N               ... set this to anything to run moses in parallel
  --mosesparallelcmd=STR ... use a different script instead of moses-parallel
  --queue-flags=STRING   ... anything you with to pass to qsub, eg.
                             '-l ws06osssmt=true'. The default is: '-hard'
                             To reset the parameters, please use 
                             --queue-flags=' '
                             (i.e. a space between the quotes).
  --decoder-flags=STRING ... extra parameters for the decoder
  --continue             ... continue from the last successful iteration
  --skip-decoder         ... skip the decoder run for the first time,
                             assuming that we got interrupted during
                             optimization
  --shortest --average --closest
                         ... Use shortest/average/closest reference length
                             as effective reference length (mutually exclusive)
  --nocase               ... Do not preserve case information; i.e.
                             case-insensitive evaluation (default is false).
  --nonorm               ... Do not use text normalization (flag is not active,
                             i.e. text is NOT normalized)
  --filtercmd=STRING     ... path to filter-model-given-input.pl
  --filterfile=STRING    ... path to alternative to input-text for filtering
                             model. useful for lattice decoding
  --rootdir=STRING       ... where do helpers reside (if not given explicitly)
  --mertdir=STRING       ... path to new mert implementation
  --mertargs=STRING      ... extra args for both extractor and mert
  --extractorargs=STRING ... extra args for extractor only
  --mertmertargs=STRING  ... extra args for mert only 
  --scorenbestcmd=STRING ... path to score-nbest.py
  --old-sge              ... passed to parallelizers, assume Grid Engine < 6.0
  --inputtype=[0|1|2]    ... Handle different input types: (0 for text,
                             1 for confusion network, 2 for lattices,
                             default is 0)
  --no-filter-phrase-table ... disallow filtering of phrase tables
                              (useful if binary phrase tables are available)
  --random-restarts=INT  ... number of random restarts (default: 20)
  --predictable-seeds    ... provide predictable seeds to mert so that random
                             restarts are the same on every run
  --range=tm:0..1,-1..1  ... specify min and max value for some features
                             --range can be repeated as needed.
                             The order of the various --range specifications
                             is important only within a feature name.
                             E.g.:
                               --range=tm:0..1,-1..1 --range=tm:0..2
                             is identical to:
                               --range=tm:0..1,-1..1,0..2
                             but not to:
                               --range=tm:0..2 --range=tm:0..1,-1..1 
  --activate-features=STRING  ... comma-separated list of features to optimize,
                                  others are fixed to the starting values
                                  default: optimize all features
                                  example: tm_0,tm_4,d_0
  --prev-aggregate-nbestlist=INT ... number of previous step to consider when
                                     loading data (default = $prev_aggregate_nbl_size)
                                    -1 means all previous, i.e. from iteration 1
                                     0 means no previous data, i.e. only the
                                       current iteration
                                     N means this and N previous iterations

  --maximum-iterations=ITERS ... Maximum number of iterations. Default: $maximum_iterations
  --random-directions               ... search only in random directions
  --number-of-random-directions=int ... number of random directions
                                        (also works with regular optimizer, default: 0)
  --pairwise-ranked         ... Use PRO for optimisation (Hopkins and May, emnlp 2011)
  --pro-starting-point      ... Use PRO to get a starting point for MERT
  --threads=NUMBER          ... Use multi-threaded mert (must be compiled in).
  --historic-interpolation  ... Interpolate optimized weights with prior iterations' weight
                                (parameter sets factor [0;1] given to current weights)
";
  exit 1;
}


# Check validity of input parameters and set defaults if needed

print STDERR "Using SCRIPTS_ROOTDIR: $SCRIPTS_ROOTDIR\n";

# path of script for filtering phrase tables and running the decoder
$filtercmd="$SCRIPTS_ROOTDIR/training/filter-model-given-input.pl" if !defined $filtercmd;

if ( ! -x $filtercmd && ! $___FILTER_PHRASE_TABLE) {
  print STDERR "Filtering command not found: $filtercmd.\n";
  print STDERR "Use --filtercmd=PATH to specify a valid one or --no-filter-phrase-table\n";
  exit 1;
}

# $qsubwrapper="$SCRIPTS_ROOTDIR/generic/qsub-wrapper.pl" if !defined $qsubwrapper;
$qsubwrapper = "$SCRIPTS_ROOTDIR/generic/qsub-wrapper-sge-nosync.pl" if !defined $qsubwrapper;

$qsubwrapper_exit = "$SCRIPTS_ROOTDIR/generic/qsub-wrapper-exit-sge-nosync.pl" if !defined $qsubwrapper_exit;

# $moses_parallel_cmd = "$SCRIPTS_ROOTDIR/generic/moses-parallel.pl"
#  if !defined $moses_parallel_cmd;
$moses_parallel_cmd = "$SCRIPTS_ROOTDIR/generic/moses-parallel-sge-nosync.pl"
  if !defined $moses_parallel_cmd;

if (!defined $mertdir) {
  $mertdir = "$SCRIPTS_ROOTDIR/../mert";
  print STDERR "Assuming --mertdir=$mertdir\n";
}

my $mert_extract_cmd = "$mertdir/extractor";
my $mert_mert_cmd = "$mertdir/mert";
my $mert_pro_cmd = "$mertdir/pro";

die "Not executable: $mert_extract_cmd" if ! -x $mert_extract_cmd;
die "Not executable: $mert_mert_cmd" if ! -x $mert_mert_cmd;
die "Not executable: $mert_pro_cmd" if ! -x $mert_pro_cmd;

my $pro_optimizer = "$mertdir/megam_i686.opt"; # or set to your installation
if (($___PAIRWISE_RANKED_OPTIMIZER || $___PRO_STARTING_POINT) && ! -x $pro_optimizer) {
  print "did not find $pro_optimizer, installing it in $mertdir\n";
  `cd $mertdir; wget http://www.cs.utah.edu/~hal/megam/megam_i686.opt.gz;`;
  `gunzip $pro_optimizer.gz`;
  `chmod +x $pro_optimizer`;
  die("ERROR: Installation of megam_i686.opt failed! Install by hand from http://www.cs.utah.edu/~hal/megam/") unless -x $pro_optimizer;
}

$mertargs = "" if !defined $mertargs;

my $scconfig = undef;
if ($mertargs =~ /\-\-scconfig\s+(.+?)(\s|$)/){
  $scconfig=$1;
  $scconfig =~ s/\,/ /g;
  $mertargs =~ s/\-\-scconfig\s+(.+?)(\s|$)//;
}

# handling reference lengh strategy
if (($___CLOSEST + $___AVERAGE + $___SHORTEST) > 1){
  die "You can specify just ONE reference length strategy (closest or shortest or average) not both\n";
}

if ($___SHORTEST){
  $scconfig .= " reflen:shortest";
}elsif ($___AVERAGE){
  $scconfig .= " reflen:average";
}elsif ($___CLOSEST){
  $scconfig .= " reflen:closest";
}

# handling case-insensitive flag
if ($___NOCASE) {
  $scconfig .= " case:false";
}else{
  $scconfig .= " case:true";
}
$scconfig =~ s/^\s+//;
$scconfig =~ s/\s+$//;
$scconfig =~ s/\s+/,/g;

$scconfig = "--scconfig $scconfig" if ($scconfig);

my $mert_extract_args=$mertargs;
$mert_extract_args .=" $scconfig";
$mert_extract_args .=" $extractorargs";

$mertmertargs = "" if !defined $mertmertargs;

my $mert_mert_args="$mertargs $mertmertargs";
$mert_mert_args =~ s/\-+(binary|b)\b//;
$mert_mert_args .=" $scconfig";
if ($___ACTIVATE_FEATURES){ $mert_mert_args .=" -o \"$___ACTIVATE_FEATURES\""; }

my ($just_cmd_filtercmd,$x) = split(/ /,$filtercmd);
die "Not executable: $just_cmd_filtercmd" if ! -x $just_cmd_filtercmd;
die "Not executable: $moses_parallel_cmd" if defined $___JOBS && ! -x $moses_parallel_cmd;
die "Not executable: $qsubwrapper" if defined $___JOBS && ! -x $qsubwrapper;
#@@# die "Not executable: $___DECODER" if ! -x $___DECODER;

#@@# my $input_abs = ensure_full_path($___DEV_F);
#@@# die "File not found: $___DEV_F (interpreted as $input_abs)."
#@@#   if ! -e $input_abs;
#@@# $___DEV_F = $input_abs;

# Option to pass to qsubwrapper and moses-parallel
my $pass_old_sge = $old_sge ? "-old-sge" : "";

#@@# my $decoder_abs = ensure_full_path($___DECODER);
#@@# die "File not executable: $___DECODER (interpreted as $decoder_abs)."
#@@#   if ! -x $decoder_abs;
#@@# $___DECODER = $decoder_abs;

my $ref_abs = ensure_full_path($___DEV_E);
# check if English dev set (reference translations) exist and store a list of all references
my @references;
if (-e $ref_abs) {
  push @references, $ref_abs;
}
else {
 # if multiple file, get a full list of the files
   my $part = 0;
   if (! -e $ref_abs."0" && -e $ref_abs.".ref0") {
       $ref_abs .= ".ref";
   }
   while (-e $ref_abs.$part) {
       push @references, $ref_abs.$part;
       $part++;
   }
   die("Reference translations not found: $___DEV_E (interpreted as $ref_abs)") unless $part;
}

my $config_abs = ensure_full_path($___CONFIG);
die "File not found: $___CONFIG (interpreted as $config_abs)."
  if ! -e $config_abs;
$___CONFIG = $config_abs;

# moses should use our config
if ($___DECODER_FLAGS =~ /(^|\s)-(config|f) /
|| $___DECODER_FLAGS =~ /(^|\s)-(ttable-file|t) /
|| $___DECODER_FLAGS =~ /(^|\s)-(distortion-file) /
|| $___DECODER_FLAGS =~ /(^|\s)-(generation-file) /
|| $___DECODER_FLAGS =~ /(^|\s)-(lmodel-file) /
|| $___DECODER_FLAGS =~ /(^|\s)-(global-lexical-file) /
) {
  die "It is forbidden to supply any of -config, -ttable-file, -distortion-file, -generation-file or -lmodel-file in the --decoder-flags.\nPlease use only the --config option to give the config file that lists all the supplementary files.";
}

# as weights are normalized in the next steps (by cmert)
# normalize initial LAMBDAs, too
my $need_to_normalize = 1;

#store current directory and create the working directory (if needed)
my $cwd = `pawd 2>/dev/null`; 
if(!$cwd){$cwd = `pwd`;}
chomp($cwd);

$___WORKING_DIR = $cwd if (!defined $___WORKING_DIR); 
chomp $___WORKING_DIR;

print STDERR "working dir is $___WORKING_DIR\n";
#@@# mkpath($___WORKING_DIR);

{
# open local scope

#chdir to the working directory
chdir($___WORKING_DIR) or die "Can't chdir to $___WORKING_DIR";

# fixed file names
my $mert_outfile = "mert.out";
my $mert_logfile = "mert.log";
my $weights_in_file = "init.opt";
my $weights_out_file = "weights.txt";

# set start run
my $start_run = 1;
my $bestpoint = undef;
my $devbleu = undef;
my $sparse_weights_file = undef;
my $jobid = -1;

my $prev_feature_file = undef;
my $prev_score_file = undef;
my $prev_init_file = undef;


#########################
# set jobid to trace different jobs
my $prevjid = undef;



#### my $run=$start_run-1;

my $oldallsorted = undef;
my $allsorted = undef;

my $nbest_file=undef;
my $lsamp_file=undef; #Lattice samples
my $orig_nbest_file=undef; # replaced if lattice sampling
my $cmd=undef;


  

     
  $nbest_file="run$run.best$___N_BEST_LIST_SIZE.out";     
  safesystem("gzip -f $nbest_file") or die "Failed to gzip run*out";
  $nbest_file = $nbest_file.".gz";


  # extract score statistics and features from the nbest lists
  print STDERR "Scoring the nbestlist.\n";

  my $base_feature_file = "features.dat";
  my $base_score_file = "scores.dat";
  my $feature_file = "run$run.${base_feature_file}";
  my $score_file = "run$run.${base_score_file}";

  $cmd = "$mert_extract_cmd $mert_extract_args --scfile $score_file --ffile $feature_file -r ".join(",", @references)." -n $nbest_file";
  $cmd = create_extractor_script($cmd, $___WORKING_DIR);

  &submit_or_exec($cmd,"extract.out","extract.err","extract.id");



} # end of local scope

sub get_weights_from_mert {
  my ($outfile,$logfile,$weight_count,$sparse_weights) = @_;
  my ($bestpoint,$devbleu);
  if ($___PAIRWISE_RANKED_OPTIMIZER || ($___PRO_STARTING_POINT && $logfile =~ /pro/)) {
    open(IN,$outfile) or die "Can't open $outfile";
    my (@WEIGHT,$sum);
    for(my $i=0;$i<$weight_count;$i++) { push @WEIGHT, 0; }
    while(<IN>) {
      # regular features
      if (/^F(\d+) ([\-\.\de]+)/) {
        $WEIGHT[$1] = $2;
        $sum += abs($2);
      }
      # sparse features
      elsif(/^(.+_.+) ([\-\.\de]+)/) {
        $$sparse_weights{$1} = $2;
      }
    }
    $devbleu = "unknown";
    foreach (@WEIGHT) { $_ /= $sum; }
    foreach (keys %{$sparse_weights}) { $$sparse_weights{$_} /= $sum; }
    $bestpoint = join(" ",@WEIGHT);
    close IN;
  }
  else {
    open(IN,$logfile) or die "Can't open $logfile";
    while (<IN>) {
      if (/Best point:\s*([\s\d\.\-e]+?)\s*=> ([\-\d\.]+)/) {
        $bestpoint = $1;
        $devbleu = $2;
        last;
      }
    }
    close IN;
  }
  return ($bestpoint,$devbleu);
}



sub sanity_check_order_of_lambdas {
  my $featlist = shift;
  my $filename_or_stream = shift;

  my @expected_lambdas = @{$featlist->{"names"}};
  my @got = get_order_of_scores_from_nbestlist($filename_or_stream);
  die "Mismatched lambdas. Decoder returned @got, we expected @expected_lambdas"
  if "@got" ne "@expected_lambdas";
}
    
sub get_featlist_from_moses {
  # run moses with the given config file and return the list of features and
  # their initial values
  my $configfn = shift;
  my $featlistfn = "./features.list";
  if (-e $featlistfn) {
    print STDERR "Using cached features list: $featlistfn\n";
  } else {
    print STDERR "Asking moses for feature names and values from $___CONFIG\n";
    my $cmd = "$___DECODER $___DECODER_FLAGS -config $configfn  -inputtype $___INPUTTYPE -show-weights > $featlistfn";
    safesystem($cmd) or die "Failed to run moses with the config $configfn";
  }

  # read feature list
  my @names = ();
  my @startvalues = ();
  open(INI,$featlistfn) or die "Can't read $featlistfn";
  my $nr = 0;
  my @errs = ();
  while (<INI>) {
    $nr++;
    chomp;
    /^(.+) (\S+) (\S+)$/ || die("invalid feature: $_");
    my ($longname, $feature, $value) = ($1,$2,$3);
    next if $value eq "sparse";
    push @errs, "$featlistfn:$nr:Bad initial value of $feature: $value\n"
    if $value !~ /^[+-]?[0-9.e]+$/;
    push @errs, "$featlistfn:$nr:Unknown feature '$feature', please add it to \@ABBR_FULL_MAP\n"
  if !defined $ABBR2FULL{$feature};
    push @names, $feature;
    push @startvalues, $value;
  }
  close INI;
  if (scalar @errs) {
    print STDERR join("", @errs);
   exit 1;
  }
  return {"names"=>\@names, "values"=>\@startvalues};
}

sub get_order_of_scores_from_nbestlist {
  # read the first line and interpret the ||| label: num num num label2: num ||| column in nbestlist
  # return the score labels in order
  my $fname_or_source = shift;
  # print STDERR "Peeking at the beginning of nbestlist to get order of scores: $fname_or_source\n";
  open IN, $fname_or_source or die "Failed to get order of scores from nbestlist '$fname_or_source'";
  my $line = <IN>;
  close IN;
  die "Line empty in nbestlist '$fname_or_source'" if !defined $line;
  my ($sent, $hypo, $scores, $total) = split /\|\|\|/, $line;
  $scores =~ s/^\s*|\s*$//g;
  die "No scores in line: $line" if $scores eq "";

  my @order = ();
  my $label = undef;
  my $sparse = 0; # we ignore sparse features here
  foreach my $tok (split /\s+/, $scores) {
    if ($tok =~ /.+_.+:/) {
      $sparse = 1;
    } elsif ($tok =~ /^([a-z][0-9a-z]*):/i) {
      $label = $1;
    } elsif ($tok =~ /^-?[-0-9.e]+$/) {
      if (!$sparse) {
        # a score found, remember it
        die "Found a score but no label before it! Bad nbestlist '$fname_or_source'!"
          if !defined $label;
        push @order, $label;
      }
      $sparse = 0;
    } else {
      die "Not a label, not a score '$tok'. Failed to parse the scores string: '$scores' of nbestlist '$fname_or_source'";
    }
  }
  print STDERR "The decoder returns the scores in this order: @order\n";
  return @order;
}


sub safesystem {
  print STDERR "Executing: @_\n";
  system(@_);
  if ($? == -1) {
      print STDERR "Failed to execute: @_\n  $!\n";
      exit(1);
  }
  elsif ($? & 127) {
      printf STDERR "Execution of: @_\n  died with signal %d, %s coredump\n",
          ($? & 127),  ($? & 128) ? 'with' : 'without';
      exit(1);
  }
  else {
    my $exitcode = $? >> 8;
    print STDERR "Exit code: $exitcode\n" if $exitcode;
    return ! $exitcode;
  }
}
sub ensure_full_path {
    my $PATH = shift;
$PATH =~ s/\/nfsmnt//;
    return $PATH if $PATH =~ /^\//;
    my $dir = `pawd 2>/dev/null`; 
    if(!$dir){$dir = `pwd`;}
    chomp($dir);
    $PATH = $dir."/".$PATH;
    $PATH =~ s/[\r\n]//g;
    $PATH =~ s/\/\.\//\//g;
    $PATH =~ s/\/+/\//g;
    my $sanity = 0;
    while($PATH =~ /\/\.\.\// && $sanity++<10) {
        $PATH =~ s/\/+/\//g;
        $PATH =~ s/\/[^\/]+\/\.\.\//\//g;
    }
    $PATH =~ s/\/[^\/]+\/\.\.$//;
    $PATH =~ s/\/+$//;
$PATH =~ s/\/nfsmnt//;
    return $PATH;
}

sub submit_or_exec {

  my $argvlen = @_;
  my $cmd = undef;
  my $stdout = undef;
  my $stderr = undef;
  my $jidfile = undef;
  my $prevjid = undef;

  # if supply 3 arguments, exec without submit
  # if supply 4 arguments, then submit new job
  # if supply 5 arguments, wait for the previous job to finish
  if ($argvlen == 3){
    ($cmd,$stdout,$stderr) = @_;
  } elsif ($argvlen == 4){
    ($cmd,$stdout,$stderr,$jidfile) = @_;
  } elsif ($argvlen == 5){
    ($cmd,$stdout,$stderr,$jidfile,$prevjid) = @_;
  }

  print STDERR "exec: $cmd\n";
  if (defined $___JOBS && $___JOBS > 0 && $argvlen==5) {
    safesystem("$qsubwrapper $pass_old_sge -command='$cmd' -queue-parameters=\"$queue_flags\" -stdout=$stdout -stderr=$stderr -jidfile=$jidfile -prevjid=$prevjid" )
      or die "ERROR: Failed to submit '$cmd' (via $qsubwrapper)";
  } 
  elsif (defined $___JOBS && $___JOBS > 0 && $argvlen==4) {
    safesystem("$qsubwrapper $pass_old_sge -command='$cmd' -queue-parameters=\"$queue_flags\" -stdout=$stdout -stderr=$stderr -jidfile=$jidfile" )
      or die "ERROR: Failed to submit '$cmd' (via $qsubwrapper)";
  } else {
    safesystem("$cmd > $stdout 2> $stderr") or die "ERROR: Failed to run '$cmd'.";
  }
}

sub exit_submit {

  my $argvlen = @_;
  my $cmd = undef;
  my $stdout = undef;
  my $stderr = undef;
  my $jidfile = undef;
  my $pidfile = undef;
  my $prevjid = undef;
  my $prevjidarraysize = 0;
  my @prevjidarray = ();
  my $pid = undef; 
  my $qsubcmd="";
  my $hj="";

  # if supply 4 arguments, then submit new job
  # if supply 5 arguments, wait for the previous job to finish
  if ($argvlen == 2) {
    ($stdout,$stderr) = @_;
  } elsif ($argvlen == 4){
    ($stdout,$stderr,$jidfile,$pidfile) = @_;
  } elsif ($argvlen == 5){
    ($stdout,$stderr,$jidfile,$pidfile,$prevjid) = @_;
  }

  # parse prevjid ########################
  $prevjid =~ s/^\s+|\s+$//g;
  @prevjidarray = split(/\s+/,$prevjid);
  $prevjidarraysize = scalar(@prevjidarray);
  ########################################


  # print STDERR "exec: $stdout\n";
  
  # read pid from file, and draft exit script ##################
  chomp ($pid=`tail -n 1 $pidfile`);
  open (OUT, ">exitjob$pid.sh");

  my $scriptheader="\#\!/bin/bash\n\#\$ -S /bin/sh\n# Both lines are needed to invoke base\n#the above line is ignored by qsub, unless parameter \"-b yes\" is set!\n\n";
  $scriptheader .="uname -a\n\n";
  $scriptheader .="cd $___WORKING_DIR\n\n";

  print OUT $scriptheader;

  print OUT "if $qsubwrapper_exit -submithost=$submithost -stdout=$stdout -stderr=$stderr -jidfile=$jidfile -pidfile=$pidfile > exitjob$pid.out 2> exitjob$pid.err ; then
    echo 'succeeded'
else
    echo failed with exit status \$\?
    die=1
fi
";
  print OUT "\n\n";

  close (OUT);
  # setting permissions of the script
  chmod(oct(755),"exitjob$pid.sh");
  ##############################################################


  if (defined $___JOBS && $___JOBS > 0 && $argvlen==5) {
    if (defined $prevjid && $prevjid!=-1 && $prevjidarraysize == 1){
      $hj = "-hold_jid $prevjid";
    } elsif (defined $prevjid && $prevjidarraysize > 1){
      $hj = "-hold_jid " . join(" -hold_jid ", @prevjidarray);
    }
    $qsubcmd="qsub $queue_flags -V $hj exitjob$pid.sh > exitjob$pid.log 2>&1";
    safesystem($qsubcmd) or die "ERROR: Failed to exit-submit $pid (via $qsubwrapper_exit)";
  } elsif (defined $___JOBS && $___JOBS > 0 && $argvlen==4) {
    $qsubcmd="qsub $queue_flags -V exitjob$pid.sh > exitjob$pid.log 2>&1";
    safesystem($qsubcmd) or die "ERROR: Failed to exit-submit $pid (via $qsubwrapper_exit)";
  } else {
    safesystem("rm $stdout") or die "ERROR: Failed to remove '$stdout'.";
    safesystem("rm $stderr") or die "ERROR: Failed to remove '$stderr'.";
  }
}
   
   





sub create_extractor_script()
{
  my ($cmd, $outdir) = @_;
  my $script_path = File::Spec->catfile($outdir, "extractor.sh");

  open my $out, '>', $script_path
      or die "Couldn't open $script_path for writing: $!\n";
  print $out "#!/bin/bash\n";
  print $out "cd $outdir\n";
  print $out "$cmd\n";
  close($out);

  `chmod +x $script_path`;

  return $script_path;
}
