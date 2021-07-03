<?php

/*
This file is part of moses.  Its use is licensed under the GNU Lesser General
Public License version 2.1 or, at your option, any later version.
*/

require("lib.php");
require("overview.php");
require("analysis.php");
require("analysis_diff.php");
require("diff.php");
require("sgviz.php");

function head($title) {
  print '<!DOCTYPE html>
<html><head><title>'.$title.'</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<script language="javascript" src="javascripts/prototype.js"></script>
<script language="javascript" src="javascripts/scriptaculous.js"></script>
<script language="javascript" src="hierarchical-segmentation.js"></script>
<script language="javascript" src="base64.js"></script>
<link href="general.css" rel="stylesheet" type="text/css">
<link href="hierarchical-segmentation.css" rel="stylesheet" type="text/css">
<link href="bilingual-concordance.css" rel="stylesheet" type="text/css">
</head>
<body><h2>'.$title."</h2>\n";
}

if (array_key_exists("setStepStatus",$_GET)) { set_step_status($_GET["setStepStatus"]); }
else if (array_key_exists("setup",$_POST) || array_key_exists("setup",$_GET)) {
  load_experiment_info();
  load_comment();

  if (array_key_exists("show",$_GET)) { show(); }
  else if (array_key_exists("diff",$_GET)) { diff(); }
  else if (array_key_exists("analysis",$_GET)) {
    $action = $_GET["analysis"];
    $set = $_GET["set"];
    $id = $_GET["id"];
    if (array_key_exists("id2",$_GET)) { $id2 = $_GET["id2"]; }
    if ($action == "show") { show_analysis(); }
    else if ($action == "bleu_show") { bleu_show(); }
    else if ($action == "ngram_precision_show") { ngram_show("precision");}
    else if ($action == "ngram_recall_show") { ngram_show("recall"); }
    else if ($action == "CoverageSummary_show") { coverage_summary(); }
    else if ($action == "PrecisionRecallDetails_show") { precision_recall_details(); }
    else if ($action == "PrecisionRecallDetailsDiff_show") { precision_recall_details_diff(); }
    else if ($action == "PrecisionByCoverage_show") { precision_by_coverage(); }
    else if ($action == "PrecisionByCoverageDiff_show") { precision_by_coverage_diff(); }
    else if (preg_match("/PrecisionByWordDiff(.+)_show/",$action,$match)) { precision_by_word_diff($match[1]); }
    else if (preg_match("/PrecisionByWord(.+)_show/",$action,$match)) { precision_by_word($match[1]); }
    else if ($action == "CoverageDetails_show") { coverage_details(); }
    else if ($action == "CoverageMatrixDetails_show") { precision_by_coverage_diff_matrix_details(); }
    else if ($action == "SegmentationSummary_show") { segmentation_summary(); }
    else if ($action == "biconcor") { biconcor(base64_decode($_GET["phrase"])); }
    else if ($action == "sgviz") { sgviz($_GET["sentence"]); }
    else if ($action == "sgviz_data") { sgviz_data($_GET["sentence"]); }
    else { print "ERROR! $action"; }
  }
  else if (array_key_exists("analysis_diff_home",$_GET)) {
    $set = $_GET["analysis_diff_home"];
    while (list($parameter,$value) = each($_GET)) {
      if (preg_match("/analysis\-(\d+)\-(.+)/",$parameter,$match)) {
        if ($match[2] == $set) {
          $id_array[] = $match[1];
        }
      }
    }
    if (count($id_array) != 2) {
      print "ERROR: comp 2!";
      exit();
    }
    $id = $id_array[0];
    $id2 = $id_array[1];
    if ($id>$id2) { $i=$id; $id=$id2; $id2=$i; }
    diff_analysis();
  }
  else if (array_key_exists("analysis_diff",$_GET)) {
    $action = $_GET["analysis_diff"];
    $set = $_GET["set"];
    $id = $_GET["id"];
    $id2 = $_GET["id2"];
    if ($action == "bleu_diff") { bleu_diff(); }
    else if ($action == "ngram_precision_diff") { ngram_diff("precision");}
    else if ($action == "ngram_recall_diff") { ngram_diff("recall"); }
    else { print "ERROR! $action"; }
  }
  else { overview(); }
}
else {
  setup();
}

print "</BODY></HTML>\n";
