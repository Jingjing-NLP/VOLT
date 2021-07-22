/***********************************************************************
 Moses - statistical machine translation system
 Copyright (C) 2006-2011 University of Edinburgh

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include "LoaderStandard.h"

#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <sys/stat.h>
#include <cstdlib>
#include <boost/algorithm/string/predicate.hpp>
#include "Trie.h"
#include "moses/FactorCollection.h"
#include "moses/Word.h"
#include "moses/Util.h"
#include "moses/InputFileStream.h"
#include "moses/StaticData.h"
#include "moses/Range.h"
#include "moses/ChartTranslationOptionList.h"
#include "moses/FactorCollection.h"
#include "util/file_piece.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"
#include "util/double-conversion/double-conversion.h"
#include "util/exception.hh"

using namespace std;
using namespace boost::algorithm;

namespace Moses
{

bool
RuleTableLoaderStandard::
Load(AllOptions const& opts
     , const std::vector<FactorType> &input
     , const std::vector<FactorType> &output
     , const std::string &inFile
     , size_t tableLimit
     , RuleTableTrie &ruleTable)
{
  return Load(opts, MosesFormat,input, output ,inFile ,tableLimit ,ruleTable);
}

void ReformatHieroRule(int sourceTarget, string &phrase, map<size_t, pair<size_t, size_t> > &ntAlign)
{
  vector<string> toks;
  Tokenize(toks, phrase, " ");

  for (size_t i = 0; i < toks.size(); ++i) {
    string &tok = toks[i];
    if (starts_with(tok, "[") && ends_with(tok, "]")) {
      // no-term
      vector<string> split = Tokenize(tok, ",");
      UTIL_THROW_IF2(split.size() != 2,
                     "Incorrectly formmatted non-terminal: " << tok);

      tok = "[X]" + split[0] + "]";
      size_t coIndex = Scan<size_t>(split[1]);

      pair<size_t, size_t> &alignPoint = ntAlign[coIndex];
      if (sourceTarget == 0) {
        alignPoint.first = i;
      } else {
        alignPoint.second = i;
      }
    }
  }

  phrase = Join(" ", toks) + " [X]";

}

void ReformateHieroScore(string &scoreString)
{
  vector<string> toks;
  Tokenize(toks, scoreString, " ");

  for (size_t i = 0; i < toks.size(); ++i) {
    string &tok = toks[i];
    vector<string> nameValue = Tokenize(tok, "=");
    UTIL_THROW_IF2(nameValue.size() != 2,
                   "Incorrectly formatted score: " << tok);

    float score = Scan<float>(nameValue[1]);
    score = exp(-score);
    tok = SPrint(score);
  }

  scoreString = Join(" ", toks);
}

void ReformatHieroRule(const string &lineOrig, string &out)
{
  vector<string> tokens;
  vector<float> scoreVector;

  TokenizeMultiCharSeparator(tokens, lineOrig, "|||" );

  string &sourcePhraseString = tokens[1]
                               , &targetPhraseString = tokens[2]
                                   , &scoreString        = tokens[3];

  map<size_t, pair<size_t, size_t> > ntAlign;
  ReformatHieroRule(0, sourcePhraseString, ntAlign);
  ReformatHieroRule(1, targetPhraseString, ntAlign);
  ReformateHieroScore(scoreString);

  util::StringStream align;
  map<size_t, pair<size_t, size_t> >::const_iterator iterAlign;
  for (iterAlign = ntAlign.begin(); iterAlign != ntAlign.end(); ++iterAlign) {
    const pair<size_t, size_t> &alignPoint = iterAlign->second;
    align << alignPoint.first << "-" << alignPoint.second << " ";
  }

  util::StringStream ret;
  ret << sourcePhraseString << " ||| "
      << targetPhraseString << " ||| "
      << scoreString << " ||| "
      << align.str();

  out = ret.str();
}

bool RuleTableLoaderStandard::Load(AllOptions const& opts, FormatType format
                                   , const std::vector<FactorType> &input
                                   , const std::vector<FactorType> &output
                                   , const std::string &inFile
                                   , size_t /* tableLimit */
                                   , RuleTableTrie &ruleTable)
{
  PrintUserTime(string("Start loading text phrase table. ") + (format==MosesFormat?"Moses":"Hiero") + " format");

  // const StaticData &staticData = StaticData::Instance();

  string lineOrig;
  size_t count = 0;

  std::ostream *progress = NULL;
  IFVERBOSE(1) progress = &std::cerr;
  util::FilePiece in(inFile.c_str(), progress);

  // reused variables
  vector<float> scoreVector;
  StringPiece line;
  std::string hiero_before, hiero_after;

  double_conversion::StringToDoubleConverter converter(double_conversion::StringToDoubleConverter::NO_FLAGS, NAN, NAN, "inf", "nan");

  while(true) {
    try {
      line = in.ReadLine();
    } catch (const util::EndOfFileException &e) {
      break;
    }

    if (format == HieroFormat) { // inefficiently reformat line
      hiero_before.assign(line.data(), line.size());
      ReformatHieroRule(hiero_before, hiero_after);
      line = hiero_after;
    }

    util::TokenIter<util::MultiCharacter> pipes(line, "|||");
    StringPiece sourcePhraseString(*pipes);
    StringPiece targetPhraseString(*++pipes);
    StringPiece scoreString(*++pipes);

    StringPiece alignString;
    if (++pipes) {
      StringPiece temp(*pipes);
      alignString = temp;
    }

    bool isLHSEmpty = (sourcePhraseString.find_first_not_of(" \t", 0) == string::npos);
    if (isLHSEmpty && !opts.unk.word_deletion_enabled) {
      TRACE_ERR( ruleTable.GetFilePath() << ":" << count << ": pt entry contains empty target, skipping\n");
      continue;
    }

    scoreVector.clear();
    for (util::TokenIter<util::AnyCharacter, true> s(scoreString, " \t"); s; ++s) {
      int processed;
      float score = converter.StringToFloat(s->data(), s->length(), &processed);
      UTIL_THROW_IF2(isnan(score), "Bad score " << *s << " on line " << count);
      scoreVector.push_back(FloorScore(TransformScore(score)));
    }
    const size_t numScoreComponents = ruleTable.GetNumScoreComponents();
    if (scoreVector.size() != numScoreComponents) {
      UTIL_THROW2("Size of scoreVector != number (" << scoreVector.size() << "!="
                  << numScoreComponents << ") of score components on line " << count);
    }

    // parse source & find pt node

    // constituent labels
    Word *sourceLHS = NULL;
    Word *targetLHS;

    // create target phrase obj
    TargetPhrase *targetPhrase = new TargetPhrase(&ruleTable);
    targetPhrase->CreateFromString(Output, output, targetPhraseString, &targetLHS);
    // source
    Phrase sourcePhrase;
    sourcePhrase.CreateFromString(Input, input, sourcePhraseString, &sourceLHS);

    // rest of target phrase
    targetPhrase->SetAlignmentInfo(alignString);
    targetPhrase->SetTargetLHS(targetLHS);

    ++pipes;  // skip over counts field

    if (++pipes) {
      StringPiece sparseString(*pipes);
      targetPhrase->SetSparseScore(&ruleTable, sparseString);
    }

    if (++pipes) {
      StringPiece propertiesString(*pipes);
      targetPhrase->SetProperties(propertiesString);
    }

    targetPhrase->GetScoreBreakdown().Assign(&ruleTable, scoreVector);
    targetPhrase->EvaluateInIsolation(sourcePhrase, ruleTable.GetFeaturesToApply());

    TargetPhraseCollection::shared_ptr phraseColl
    = GetOrCreateTargetPhraseCollection(ruleTable, sourcePhrase,
                                        *targetPhrase, sourceLHS);
    phraseColl->Add(targetPhrase);

    // not implemented correctly in memory pt. just delete it for now
    delete sourceLHS;

    count++;
  }

  // sort and prune each target phrase collection
  SortAndPrune(ruleTable);

  return true;
}

}
