// $Id$
// vim:tabstop=2
/***********************************************************************
 Moses - factored phrase-based language decoder
 Copyright (C) 2010 Hieu Hoang

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

#include "ChartParser.h"
#include "ChartParserCallback.h"
#include "ChartRuleLookupManager.h"
#include "StaticData.h"
#include "TreeInput.h"
#include "Sentence.h"
#include "DecodeGraph.h"
#include "moses/FF/UnknownWordPenaltyProducer.h"
#include "moses/TranslationModel/PhraseDictionary.h"
#include "moses/TranslationTask.h"

using namespace std;
using namespace Moses;

namespace Moses
{

ChartParserUnknown
::ChartParserUnknown(ttasksptr const& ttask)
  : m_ttask(ttask)
{ }

ChartParserUnknown::~ChartParserUnknown()
{
  RemoveAllInColl(m_unksrcs);
}

AllOptions::ptr const&
ChartParserUnknown::
options() const
{
  return m_ttask.lock()->options();
}

void
ChartParserUnknown::
Process(const Word &sourceWord, const Range &range, ChartParserCallback &to)
{
  // unknown word, add as trans opt
  const StaticData &staticData = StaticData::Instance();
  const UnknownWordPenaltyProducer &unknownWordPenaltyProducer
  = UnknownWordPenaltyProducer::Instance();

  size_t isDigit = 0;
  if (options()->unk.drop) {
    const Factor *f = sourceWord[0]; // TODO hack. shouldn't know which factor is surface
    const StringPiece s = f->GetString();
    isDigit = s.find_first_of("0123456789");
    if (isDigit == string::npos)
      isDigit = 0;
    else
      isDigit = 1;
    // modify the starting bitmap
  }

  Phrase* unksrc = new Phrase(1);
  unksrc->AddWord() = sourceWord;
  Word &newWord = unksrc->GetWord(0);
  newWord.SetIsOOV(true);

  m_unksrcs.push_back(unksrc);

  // hack. Once the OOV FF is a phrase table, get rid of this
  PhraseDictionary *firstPt = NULL;
  if (PhraseDictionary::GetColl().size() == 0) {
    firstPt = PhraseDictionary::GetColl()[0];
  }

  //TranslationOption *transOpt;
  if (! options()->unk.drop || isDigit) {
    // loop
    const UnknownLHSList &lhsList = options()->syntax.unknown_lhs; // staticData.GetUnknownLHS();
    UnknownLHSList::const_iterator iterLHS;
    for (iterLHS = lhsList.begin(); iterLHS != lhsList.end(); ++iterLHS) {
      const string &targetLHSStr = iterLHS->first;
      float prob = iterLHS->second;

      // lhs
      //const Word &sourceLHS = staticData.GetInputDefaultNonTerminal();
      Word *targetLHS = new Word(true);

      targetLHS->CreateFromString(Output, options()->output.factor_order,
                                  targetLHSStr, true);
      UTIL_THROW_IF2(targetLHS->GetFactor(0) == NULL, "Null factor for target LHS");

      // add to dictionary
      TargetPhrase *targetPhrase = new TargetPhrase(firstPt);
      Word &targetWord = targetPhrase->AddWord();
      targetWord.CreateUnknownWord(sourceWord);

      // scores
      float unknownScore = FloorScore(TransformScore(prob));

      targetPhrase->GetScoreBreakdown().Assign(&unknownWordPenaltyProducer, unknownScore);
      targetPhrase->SetTargetLHS(targetLHS);
      targetPhrase->SetAlignmentInfo("0-0");
      targetPhrase->EvaluateInIsolation(*unksrc);

      if (!options()->output.detailed_tree_transrep_filepath.empty() ||
          options()->nbest.print_trees || staticData.GetTreeStructure() != NULL) {
        std::string prop = "[ ";
        prop += (*targetLHS)[0]->GetString().as_string() + " ";
        prop += sourceWord[0]->GetString().as_string() + " ]";
        targetPhrase->SetProperty("Tree", prop);
      }

      // chart rule
      to.AddPhraseOOV(*targetPhrase, m_cacheTargetPhraseCollection, range);
    } // for (iterLHS
  } else {
    // drop source word. create blank trans opt
    float unknownScore = FloorScore(-numeric_limits<float>::infinity());

    TargetPhrase *targetPhrase = new TargetPhrase(firstPt);
    // loop
    const UnknownLHSList &lhsList = options()->syntax.unknown_lhs;//staticData.GetUnknownLHS();
    UnknownLHSList::const_iterator iterLHS;
    for (iterLHS = lhsList.begin(); iterLHS != lhsList.end(); ++iterLHS) {
      const string &targetLHSStr = iterLHS->first;
      //float prob = iterLHS->second;

      Word *targetLHS = new Word(true);
      targetLHS->CreateFromString(Output, staticData.options()->output.factor_order,
                                  targetLHSStr, true);
      UTIL_THROW_IF2(targetLHS->GetFactor(0) == NULL, "Null factor for target LHS");

      targetPhrase->GetScoreBreakdown().Assign(&unknownWordPenaltyProducer, unknownScore);
      targetPhrase->EvaluateInIsolation(*unksrc);

      targetPhrase->SetTargetLHS(targetLHS);

      // chart rule
      to.AddPhraseOOV(*targetPhrase, m_cacheTargetPhraseCollection, range);
    }
  }
}

ChartParser
::ChartParser(ttasksptr const& ttask, ChartCellCollectionBase &cells)
  : m_ttask(ttask)
  , m_unknown(ttask)
  , m_decodeGraphList(StaticData::Instance().GetDecodeGraphs())
  , m_source(*(ttask->GetSource().get()))
{
  const StaticData &staticData = StaticData::Instance();

  staticData.InitializeForInput(ttask);
  CreateInputPaths(m_source);

  const std::vector<PhraseDictionary*> &dictionaries = PhraseDictionary::GetColl();
  assert(dictionaries.size() == m_decodeGraphList.size());
  m_ruleLookupManagers.reserve(dictionaries.size());
  for (std::size_t i = 0; i < dictionaries.size(); ++i) {
    const PhraseDictionary *dict = dictionaries[i];
    PhraseDictionary *nonConstDict = const_cast<PhraseDictionary*>(dict);
    std::size_t maxChartSpan = m_decodeGraphList[i]->GetMaxChartSpan();
    ChartRuleLookupManager *lookupMgr = nonConstDict->CreateRuleLookupManager(*this, cells, maxChartSpan);
    m_ruleLookupManagers.push_back(lookupMgr);
  }

}

ChartParser::~ChartParser()
{
  RemoveAllInColl(m_ruleLookupManagers);
  StaticData::Instance().CleanUpAfterSentenceProcessing(m_ttask.lock());

  InputPathMatrix::const_iterator iterOuter;
  for (iterOuter = m_inputPathMatrix.begin(); iterOuter != m_inputPathMatrix.end(); ++iterOuter) {
    const std::vector<InputPath*> &outer = *iterOuter;

    std::vector<InputPath*>::const_iterator iterInner;
    for (iterInner = outer.begin(); iterInner != outer.end(); ++iterInner) {
      InputPath *path = *iterInner;
      delete path;
    }
  }
}

void ChartParser::Create(const Range &range, ChartParserCallback &to)
{
  assert(m_decodeGraphList.size() == m_ruleLookupManagers.size());

  std::vector <DecodeGraph*>::const_iterator iterDecodeGraph;
  std::vector <ChartRuleLookupManager*>::const_iterator iterRuleLookupManagers = m_ruleLookupManagers.begin();
  for (iterDecodeGraph = m_decodeGraphList.begin(); iterDecodeGraph != m_decodeGraphList.end(); ++iterDecodeGraph, ++iterRuleLookupManagers) {
    const DecodeGraph &decodeGraph = **iterDecodeGraph;
    assert(decodeGraph.GetSize() == 1);
    ChartRuleLookupManager &ruleLookupManager = **iterRuleLookupManagers;
    size_t maxSpan = decodeGraph.GetMaxChartSpan();
    size_t last = m_source.GetSize()-1;
    if (maxSpan != 0) {
      last = min(last, range.GetStartPos()+maxSpan);
    }
    if (maxSpan == 0 || range.GetNumWordsCovered() <= maxSpan) {
      const InputPath &inputPath = GetInputPath(range);
      ruleLookupManager.GetChartRuleCollection(inputPath, last, to);
    }
  }

  if (range.GetNumWordsCovered() == 1
      && range.GetStartPos() != 0
      && range.GetStartPos() != m_source.GetSize()-1) {
    bool always = options()->unk.always_create_direct_transopt;
    if (to.Empty() || always) {
      // create unknown words for 1 word coverage where we don't have any trans options
      const Word &sourceWord = m_source.GetWord(range.GetStartPos());
      m_unknown.Process(sourceWord, range, to);
    }
  }
}

void ChartParser::CreateInputPaths(const InputType &input)
{
  size_t size = input.GetSize();
  m_inputPathMatrix.resize(size);

  UTIL_THROW_IF2(input.GetType() != SentenceInput && input.GetType() != TreeInputType,
                 "Input must be a sentence or a tree, " <<
                 "not lattice or confusion networks");

  TranslationTask const* ttask = m_ttask.lock().get();
  for (size_t phaseSize = 1; phaseSize <= size; ++phaseSize) {
    for (size_t startPos = 0; startPos < size - phaseSize + 1; ++startPos) {
      size_t endPos = startPos + phaseSize -1;
      vector<InputPath*> &vec = m_inputPathMatrix[startPos];

      Range range(startPos, endPos);
      Phrase subphrase(input.GetSubString(Range(startPos, endPos)));
      const NonTerminalSet &labels = input.GetLabelSet(startPos, endPos);

      InputPath *node;
      if (range.GetNumWordsCovered() == 1) {
        node = new InputPath(ttask, subphrase, labels, range, NULL, NULL);
        vec.push_back(node);
      } else {
        const InputPath &prevNode = GetInputPath(startPos, endPos - 1);
        node = new InputPath(ttask, subphrase, labels, range, &prevNode, NULL);
        vec.push_back(node);
      }

      //m_inputPathQueue.push_back(node);
    }
  }
}

const InputPath &ChartParser::GetInputPath(const Range &range) const
{
  return GetInputPath(range.GetStartPos(), range.GetEndPos());
}

const InputPath &ChartParser::GetInputPath(size_t startPos, size_t endPos) const
{
  size_t offset = endPos - startPos;
  UTIL_THROW_IF2(offset >= m_inputPathMatrix[startPos].size(),
                 "Out of bound: " << offset);
  return *m_inputPathMatrix[startPos][offset];
}

InputPath &ChartParser::GetInputPath(size_t startPos, size_t endPos)
{
  size_t offset = endPos - startPos;
  UTIL_THROW_IF2(offset >= m_inputPathMatrix[startPos].size(),
                 "Out of bound: " << offset);
  return *m_inputPathMatrix[startPos][offset];
}
/*
const Sentence &ChartParser::GetSentence() const {
  const Sentence &sentence = static_cast<const Sentence&>(m_source);
  return sentence;
}
*/
size_t ChartParser::GetSize() const
{
  return m_source.GetSize();
}

long ChartParser::GetTranslationId() const
{
  return m_source.GetTranslationId();
}


AllOptions::ptr const&
ChartParser::
options() const
{
  return m_ttask.lock()->options();
}


} // namespace Moses
