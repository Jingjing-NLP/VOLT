// vim:tabstop=2

/***********************************************************************
 Moses - factored phrase-based language decoder
 Copyright (C) 2006 University of Edinburgh

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

#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include "PhraseDictionaryMemory.h"
#include "moses/FactorCollection.h"
#include "moses/Word.h"
#include "moses/Util.h"
#include "moses/InputFileStream.h"
#include "moses/StaticData.h"
#include "moses/Range.h"
#include "moses/TranslationModel/RuleTable/LoaderFactory.h"
#include "moses/TranslationModel/RuleTable/Loader.h"
#include "moses/TranslationModel/CYKPlusParser/ChartRuleLookupManagerMemory.h"
#include "moses/InputPath.h"

using namespace std;

namespace Moses
{
PhraseDictionaryMemory::PhraseDictionaryMemory(const std::string &line)
  : RuleTableTrie(line)
{
  ReadParameters();

  // caching for memory pt is pointless
  m_maxCacheSize = 0;

}

TargetPhraseCollection::shared_ptr
PhraseDictionaryMemory::
GetOrCreateTargetPhraseCollection(const Phrase &source,
                                  const TargetPhrase &target,
                                  const Word *sourceLHS)
{
  PhraseDictionaryNodeMemory &currNode = GetOrCreateNode(source, target, sourceLHS);
  return currNode.GetTargetPhraseCollection();
}

TargetPhraseCollection::shared_ptr
PhraseDictionaryMemory::
GetTargetPhraseCollectionLEGACY(const Phrase& sourceOrig) const
{
  Phrase source(sourceOrig);
  source.OnlyTheseFactors(m_inputFactors);

  // exactly like CreateTargetPhraseCollection, but don't create
  const size_t size = source.GetSize();

  const PhraseDictionaryNodeMemory *currNode = &m_collection;
  for (size_t pos = 0 ; pos < size ; ++pos) {
    const Word& word = source.GetWord(pos);
    currNode = currNode->GetChild(word);
    if (currNode == NULL)
      return TargetPhraseCollection::shared_ptr();
  }

  return currNode->GetTargetPhraseCollection();
}

PhraseDictionaryNodeMemory &PhraseDictionaryMemory::GetOrCreateNode(const Phrase &source
    , const TargetPhrase &target
    , const Word *sourceLHS)
{
  const size_t size = source.GetSize();

  const AlignmentInfo &alignmentInfo = target.GetAlignNonTerm();
  AlignmentInfo::const_iterator iterAlign = alignmentInfo.begin();

  PhraseDictionaryNodeMemory *currNode = &m_collection;
  for (size_t pos = 0 ; pos < size ; ++pos) {
    const Word& word = source.GetWord(pos);

    if (word.IsNonTerminal()) {
      // indexed by source label 1st
      const Word &sourceNonTerm = word;

      UTIL_THROW_IF2(iterAlign == alignmentInfo.end(),
                     "No alignment for non-term at position " << pos);
      UTIL_THROW_IF2(iterAlign->first != pos,
                     "Alignment info incorrect at position " << pos);

      size_t targetNonTermInd = iterAlign->second;
      ++iterAlign;
      const Word &targetNonTerm = target.GetWord(targetNonTermInd);
#if defined(UNLABELLED_SOURCE)
      currNode = currNode->GetOrCreateNonTerminalChild(targetNonTerm);
#else
      currNode = currNode->GetOrCreateChild(sourceNonTerm, targetNonTerm);
#endif
    } else {
      currNode = currNode->GetOrCreateChild(word);
    }

    UTIL_THROW_IF2(currNode == NULL,
                   "Node not found at position " << pos);
  }

  // finally, the source LHS
  //currNode = currNode->GetOrCreateChild(sourceLHS);

  return *currNode;
}

ChartRuleLookupManager *PhraseDictionaryMemory::CreateRuleLookupManager(
  const ChartParser &parser,
  const ChartCellCollectionBase &cellCollection,
  std::size_t /*maxChartSpan */)
{
  return new ChartRuleLookupManagerMemory(parser, cellCollection, *this);
}

void PhraseDictionaryMemory::SortAndPrune()
{
  if (GetTableLimit()) {
    m_collection.Sort(GetTableLimit());
  }
}

void
PhraseDictionaryMemory::
GetTargetPhraseCollectionBatch(const InputPathList &inputPathQueue) const
{
  InputPathList::const_iterator iter;
  for (iter = inputPathQueue.begin(); iter != inputPathQueue.end(); ++iter) {
    InputPath &inputPath = **iter;
    const Phrase &phrase = inputPath.GetPhrase();
    const InputPath *prevPath = inputPath.GetPrevPath();

    const PhraseDictionaryNodeMemory *prevPtNode = NULL;

    if (prevPath) {
      prevPtNode = static_cast<const PhraseDictionaryNodeMemory*>(prevPath->GetPtNode(*this));
    } else {
      // Starting subphrase.
      assert(phrase.GetSize() == 1);
      prevPtNode = &GetRootNode();
    }

    // backoff
    if (!SatisfyBackoff(inputPath)) {
      continue;
    }

    if (prevPtNode) {
      Word lastWord = phrase.GetWord(phrase.GetSize() - 1);
      lastWord.OnlyTheseFactors(m_inputFactors);

      const PhraseDictionaryNodeMemory *ptNode = prevPtNode->GetChild(lastWord);
      TargetPhraseCollection::shared_ptr targetPhrases;
      if (ptNode) {
        targetPhrases = ptNode->GetTargetPhraseCollection();
      }
      inputPath.SetTargetPhrases(*this, targetPhrases, ptNode);
    }
  }
}

TO_STRING_BODY(PhraseDictionaryMemory);

// friend
ostream& operator<<(ostream& out, const PhraseDictionaryMemory& phraseDict)
{
  typedef PhraseDictionaryNodeMemory::TerminalMap TermMap;
  typedef PhraseDictionaryNodeMemory::NonTerminalMap NonTermMap;

  const PhraseDictionaryNodeMemory &coll = phraseDict.m_collection;
  for (NonTermMap::const_iterator p = coll.m_nonTermMap.begin(); p != coll.m_nonTermMap.end(); ++p) {
#if defined(UNLABELLED_SOURCE)
    const Word &targetNonTerm = p->first;
    out << targetNonTerm;
#else
    const Word &sourceNonTerm = p->first.first;
    out << sourceNonTerm;
#endif
  }
  for (TermMap::const_iterator p = coll.m_sourceTermMap.begin(); p != coll.m_sourceTermMap.end(); ++p) {
    const Word &sourceTerm = p->first;
    out << sourceTerm;
  }
  return out;
}

}
