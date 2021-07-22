#include "InputPath.h"
#include "ScoreComponentCollection.h"
#include "TargetPhraseCollection.h"
#include "StaticData.h"
#include "TypeDef.h"
#include "AlignmentInfo.h"
#include "util/exception.hh"
#include "TranslationModel/PhraseDictionary.h"
using namespace std;

namespace Moses
{
InputPath::
InputPath(TranslationTask const* theTask,
          Phrase const& phrase,
          NonTerminalSet const& sourceNonTerms,
          Range const& range, InputPath const *prevNode,
          const ScorePair *inputScore)
  : ttask(theTask)
  , m_prevPath(prevNode)
  , m_phrase(phrase)
  , m_range(range)
  , m_inputScore(inputScore)
  , m_nextNode(1)
  , m_sourceNonTerms(sourceNonTerms)
  , m_sourceNonTermArray(FactorCollection::Instance().GetNumNonTerminals(), false)
{
  for (NonTerminalSet::const_iterator iter = sourceNonTerms.begin(); iter != sourceNonTerms.end(); ++iter) {
    size_t idx = (*iter)[0]->GetId();
    m_sourceNonTermArray[idx] = true;
  }

  //cerr << "phrase=" << phrase << " m_inputScore=" << *m_inputScore << endl;

}

InputPath::~InputPath()
{

  // std::cerr << "Deconstructing InputPath" << std::endl;


  // // NOT NEEDED ANY MORE SINCE THE SWITCH TO SHARED POINTERS
  // // Since there is no way for the Phrase Dictionaries to tell in
  // // which (sentence) context phrases were looked up, we tell them
  // // now that the phrase isn't needed any more by this inputPath
  // typedef std::pair<boost::shared_ptr<TargetPhraseCollection>, const void* > entry;
  // std::map<const PhraseDictionary*, entry>::iterator iter;
  // ttasksptr theTask = this->ttask.lock();
  // for (iter = m_targetPhrases.begin(); iter != m_targetPhrases.end(); ++iter)
  //   {
  //     // std::cerr << iter->second.first << " decommissioned." << std::endl;
  //     iter->first->Release(theTask, iter->second.first);
  //   }

  delete m_inputScore;
}

TargetPhraseCollection::shared_ptr
InputPath::
GetTargetPhrases(const PhraseDictionary &phraseDictionary) const
{
  TargetPhrases::const_iterator iter;
  iter = m_targetPhrases.find(&phraseDictionary);
  if (iter == m_targetPhrases.end()) {
    return TargetPhraseCollection::shared_ptr();
  }
  return iter->second.first;
}

const void*
InputPath::
GetPtNode(const PhraseDictionary &phraseDictionary) const
{
  TargetPhrases::const_iterator iter;
  iter = m_targetPhrases.find(&phraseDictionary);
  if (iter == m_targetPhrases.end()) {
    return NULL;
  }
  return iter->second.second;
}

void
InputPath::
SetTargetPhrases(const PhraseDictionary &phraseDictionary,
                 TargetPhraseCollection::shared_ptr const& targetPhrases,
                 const void *ptNode)
{
  std::pair<TargetPhraseCollection::shared_ptr, const void*>
  value(targetPhrases, ptNode);
  m_targetPhrases[&phraseDictionary] = value;
}

const Word &InputPath::GetLastWord() const
{
  size_t len = m_phrase.GetSize();
  UTIL_THROW_IF2(len == 0, "Input path phrase cannot be empty");
  const Word &ret = m_phrase.GetWord(len - 1);
  return ret;
}

size_t InputPath::GetTotalRuleSize() const
{
  size_t ret = 0;
  TargetPhrases::const_iterator iter;
  for (iter = m_targetPhrases.begin(); iter != m_targetPhrases.end(); ++iter) {
    // const PhraseDictionary *pt = iter->first;
    TargetPhraseCollection::shared_ptr tpColl = iter->second.first;

    if (tpColl) {
      ret += tpColl->GetSize();
    }
  }

  return ret;
}

std::ostream& operator<<(std::ostream& out, const InputPath& obj)
{
  out << &obj << " " << obj.GetWordsRange() << " " << obj.GetPrevPath() << " " << obj.GetPhrase();

  InputPath::TargetPhrases::const_iterator iter;
  for (iter = obj.m_targetPhrases.begin(); iter != obj.m_targetPhrases.end(); ++iter) {
    const PhraseDictionary *pt = iter->first;
    boost::shared_ptr<TargetPhraseCollection const> tpColl = iter->second.first;

    out << pt << "=";
    if (tpColl) {
      cerr << tpColl->GetSize() << " ";
    } else {
      cerr << "NULL ";
    }
  }

  return out;
}

}
