// $Id$
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

#include "TranslationOption.h"
#include "Bitmap.h"
#include "GenerationDictionary.h"
#include "StaticData.h"
#include "InputType.h"
#include "moses/FF/LexicalReordering/LexicalReordering.h"

using namespace std;

namespace Moses
{

TranslationOption::TranslationOption()
  :m_targetPhrase(NULL)
  ,m_inputPath(NULL)
  ,m_sourceWordsRange(NOT_FOUND, NOT_FOUND)
{ }

//TODO this should be a factory function!
TranslationOption::TranslationOption(const Range &range
                                     , const TargetPhrase &targetPhrase)
  : m_targetPhrase(targetPhrase)
  , m_inputPath(NULL)
  , m_sourceWordsRange(range)
  , m_futureScore(targetPhrase.GetFutureScore())
{
}

bool TranslationOption::IsCompatible(const Phrase& phrase, const std::vector<FactorType>& featuresToCheck) const
{
  if (featuresToCheck.size() == 1) {
    return m_targetPhrase.IsCompatible(phrase, featuresToCheck[0]);
  } else if (featuresToCheck.empty()) {
    return true;
    /* features already there, just update score */
  } else {
    return m_targetPhrase.IsCompatible(phrase, featuresToCheck);
  }
}

bool TranslationOption::Overlap(const Hypothesis &hypothesis) const
{
  const Bitmap &bitmap = hypothesis.GetWordsBitmap();
  return bitmap.Overlap(GetSourceWordsRange());
}

void
TranslationOption::
CacheLexReorderingScores(const LexicalReordering &producer, const Scores &score)
{
  if (score.empty()) return;
  boost::shared_ptr<Scores> stored(new Scores(score));
  m_targetPhrase.SetExtraScores(&producer,stored);
  // m_lexReorderingScores[&producer] = score;
}

void TranslationOption::EvaluateWithSourceContext(const InputType &input)
{
  const InputPath &inputPath = GetInputPath();
  m_targetPhrase.EvaluateWithSourceContext(input, inputPath);
}

const InputPath &TranslationOption::GetInputPath() const
{
  UTIL_THROW_IF2(m_inputPath == NULL,
                 "No input path");
  return *m_inputPath;
}

void TranslationOption::SetInputPath(const InputPath &inputPath)
{
  UTIL_THROW_IF2(m_inputPath,
                 "Input path already specified");
  m_inputPath = &inputPath;
}


TO_STRING_BODY(TranslationOption);

// friend
ostream& operator<<(ostream& out, const TranslationOption& possibleTranslation)
{
  out << possibleTranslation.GetTargetPhrase()
      << " c=" << possibleTranslation.GetFutureScore()
      << " [" << possibleTranslation.GetSourceWordsRange() << "]"
      << possibleTranslation.GetScoreBreakdown();
  return out;
}

/** returns cached scores */
const Scores*
TranslationOption::
GetLexReorderingScores(LexicalReordering const* scoreProducer) const
{
  return m_targetPhrase.GetExtraScores(scoreProducer);
  // _ScoreCacheMap::const_iterator it;
  // it = m_lexReorderingScores.find(scoreProducer);
  // if(it == m_lexReorderingScores.end())
  //   return NULL;
  // else
  //   return &(it->second);
}

}


