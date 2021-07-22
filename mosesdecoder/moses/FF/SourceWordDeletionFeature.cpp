#include <sstream>
#include "SourceWordDeletionFeature.h"
#include "moses/Phrase.h"
#include "moses/TargetPhrase.h"
#include "moses/Hypothesis.h"
#include "moses/ChartHypothesis.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/TranslationOption.h"
#include "moses/Util.h"

#include "util/string_piece_hash.hh"
#include "util/exception.hh"

namespace Moses
{

using namespace std;

SourceWordDeletionFeature::SourceWordDeletionFeature(const std::string &line)
  :StatelessFeatureFunction(0, line),
   m_unrestricted(true)
{
  VERBOSE(1, "Initializing feature " << GetScoreProducerDescription() << " ...");
  ReadParameters();
  VERBOSE(1, " Done." << std::endl);
}

void SourceWordDeletionFeature::SetParameter(const std::string& key, const std::string& value)
{
  if (key == "factor") {
    m_factorType = Scan<FactorType>(value);
  } else if (key == "path") {
    m_filename = value;
  } else {
    StatelessFeatureFunction::SetParameter(key, value);
  }
}

void SourceWordDeletionFeature::Load(AllOptions::ptr const& opts)
{
  m_options = opts;
  if (m_filename.empty())
    return;

  FEATUREVERBOSE(1, "Loading source word deletion word list from " << m_filename << std::endl);
  ifstream inFile(m_filename.c_str());
  UTIL_THROW_IF2(!inFile, "Can't open file " << m_filename);

  std::string line;
  while (getline(inFile, line)) {
    m_vocab.insert(line);
  }

  inFile.close();

  m_unrestricted = false;
}

bool SourceWordDeletionFeature::IsUseable(const FactorMask &mask) const
{
  bool ret = mask[m_factorType];
  return ret;
}

void SourceWordDeletionFeature::EvaluateInIsolation(const Phrase &source
    , const TargetPhrase &targetPhrase
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection &estimatedScores) const
{
  const AlignmentInfo &alignmentInfo = targetPhrase.GetAlignTerm();
  ComputeFeatures(source, targetPhrase, &scoreBreakdown, alignmentInfo);
}

void SourceWordDeletionFeature::ComputeFeatures(const Phrase &source,
    const TargetPhrase& targetPhrase,
    ScoreComponentCollection* accumulator,
    const AlignmentInfo &alignmentInfo) const
{
  // handle special case: unknown words (they have no word alignment)
  size_t targetLength = targetPhrase.GetSize();
  size_t sourceLength = source.GetSize();
  if (targetLength == 1 && sourceLength == 1 && !alignmentInfo.GetSize()) return;

  // flag aligned words
  std::vector<bool> aligned(sourceLength, false);
  for (AlignmentInfo::const_iterator alignmentPoint = alignmentInfo.begin(); alignmentPoint != alignmentInfo.end(); alignmentPoint++)
    aligned[ alignmentPoint->first ] = true;

  // process unaligned source words
  for(size_t i=0; i<sourceLength; i++) {
    if (!aligned[i]) {
      const Word &w = source.GetWord(i);
      if (!w.IsNonTerminal()) {
        const StringPiece word = w.GetFactor(m_factorType)->GetString();
        if (word != "<s>" && word != "</s>") {
          if (!m_unrestricted && FindStringPiece(m_vocab, word ) == m_vocab.end()) {
            accumulator->PlusEquals(this, StringPiece("OTHER"),1);
          } else {
            accumulator->PlusEquals(this,word,1);
          }
        }
      }
    }
  }
}

}
