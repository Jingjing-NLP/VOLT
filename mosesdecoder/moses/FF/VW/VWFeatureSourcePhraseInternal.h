#pragma once

#include <string>
#include <algorithm>
#include "VWFeatureSource.h"
#include "moses/Util.h"

namespace Moses
{

class VWFeatureSourcePhraseInternal : public VWFeatureSource
{
public:
  VWFeatureSourcePhraseInternal(const std::string &line)
    : VWFeatureSource(line) {
    ReadParameters();

    // Call this last
    VWFeatureBase::UpdateRegister();
  }

  void operator()(const InputType &input
                  , const Range &sourceRange
                  , Discriminative::Classifier &classifier
                  , Discriminative::FeatureVector &outFeatures) const {
    size_t begin = sourceRange.GetStartPos();
    size_t end   = sourceRange.GetEndPos() + 1;

    while (begin < end) {
      outFeatures.push_back(classifier.AddLabelIndependentFeature("sin^" + GetWord(input, begin++)));
    }
  }

  virtual void SetParameter(const std::string& key, const std::string& value) {
    VWFeatureSource::SetParameter(key, value);
  }
};

}
