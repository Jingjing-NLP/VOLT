// $Id$
#ifndef moses_TranslationOptionCollectionConfusionNet_h
#define moses_TranslationOptionCollectionConfusionNet_h

#include "TranslationOptionCollection.h"
#include "InputPath.h"

namespace Moses
{

class ConfusionNet;

/** Holds all translation options, for all spans, of a particular confusion network input
 * Inherited from TranslationOptionCollection.
 */
class TranslationOptionCollectionConfusionNet : public TranslationOptionCollection
{
public:
  typedef std::vector< std::vector<InputPathList> > InputPathMatrix;

protected:
  InputPathMatrix	m_inputPathMatrix; /*< contains translation options */

  InputPathList &GetInputPathList(size_t startPos, size_t endPos);
  bool CreateTranslationOptionsForRangeNew(const DecodeGraph &decodeStepList
      , size_t startPosition
      , size_t endPosition
      , bool adhereTableLimit
      , size_t graphInd);

  bool CreateTranslationOptionsForRangeLEGACY(const DecodeGraph &decodeStepList
      , size_t startPosition
      , size_t endPosition
      , bool adhereTableLimit
      , size_t graphInd);

public:
  TranslationOptionCollectionConfusionNet
  (ttasksptr const& ttask, const ConfusionNet &source);
  // , size_t maxNoTransOptPerCoverage, float translationOptionThreshold);

  void ProcessUnknownWord(size_t sourcePos);
  void CreateTranslationOptions();

  bool
  CreateTranslationOptionsForRange
  (const DecodeGraph &decodeStepList, size_t spos, size_t epos,
   bool adhereTableLimit, size_t graphInd);

protected:

};

}
#endif

