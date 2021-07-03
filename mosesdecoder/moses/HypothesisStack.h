#ifndef moses_HypothesisStack_h
#define moses_HypothesisStack_h

#include <vector>
#include <set>
#include <boost/unordered_set.hpp>
#include "Hypothesis.h"
#include "Bitmap.h"

namespace Moses
{

class Manager;

/** abstract unique set of hypotheses that cover a certain number of words,
 *  ie. a stack in phrase-based decoding
 */
class HypothesisStack
{

protected:
  typedef boost::unordered_set< Hypothesis*, UnorderedComparer<Hypothesis>, UnorderedComparer<Hypothesis> > _HCType;
  _HCType m_hypos; /**< contains hypotheses */
  Manager& m_manager;

public:
  HypothesisStack(Manager& manager): m_manager(manager) {}
  typedef _HCType::iterator iterator;
  typedef _HCType::const_iterator const_iterator;
  //! iterators
  const_iterator begin() const {
    return m_hypos.begin();
  }
  const_iterator end() const {
    return m_hypos.end();
  }
  size_t size() const {
    return m_hypos.size();
  }
  virtual inline float GetWorstScore() const {
    return -std::numeric_limits<float>::infinity();
  };
  virtual float GetWorstScoreForBitmap( WordsBitmapID ) {
    return -std::numeric_limits<float>::infinity();
  };
  virtual float GetWorstScoreForBitmap( const Bitmap& ) {
    return -std::numeric_limits<float>::infinity();
  };

  virtual ~HypothesisStack();
  virtual bool AddPrune(Hypothesis *hypothesis) = 0;
  virtual const Hypothesis *GetBestHypothesis() const = 0;
  virtual std::vector<const Hypothesis*> GetSortedList() const = 0;

  //! remove hypothesis pointed to by iterator but don't delete the object
  virtual void Detach(const HypothesisStack::iterator &iter);
  /** destroy Hypothesis pointed to by iterator (object pool version) */
  virtual void Remove(const HypothesisStack::iterator &iter);

};

}

#endif
