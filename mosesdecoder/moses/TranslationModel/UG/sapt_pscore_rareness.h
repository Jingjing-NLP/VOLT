// -*- c++ -*-
// Phrase scorer that rewards the number of phrase pair occurrences in a bitext
// with the asymptotic function x/(j+x) where x > 0 is a function
// parameter that determines the steepness of the rewards curve
// written by Ulrich Germann

#include "sapt_pscore_base.h"
#include <boost/dynamic_bitset.hpp>

namespace sapt  {

  // rareness penalty: x/(n+x)
  template<typename Token>
  class
  PScoreRareness : public SingleRealValuedParameterPhraseScorerFamily<Token>
  {
  public:
    PScoreRareness(std::string const spec)
    {
      this->m_tag = "rare";
      this->init(spec);
    }

    bool
    isLogVal(int i) const { return false; }

    void
    operator()(Bitext<Token> const& bt,
         PhrasePair<Token>& pp,
         std::vector<float> * dest = NULL) const
    {
      if (!dest) dest = &pp.fvals;
      size_t i = this->m_index;
      BOOST_FOREACH(float const x, this->m_x)
	(*dest).at(i++) = x/(x + pp.joint);
    }
  };
} // namespace sapt
