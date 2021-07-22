#include <set>
#include "moses/PP/TargetConstituentBoundariesLeftPhraseProperty.h"
#include "moses/FactorCollection.h"
#include "moses/Util.h"
#include <iostream>
#include <queue>
#include <ostream>

namespace Moses
{

void TargetConstituentBoundariesLeftPhraseProperty::ProcessValue(const std::string &value)
{
  FactorCollection &factorCollection = FactorCollection::Instance();
  std::vector<std::string> tokens;
  Tokenize(tokens, value, " ");
  std::vector<std::string>::const_iterator tokenIter = tokens.begin();
  while (tokenIter != tokens.end()) {
    try {

      std::vector<std::string> constituents;
      Tokenize(constituents, *tokenIter, "<");
      ++tokenIter;
      float count = std::atof( tokenIter->c_str() );
      ++tokenIter;

      std::set<const Factor* > dedup;

      for ( std::vector<std::string>::iterator constituentIter = constituents.begin();
            constituentIter != constituents.end(); ++constituentIter ) {

        const Factor* constituentFactor = factorCollection.AddFactor(*constituentIter,false);

        std::pair< std::set<const Factor* >::iterator, bool > dedupIns =
          dedup.insert(constituentFactor);
        if ( dedupIns.second ) {

          std::pair< TargetConstituentBoundariesLeftCollection::iterator, bool > inserted =
            m_constituentsCollection.insert(std::make_pair(constituentFactor,count));
          if ( !inserted.second ) {
            (inserted.first)->second += count;
          }
        }
      }

    } catch (const std::exception &e) {
      UTIL_THROW2("TargetConstituentBoundariesLeftPhraseProperty: Read error. Flawed property?  " << value);
    }
  }
};

void TargetConstituentBoundariesLeftPhraseProperty::Print(std::ostream& out) const
{
  for ( TargetConstituentBoundariesLeftCollection::const_iterator it = m_constituentsCollection.begin();
        it != m_constituentsCollection.end(); ++it ) {
    if ( it != m_constituentsCollection.begin() ) {
      out << " ";
    }
    out << *(it->first) << " " << it->second;
  }
}

} // namespace Moses

