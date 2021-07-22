// -*- c++ -*-

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <iostream>

#ifdef WITH_THREADS
#include <boost/thread/tss.hpp>
#endif

#include "moses/TypeDef.h"
#include "moses/Phrase.h"
#include "moses/InputType.h"
#include "moses/ConfusionNet.h"
#include "moses/Sentence.h"
#include "moses/PrefixTreeMap.h"

namespace Moses
{

class Phrase;
class InputType;
class ConfusionNet;

//! additional types
class LexicalReorderingTable
{
public:
  LexicalReorderingTable(const FactorList& f_factors,
                         const FactorList& e_factors,
                         const FactorList& c_factors)
    : m_FactorsF(f_factors)
    , m_FactorsE(e_factors)
    , m_FactorsC(c_factors) { }

  virtual
  ~LexicalReorderingTable() { }

public:
  static
  LexicalReorderingTable*
  LoadAvailable(const std::string& filePath,
                const FactorList& f_factors,
                const FactorList& e_factors,
                const FactorList& c_factors);

  virtual
  Scores
  GetScore(const Phrase& f, const Phrase& e, const Phrase& c) = 0;

  virtual
  void
  InitializeForInput(ttasksptr const& ttask) {
    /* override for on-demand loading */
  };

  virtual
  void
  InitializeForInputPhrase(const Phrase&) { }


  const FactorList& GetFFactorMask() const {
    return m_FactorsF;
  }
  const FactorList& GetEFactorMask() const {
    return m_FactorsE;
  }
  const FactorList& GetCFactorMask() const {
    return m_FactorsC;
  }

  virtual
  void
  DbgDump(std::ostream* out) const {
    *out << "Overwrite in subclass...\n";
  };
  // why is this not a pure virtual function? - UG

protected:
  FactorList m_FactorsF;
  FactorList m_FactorsE;
  FactorList m_FactorsC;
};

//! @todo what is this?
class LexicalReorderingTableMemory
  : public LexicalReorderingTable
{
  typedef std::map< std::string, std::vector<float> > TableType;
  TableType m_Table;

  //implements LexicalReorderingTable saving all scores in one large std::map<> thingy
  //to be used for non binary tables... uses a LOT of memory
public:
  LexicalReorderingTableMemory(const std::string& filePath,
                               const std::vector<FactorType>& f_factors,
                               const std::vector<FactorType>& e_factors,
                               const std::vector<FactorType>& c_factors);

  virtual
  ~LexicalReorderingTableMemory();

public:
  virtual
  std::vector<float>
  GetScore(const Phrase& f, const Phrase& e, const Phrase& c);

  void
  DbgDump(std::ostream* out) const;

private:

  std::string
  MakeKey(const Phrase& f, const Phrase& e, const Phrase& c) const;

  std::string
  MakeKey(const std::string& f, const std::string& e, const std::string& c) const;

  void
  LoadFromFile(const std::string& filePath);
};

class LexicalReorderingTableTree
  : public LexicalReorderingTable
{
  //implements LexicalReorderingTable using the crafty PDT code...

  typedef std::map< std::string, Candidates > CacheType;

#ifdef WITH_THREADS
  typedef boost::thread_specific_ptr<PrefixTreeMap> TableType;
#else
  typedef std::auto_ptr<PrefixTreeMap> TableType;
#endif

  static const int SourceVocId = 0;
  static const int TargetVocId = 1;

  bool        m_UseCache;
  std::string m_FilePath;
  CacheType   m_Cache;
  TableType   m_Table;

public:

  static
  bool
  Create(std::istream& inFile, const std::string& outFileName);

  LexicalReorderingTableTree(const std::string& filePath,
                             const std::vector<FactorType>& f_factors,
                             const std::vector<FactorType>& e_factors,
                             const std::vector<FactorType>& c_factors);

  ~LexicalReorderingTableTree();

  bool IsCacheEnabled() const {
    return m_UseCache;
  };
  void EnableCache()  {
    m_UseCache = true;
  };
  void DisableCache() {
    m_UseCache = false;
  };
  void ClearCache()   {
    if (m_UseCache) m_Cache.clear();
  };

  virtual
  std::vector<float>
  GetScore(const Phrase& f, const Phrase& e, const Phrase& c);

  virtual
  void
  InitializeForInput(ttasksptr const& ttask);

  virtual
  void
  InitializeForInputPhrase(const Phrase& f) {
    ClearCache();
    auxCacheForSrcPhrase(f);
  }


private:
  std::string
  MakeCacheKey(const Phrase& f, const Phrase& e) const;

  IPhrase
  MakeTableKey(const Phrase& f, const Phrase& e) const;

  void
  Cache(const ConfusionNet& input);

  void
  Cache(const Sentence& input);

  void
  auxCacheForSrcPhrase(const Phrase& f);

  Scores
  auxFindScoreForContext(const Candidates& cands, const Phrase& contex);

};

}

