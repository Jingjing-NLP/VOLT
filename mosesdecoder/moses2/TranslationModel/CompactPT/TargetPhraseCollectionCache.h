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

#pragma once

#include <map>
#include <set>
#include <vector>

#include <boost/thread/tss.hpp>
#include <boost/shared_ptr.hpp>

#include "../../Word.h"
#include "../../Phrase.h"

namespace Moses2
{
typedef std::pair<size_t, size_t> AlignPointSizeT;

struct PhraseCompact : public std::vector<Word> {
public:
  PhraseCompact(const Phrase<Word> &copy);
};

struct TPCompact {
  std::vector<Word> words;
  std::set<AlignPointSizeT> alignment;
  std::vector<float> scores;

};

// Avoid using new due to locking
typedef std::vector<TPCompact> TargetPhraseVector;
typedef boost::shared_ptr<TargetPhraseVector> TargetPhraseVectorPtr;

/** Implementation of Persistent Cache **/
class TargetPhraseCollectionCache
{
private:
  size_t m_max;
  float m_tolerance;

  struct LastUsed {
    clock_t m_clock;
    TargetPhraseVectorPtr m_tpv;
    size_t m_bitsLeft;

    LastUsed() : m_clock(0), m_bitsLeft(0) {}

    LastUsed(clock_t clock, TargetPhraseVectorPtr tpv, size_t bitsLeft = 0)
      : m_clock(clock), m_tpv(tpv), m_bitsLeft(bitsLeft) {}
  };

  typedef std::map<PhraseCompact, LastUsed> CacheMap;
  static boost::thread_specific_ptr<CacheMap> m_phraseCache;

public:

  typedef CacheMap::iterator iterator;
  typedef CacheMap::const_iterator const_iterator;

  TargetPhraseCollectionCache(size_t max = 5000, float tolerance = 0.2)
    : m_max(max), m_tolerance(tolerance) {
  }

  iterator Begin() {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    return m_phraseCache->begin();
  }

  const_iterator Begin() const {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    return m_phraseCache->begin();
  }

  iterator End() {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    return m_phraseCache->end();
  }

  const_iterator End() const {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    return m_phraseCache->end();
  }

  /** retrieve translations for source phrase from persistent cache **/
  void Cache(const Phrase<Word> &sourcePhrase, TargetPhraseVectorPtr tpv,
             size_t bitsLeft = 0, size_t maxRank = 0) {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    // check if source phrase is already in cache
    iterator it = m_phraseCache->find(sourcePhrase);
    if(it != m_phraseCache->end())
      // if found, just update clock
      it->second.m_clock = clock();
    else {
      // else, add to cache
      if(maxRank && tpv->size() > maxRank) {
        TargetPhraseVectorPtr tpv_temp(new TargetPhraseVector());
        tpv_temp->resize(maxRank);
        std::copy(tpv->begin(), tpv->begin() + maxRank, tpv_temp->begin());
        (*m_phraseCache)[sourcePhrase] = LastUsed(clock(), tpv_temp, bitsLeft);
      } else
        (*m_phraseCache)[sourcePhrase] = LastUsed(clock(), tpv, bitsLeft);
    }
  }

  std::pair<TargetPhraseVectorPtr, size_t> Retrieve(const Phrase<Word> &sourcePhrase) {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    iterator it = m_phraseCache->find(sourcePhrase);
    if(it != m_phraseCache->end()) {
      LastUsed &lu = it->second;
      lu.m_clock = clock();
      return std::make_pair(lu.m_tpv, lu.m_bitsLeft);
    } else
      return std::make_pair(TargetPhraseVectorPtr(), 0);
  }

  // if cache full, reduce
  void Prune() {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    if(m_phraseCache->size() > m_max * (1 + m_tolerance)) {
      typedef std::set<std::pair<clock_t, PhraseCompact > > Cands;
      Cands cands;
      for(CacheMap::iterator it = m_phraseCache->begin();
          it != m_phraseCache->end(); it++) {
        LastUsed &lu = it->second;
        cands.insert(std::make_pair(lu.m_clock, it->first));
      }

      for(Cands::iterator it = cands.begin(); it != cands.end(); it++) {
        const PhraseCompact& p = it->second;
        m_phraseCache->erase(p);

        if(m_phraseCache->size() < (m_max * (1 - m_tolerance)))
          break;
      }
    }
  }

  void CleanUp() {
    if(!m_phraseCache.get())
      m_phraseCache.reset(new CacheMap());
    m_phraseCache->clear();
  }

};

}

