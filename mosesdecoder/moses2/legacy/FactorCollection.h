// $Id$

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

// reserve space for non-terminal symbols (ensuring consecutive numbering, and allowing quick lookup by ID)
#ifndef moses_MaxNumNonterminals
#define moses_MaxNumNonterminals 10000
#endif

#ifdef WITH_THREADS
#include <boost/thread/shared_mutex.hpp>
#endif

#include "util/murmur_hash.hh"
#include <boost/unordered_set.hpp>

#include <functional>
#include <string>

#include "util/string_piece.hh"
#include "util/pool.hh"
#include "Factor.h"

namespace Moses2
{

class System;

/** We don't want Factor to be copyable by anybody.  But we also want to store
 * it in an STL container.  The solution is that Factor's copy constructor is
 * private and friended to FactorFriend.  The STL containers can delegate
 * copying, so friending the container isn't sufficient.  STL containers see
 * FactorFriend's public copy constructor and everybody else sees Factor's
 * private copy constructor.
 */
struct FactorFriend {
  Factor in;
};

/** collection of factors
 *
 * All Factors in moses are accessed and created by a FactorCollection.
 * By enforcing this strict creation processes (ie, forbidding factors
 * from being created on the stack, etc), their memory addresses can
 * be used as keys to uniquely identify them.
 * Only 1 FactorCollection object should be created.
 */
class FactorCollection
{
  friend std::ostream& operator<<(std::ostream&, const FactorCollection&);
  friend class System;

  struct HashFactor: public std::unary_function<const FactorFriend &,
      std::size_t> {
    std::size_t operator()(const FactorFriend &factor) const {
      return util::MurmurHashNative(factor.in.m_string.data(),
                                    factor.in.m_string.size());
    }
  };
  struct EqualsFactor: public std::binary_function<const FactorFriend &,
      const FactorFriend &, bool> {
    bool operator()(const FactorFriend &left, const FactorFriend &right) const {
      return left.in.GetString() == right.in.GetString();
    }
  };
  typedef boost::unordered_set<FactorFriend, HashFactor, EqualsFactor> Set;
  Set m_set;
  Set m_setNonTerminal;

  util::Pool m_string_backing;

#ifdef WITH_THREADS
  //reader-writer lock
  mutable boost::shared_mutex m_accessLock;
#endif

  size_t m_factorIdNonTerminal; /**< unique, contiguous ids, starting from 0, for each non-terminal factor */
  size_t m_factorId; /**< unique, contiguous ids, starting from moses_MaxNumNonterminals, for each terminal factor */

  //! constructor. only the 1 static variable can be created
  FactorCollection() :
    m_factorIdNonTerminal(0), m_factorId(moses_MaxNumNonterminals) {
  }

public:
  ~FactorCollection();

  /** returns a factor with the same direction, factorType and factorString.
   *	If a factor already exist in the collection, return the existing factor, if not create a new 1
   */
  const Factor *AddFactor(const StringPiece &factorString, const System &system,
                          bool isNonTerminal);

  size_t GetNumNonTerminals() {
    return m_factorIdNonTerminal;
  }

  const Factor *GetFactor(const StringPiece &factorString, bool isNonTerminal =
                            false);

};

}

