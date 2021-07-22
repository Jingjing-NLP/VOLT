// -*- mode: c++; indent-tabs-mode: nil; tab-width:2  -*-
// Memory-mapped corpus track
// (c) Ulrich Germann. All rights reserved

#include <sstream>

#include "ug_mm_ttrack.h"
#include "tpt_pickler.h"

namespace sapt
{
  using namespace std;

#if 0
  template<>
  id_type
  Ttrack<id_type>::
  toID(id_type const& t)
  {
    return t;
  }
#endif

  /** @return string representation of sentence /sid/ */
  template<>
  string
  Ttrack<id_type>::
  str(id_type sid, TokenIndex const& T) const
  {
    assert(sid < numTokens());
    id_type const* stop = sntEnd(sid);
    id_type const* strt = sntStart(sid);
    ostringstream buf;
    if (strt < stop) buf << T[*strt];
    while (++strt < stop)
      buf << " " << T[*strt];
    return buf.str();
  }

  size_t 
  len_from_pid(uint64_t pid)
  {
    static uint64_t two16 = uint64_t(1)<<16;
    return pid%two16;
  }

#if 0
  template<>
  string
  Ttrack<id_type>::
  str(id_type sid, Vocab const& V) const
  {
    assert(sid < numTokens());
    id_type const* stop = sntEnd(sid);
    id_type const* strt = sntStart(sid);
    ostringstream buf;
    if (strt < stop) buf << V[*strt].str;
    while (++strt < stop)
      buf << " " << V[*strt].str;
    return buf.str();
  }
#endif
}
