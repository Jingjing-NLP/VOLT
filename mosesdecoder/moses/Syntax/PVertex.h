#pragma once

#include "moses/Word.h"
#include "moses/Range.h"

namespace Moses
{
namespace Syntax
{

struct PVertex {
public:
  PVertex(const Range &wr, const Word &w) : span(wr), symbol(w) {}

  Range span;
  Word symbol;
};

inline bool operator==(const PVertex &v, const PVertex &w)
{
  return v.span == w.span && v.symbol == w.symbol;
}

}  // Syntax
}  // Moses
