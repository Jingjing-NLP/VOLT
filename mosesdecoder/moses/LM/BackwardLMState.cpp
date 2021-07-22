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

#include "moses/LM/BackwardLMState.h"
#include "lm/state.hh"

namespace Moses
{

size_t BackwardLMState::hash() const
{
  size_t ret = hash_value(state.left);
  return ret;
}
bool BackwardLMState::operator==(const FFState& o) const
{
  const BackwardLMState &other = static_cast<const BackwardLMState &>(o);
  bool ret = state.left == other.state.left;
  return ret;
}

}
