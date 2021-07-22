/*********************************
tercpp: an open-source Translation Edit Rate (TER) scorer tool for Machine Translation.

Copyright 2010-2013, Christophe Servan, LIUM, University of Le Mans, France
Contact: christophe.servan@lium.univ-lemans.fr

The tercpp tool and library are free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 2.1 of the licence, or
(at your option) any later version.

This program and library are distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation,
Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
**********************************/
#ifndef __STRINGINFOSHASHER_H__
#define __STRINGINFOSHASHER_H__
#include <string>
// #include <ext/hash_map>
#include <iostream>
#include <vector>
#include "tools.h"

using namespace std;
namespace TERCPPNS_HashMapSpace
{
class stringInfosHasher
{
private:
  long m_hashKey;
  string m_key;
  vector<string> m_value;

public:
  stringInfosHasher ( long cle, string cleTxt, vector<string> valueVecInt );
  long getHashKey();
  string getKey();
  vector<string> getValue();
  void setValue ( vector<string> value );
  string toString();


};


}
#endif
