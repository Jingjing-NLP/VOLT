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
#ifndef __TERCPPALIGNMENTSTRUCT_H__
#define __TERCPPALIGNMENTSTRUCT_H__


#include <vector>
#include <cstdio>
#include <string>
#include <sstream>
#include "tools.h"


using namespace std;
using namespace TERCPPNS_Tools;

namespace TERCPPNS_TERCpp
{
class alignmentStruct
{
private:
public:

// 	alignmentStruct();
// 	alignmentStruct (int _start, int _end, int _moveto, int _newloc);
// 	alignmentStruct (int _start, int _end, int _moveto, int _newloc, vector<string> _shifted);
// 	string toString();
// 	int distance() ;
// 	bool leftShift();
// 	int size();
// 	alignmentStruct operator=(alignmentStruct t);
// 	string vectorToString(vector<string> vec);

//   int start;
//   int end;
//   int moveto;
//   int newloc;
  vector<string> nwords; // The words we shifted
  vector<char> alignment ; // for pra_more output
  vector<vecInt> aftershift; // for pra_more output
  // This is used to store the cost of a shift, so we don't have to
  // calculate it multiple times.
  double cost;
  string toString();
  void set(alignmentStruct l_alignmentStruct);
};

}
#endif
