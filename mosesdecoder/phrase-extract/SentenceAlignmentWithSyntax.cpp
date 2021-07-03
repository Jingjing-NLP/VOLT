/***********************************************************************
  Moses - factored phrase-based language decoder
  Copyright (C) 2010 University of Edinburgh

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

#include "SentenceAlignmentWithSyntax.h"

#include <map>
#include <set>
#include <string>

#include "tables-core.h"
#include "XmlException.h"
#include "XmlTree.h"
#include "util/tokenize.hh"

using namespace std;

namespace MosesTraining
{

bool SentenceAlignmentWithSyntax::processTargetSentence(const char * targetString, int sentenceID, bool boundaryRules)
{
  if (!m_targetSyntax) {
    return SentenceAlignment::processTargetSentence(targetString, sentenceID, boundaryRules);
  }

  string targetStringCPP(targetString);
  try {
    ProcessAndStripXMLTags(targetStringCPP, targetTree,
                           m_targetLabelCollection,
                           m_targetTopLabelCollection,
                           false);
  } catch (const XmlException & e) {
    std::cerr << "WARNING: failed to process target sentence at line "
              << sentenceID << ": " << e.getMsg() << std::endl;
    return false;
  }
  target = util::tokenize(targetStringCPP);
  return true;
}

bool SentenceAlignmentWithSyntax::processSourceSentence(const char * sourceString, int sentenceID, bool boundaryRules)
{
  if (!m_sourceSyntax) {
    return SentenceAlignment::processSourceSentence(sourceString, sentenceID, boundaryRules);
  }

  string sourceStringCPP(sourceString);
  try {
    ProcessAndStripXMLTags(sourceStringCPP, sourceTree,
                           m_sourceLabelCollection ,
                           m_sourceTopLabelCollection,
                           false);
  } catch (const XmlException & e) {
    std::cerr << "WARNING: failed to process source sentence at line "
              << sentenceID << ": " << e.getMsg() << std::endl;
    return false;
  }
  source = util::tokenize(sourceStringCPP);
  return true;
}

} // namespace
