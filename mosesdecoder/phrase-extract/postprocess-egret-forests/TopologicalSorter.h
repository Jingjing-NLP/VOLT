#pragma once

#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "Forest.h"

namespace MosesTraining
{
namespace Syntax
{
namespace PostprocessEgretForests
{

class TopologicalSorter
{
public:
  void Sort(const Forest &, std::vector<const Forest::Vertex *> &);

private:
  typedef boost::unordered_set<const Forest::Vertex *> VertexSet;

  void BuildPredSets(const Forest &);
  void Visit(const Forest::Vertex &, std::vector<const Forest::Vertex *> &);

  boost::unordered_set<const Forest::Vertex *> m_visited;
  boost::unordered_map<const Forest::Vertex *, VertexSet> m_predSets;
};

}  // namespace PostprocessEgretForests
}  // namespace Syntax
}  // namespace MosesTraining
