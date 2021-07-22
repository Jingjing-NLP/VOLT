/*
 * Transliteration.h
 *
 *  Created on: 28 Oct 2015
 *      Author: hieu
 */

#pragma once

#include "PhraseTable.h"

namespace Moses2
{
class Sentence;
class InputPaths;
class Range;

class Transliteration: public PhraseTable
{
public:
  Transliteration(size_t startInd, const std::string &line);
  virtual ~Transliteration();

  void Lookup(const Manager &mgr, InputPathsBase &inputPaths) const;
  virtual TargetPhrases *Lookup(const Manager &mgr, MemPool &pool,
                                InputPath &inputPath) const;

  virtual void
  EvaluateInIsolation(const System &system, const Phrase<Moses2::Word> &source,
                      const TargetPhraseImpl &targetPhrase, Scores &scores,
                      SCORE &estimatedScore) const;

  virtual void InitActiveChart(
    MemPool &pool,
    const SCFG::Manager &mgr,
    SCFG::InputPath &path) const;

  void Lookup(MemPool &pool,
              const SCFG::Manager &mgr,
              size_t maxChartSpan,
              const SCFG::Stacks &stacks,
              SCFG::InputPath &path) const;

  void LookupUnary(MemPool &pool,
                   const SCFG::Manager &mgr,
                   const SCFG::Stacks &stacks,
                   SCFG::InputPath &path) const;

protected:
  virtual void LookupNT(
    MemPool &pool,
    const SCFG::Manager &mgr,
    const Moses2::Range &subPhraseRange,
    const SCFG::InputPath &prevPath,
    const SCFG::Stacks &stacks,
    SCFG::InputPath &outPath) const;

  virtual void LookupGivenWord(
    MemPool &pool,
    const SCFG::Manager &mgr,
    const SCFG::InputPath &prevPath,
    const SCFG::Word &wordSought,
    const Moses2::Hypotheses *hypos,
    const Moses2::Range &subPhraseRange,
    SCFG::InputPath &outPath) const;

  virtual void LookupGivenNode(
    MemPool &pool,
    const SCFG::Manager &mgr,
    const SCFG::ActiveChartEntry &prevEntry,
    const SCFG::Word &wordSought,
    const Moses2::Hypotheses *hypos,
    const Moses2::Range &subPhraseRange,
    SCFG::InputPath &outPath) const;

  void SetParameter(const std::string& key, const std::string& value);

protected:
  std::string m_filePath;
  std::string m_mosesDir, m_scriptDir, m_externalDir, m_inputLang, m_outputLang;

  std::vector<TargetPhraseImpl*> CreateTargetPhrases(
    const Manager &mgr,
    MemPool &pool,
    const SubPhrase<Moses2::Word> &sourcePhrase,
    const std::string &outDir) const;

};

}

