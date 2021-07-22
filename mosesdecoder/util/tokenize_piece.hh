#ifndef UTIL_TOKENIZE_PIECE_H
#define UTIL_TOKENIZE_PIECE_H

#include "util/exception.hh"
#include "util/string_piece.hh"

#include <boost/iterator/iterator_facade.hpp>

#include <algorithm>
#include <cstring>

namespace util {

// Thrown on dereference when out of tokens to parse
class OutOfTokens : public Exception {
  public:
    OutOfTokens() throw() {}
    ~OutOfTokens() throw() {}
};

class SingleCharacter {
  public:
    SingleCharacter() {}
    explicit SingleCharacter(char delim) : delim_(delim) {}

    StringPiece Find(const StringPiece &in) const {
      return StringPiece(std::find(in.data(), in.data() + in.size(), delim_), 1);
    }

  private:
    char delim_;
};

class MultiCharacter {
  public:
    MultiCharacter() {}

    explicit MultiCharacter(const StringPiece &delimiter) : delimiter_(delimiter) {}

    StringPiece Find(const StringPiece &in) const {
      return StringPiece(std::search(in.data(), in.data() + in.size(), delimiter_.data(), delimiter_.data() + delimiter_.size()), delimiter_.size());
    }

  private:
    StringPiece delimiter_;
};

class AnyCharacter {
  public:
    AnyCharacter() {}
    explicit AnyCharacter(const StringPiece &chars) : chars_(chars) {}

    StringPiece Find(const StringPiece &in) const {
      return StringPiece(std::find_first_of(in.data(), in.data() + in.size(), chars_.data(), chars_.data() + chars_.size()), 1);
    }

  private:
    StringPiece chars_;
};

class BoolCharacter {
  public:
    BoolCharacter() {}

    explicit BoolCharacter(const bool *delimiter) { delimiter_ = delimiter; }

    StringPiece Find(const StringPiece &in) const {
      for (const char *i = in.data(); i != in.data() + in.size(); ++i) {
        if (delimiter_[static_cast<unsigned char>(*i)]) return StringPiece(i, 1);
      }
      return StringPiece(in.data() + in.size(), 0);
    }

    template <unsigned Length> static void Build(const char (&characters)[Length], bool (&out)[256]) {
      memset(out, 0, sizeof(out));
      for (const char *i = characters; i != characters + Length; ++i) {
        out[static_cast<unsigned char>(*i)] = true;
      }
    }

  private:
    const bool *delimiter_;
};

class AnyCharacterLast {
  public:
    AnyCharacterLast() {}

    explicit AnyCharacterLast(const StringPiece &chars) : chars_(chars) {}

    StringPiece Find(const StringPiece &in) const {
      return StringPiece(std::find_end(in.data(), in.data() + in.size(), chars_.data(), chars_.data() + chars_.size()), 1);
    }

  private:
    StringPiece chars_;
};

template <class Find, bool SkipEmpty = false> class TokenIter : public boost::iterator_facade<TokenIter<Find, SkipEmpty>, const StringPiece, boost::forward_traversal_tag> {
  public:
    TokenIter() {}

    template <class Construct> TokenIter(const StringPiece &str, const Construct &construct) : after_(str), finder_(construct) {
      increment();
    }

    bool operator!() const {
      return current_.data() == 0;
    }
    operator bool() const {
      return current_.data() != 0;
    }

    static TokenIter<Find, SkipEmpty> end() {
      return TokenIter<Find, SkipEmpty>();
    }

  private:
    friend class boost::iterator_core_access;

    void increment() {
      do {
        StringPiece found(finder_.Find(after_));
        current_ = StringPiece(after_.data(), found.data() - after_.data());
        if (found.data() == after_.data() + after_.size()) {
          after_ = StringPiece(NULL, 0);
        } else {
          after_ = StringPiece(found.data() + found.size(), after_.data() - found.data() + after_.size() - found.size());
        }
      } while (SkipEmpty && current_.data() && current_.empty()); // Compiler should optimize this away if SkipEmpty is false.
    }

    bool equal(const TokenIter<Find, SkipEmpty> &other) const {
      return current_.data() == other.current_.data();
    }

    const StringPiece &dereference() const {
      UTIL_THROW_IF(!current_.data(), OutOfTokens, "Ran out of tokens");
      return current_;
    }

    StringPiece current_;
    StringPiece after_;

    Find finder_;
};

} // namespace util

#endif // UTIL_TOKENIZE_PIECE_H
