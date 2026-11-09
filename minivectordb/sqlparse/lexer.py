"""Hand written tokenizer for the MiniVectorDB SQL subset.

The lexer turns a SQL text into a flat token stream that the recursive descent
parser (:mod:`minivectordb.sqlparse.parser`) consumes.  Only the constructs
that the competition subset allows are recognised; everything else is reported
as a syntax error.

Token kinds
-----------
``word``    an identifier or a keyword -- the original spelling is kept in
            ``value`` so that the emitted column labels match what the user
            wrote.  Keywords are *not* a separate token kind: the parser
            decides from the context whether a word is a keyword or a name,
            which keeps identifiers such as ``set`` or ``from`` usable.
``int``     an optionally signed integer literal (``value`` is an ``int``).
``string``  a single quoted string literal (``value`` is the decoded text).
``op``      a punctuation mark or an operator.
``eof``     end of input.

The lexer never raises anything but :class:`~minivectordb.errors.ParseError`.
Errors that are caused by the input ending in the middle of a token carry the
dynamic attribute ``incomplete = True`` so that the caller knows that more
input may complete the statement.
"""

from __future__ import annotations

from typing import List, Optional

from ..errors import ParseError

__all__ = ["Token", "tokenize", "Lexer"]


WORD_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
WORD_CONT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
)
DIGITS = frozenset("0123456789")
WHITESPACE = frozenset(" \t\r\n\f\v")

TWO_CHAR_OPS = frozenset(("<=", ">=", "!=", "<>"))
SINGLE_CHAR_OPS = frozenset("(),;*.=<>")

#: backslash escapes that are decoded inside a string literal
STRING_ESCAPES = {
    "'": "'",
    "\\": "\\",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
}


class Token:
    """A single lexical token."""

    __slots__ = ("kind", "value", "text", "pos")

    def __init__(self, kind: str, value, text: str, pos: int) -> None:
        self.kind = kind
        self.value = value
        self.text = text
        self.pos = pos

    # -- helpers ---------------------------------------------------------
    def is_op(self, symbol: str) -> bool:
        return self.kind == "op" and self.value == symbol

    def is_word(self) -> bool:
        return self.kind == "word"

    def upper(self) -> str:
        """The case folded word value (``""`` for non word tokens)."""
        if self.kind == "word":
            return self.value.upper()
        return ""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Token({self.kind!r}, {self.value!r}, pos={self.pos})"


class Lexer:
    """Tokenizer for the MiniVectorDB SQL subset."""

    def __init__(self, sql: str) -> None:
        self.sql = sql
        self.n = len(sql)
        self.i = 0

    # -- public ----------------------------------------------------------
    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            self._skip_trivia()
            if self.i >= self.n:
                break
            c = self.sql[self.i]
            if c == "'":
                tokens.append(self._string())
            elif c == '"':
                # MySQL treats double quoted text as a string literal unless
                # ANSI_QUOTES is enabled.
                tokens.append(self._string(quote='"'))
            elif c == "`":
                tokens.append(self._quoted_ident())
            elif c in DIGITS:
                tokens.append(self._number())
            elif c in "+-" and self.i + 1 < self.n and self.sql[self.i + 1] in DIGITS:
                tokens.append(self._number())
            elif c in WORD_START:
                tokens.append(self._word())
            elif self.sql[self.i : self.i + 2] in TWO_CHAR_OPS:
                tokens.append(self._op(self.sql[self.i : self.i + 2]))
            elif c in SINGLE_CHAR_OPS:
                tokens.append(self._op(c))
            else:
                self._fail(f"unexpected character {c!r}", self.i)
        tokens.append(Token("eof", None, "", self.n))
        return tokens

    # -- scanners --------------------------------------------------------
    def _skip_trivia(self) -> None:
        """Skip whitespace and comments."""
        while self.i < self.n:
            c = self.sql[self.i]
            if c in WHITESPACE:
                self.i += 1
                continue
            if c == "-" and self.sql.startswith("--", self.i):
                self.i += 2
                while self.i < self.n and self.sql[self.i] != "\n":
                    self.i += 1
                continue
            if c == "#":
                self.i += 1
                while self.i < self.n and self.sql[self.i] != "\n":
                    self.i += 1
                continue
            if c == "/" and self.sql.startswith("/*", self.i):
                end = self.sql.find("*/", self.i + 2)
                if end < 0:
                    self._fail(
                        "unterminated block comment", self.i, incomplete=True
                    )
                self.i = end + 2
                continue
            break

    def _word(self) -> Token:
        start = self.i
        while self.i < self.n and self.sql[self.i] in WORD_CONT:
            self.i += 1
        text = self.sql[start : self.i]
        return Token("word", text, text, start)

    def _quoted_ident(self) -> Token:
        start = self.i
        self.i += 1  # opening back quote
        out: List[str] = []
        while True:
            if self.i >= self.n:
                self._fail("unterminated quoted identifier", start, incomplete=True)
            c = self.sql[self.i]
            if c == "`":
                if self.sql.startswith("``", self.i):
                    out.append("`")
                    self.i += 2
                    continue
                self.i += 1
                break
            out.append(c)
            self.i += 1
        return Token("word", "".join(out), self.sql[start : self.i], start)

    def _string(self, quote: str = "'") -> Token:
        start = self.i
        self.i += 1  # opening quote
        out: List[str] = []
        while True:
            if self.i >= self.n:
                self._fail("unterminated string literal", start, incomplete=True)
            c = self.sql[self.i]
            if c == "\\":
                self.i += 1
                if self.i >= self.n:
                    self._fail(
                        "unterminated string literal", start, incomplete=True
                    )
                esc = self.sql[self.i]
                out.append(STRING_ESCAPES.get(esc, "\\" + esc))
                self.i += 1
                continue
            if c == quote:
                if self.sql.startswith(quote * 2, self.i):
                    out.append(quote)
                    self.i += 2
                    continue
                self.i += 1
                break
            out.append(c)
            self.i += 1
        return Token("string", "".join(out), self.sql[start : self.i], start)

    def _number(self) -> Token:
        start = self.i
        if self.sql[self.i] in "+-":
            self.i += 1
        while self.i < self.n and self.sql[self.i] in DIGITS:
            self.i += 1
        text = self.sql[start : self.i]
        return Token("int", int(text), text, start)

    def _op(self, symbol: str) -> Token:
        start = self.i
        self.i += len(symbol)
        return Token("op", symbol, symbol, start)

    # -- errors ----------------------------------------------------------
    def _fail(self, detail: str, pos: int, incomplete: bool = False) -> None:
        err = ParseError(f"syntax error at position {pos}: {detail}")
        err.incomplete = incomplete
        raise err


def tokenize(sql: str) -> List[Token]:
    """Tokenize ``sql`` and return the token list (the last token is EOF)."""
    if not isinstance(sql, str):
        err = ParseError("syntax error: the statement must be a string")
        err.incomplete = False
        raise err
    return Lexer(sql).tokenize()
