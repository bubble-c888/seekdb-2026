"""SQL front end of MiniVectorDB.

The package exposes the statement parser together with the AST node
definitions:

>>> from minivectordb.sqlparse import parse
>>> parse("SELECT * FROM docs").table
'docs'
"""

from __future__ import annotations

from . import ast
from .lexer import Token, tokenize
from .parser import Parser, parse

__all__ = ["parse", "ast", "tokenize", "Token", "Parser"]
