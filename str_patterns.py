"""SQL LIKE patterns (F-034): one parse, three consumers.

``parse_like`` turns a pattern (and an optional ESCAPE character) into tokens, with DuckDB's rules: no
escape character unless one is given (a backslash is an ordinary character), ``%`` is any run of
characters, ``_`` is exactly one character (a UTF-8 code point, not a byte). The tokens give

* ``regex``  a compiled Python regex, for the entries of a dictionary (dictionary-sized work),
* ``arrow``  an Arrow ``match_like`` pattern, for the Python path (Arrow's escape is always a backslash),
* ``encode`` the byte program the native filter interprets: UTF-8 literals, ``0xFF`` = ``%`` and
  ``0xFE`` = ``_`` (neither byte occurs in UTF-8 text).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

ANY_RUN = 0xFF      # '%' in the encoded program
ANY_ONE = 0xFE      # '_' in the encoded program

Token = Tuple[str, str]     # ("lit", text) | ("run", "%") | ("one", "_")


def check_escape(escape: Optional[str]) -> None:
    if escape is not None and (not isinstance(escape, str) or len(escape) != 1):
        raise ValueError("LIKE escape must be a single character")


def parse_like(pattern: str, escape: Optional[str] = None) -> List[Token]:
    check_escape(escape)
    toks: List[Token] = []

    def lit(ch: str):
        if toks and toks[-1][0] == "lit":
            toks[-1] = ("lit", toks[-1][1] + ch)
        else:
            toks.append(("lit", ch))

    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if escape is not None and ch == escape:
            if i + 1 >= n:
                raise ValueError("LIKE pattern must not end with the escape character")
            lit(pattern[i + 1])
            i += 2
            continue
        if ch == "%":
            if not toks or toks[-1][0] != "run":
                toks.append(("run", "%"))
        elif ch == "_":
            toks.append(("one", "_"))
        else:
            lit(ch)
        i += 1
    return toks


def literal_like(kind: str, text: str) -> List[Token]:
    """The tokens of ``startswith`` / ``endswith`` / ``contains`` (the text has no wildcards)."""
    body: List[Token] = [("lit", text)] if text else []
    run: List[Token] = [("run", "%")]
    return {"startswith": body + run, "endswith": run + body, "contains": run + body + run}[kind] \
        if text else run


def regex(toks: List[Token], ignore_case: bool = False):
    out = []
    for kind, text in toks:
        out.append(re.escape(text) if kind == "lit" else ".*" if kind == "run" else ".")
    return re.compile("".join(out), re.DOTALL | (re.IGNORECASE if ignore_case else 0))


def arrow(toks: List[Token]) -> str:
    out = []
    for kind, text in toks:
        if kind == "lit":
            out.append(text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))
        else:
            out.append(text)
    return "".join(out)


def encode(toks: List[Token]) -> bytes:
    out = bytearray()
    for kind, text in toks:
        if kind == "lit":
            out += text.encode("utf-8")
        else:
            out.append(ANY_RUN if kind == "run" else ANY_ONE)
    return bytes(out)


def match_like(arr, pattern: str, escape: Optional[str] = None, ignore_case: bool = False):
    """Arrow's boolean mask of ``arr LIKE pattern`` (the Python path: dictionaries, fallbacks)."""
    from ._lazy_pc import pc
    return pc.match_like(arr, pattern=arrow(parse_like(pattern, escape)), ignore_case=ignore_case)
