# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Utilities for internal use by the ``HdkOnNativeDataframe``."""

import re
import sys
import pandas
import pyarrow

from pandas import Timestamp
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
from string import ascii_uppercase, ascii_lowercase, digits
from typing import List, Tuple, Any

IDX_COL_NAME = "__index__"
ROWID_COL_NAME = "__rowid__"

# Bytes 62 and 63 are encoded with 2 characters
_BASE_EXT = ("_A", "_B")
_BASE_LIST = tuple(ascii_uppercase + ascii_lowercase + digits) + _BASE_EXT
_BASE_DICT = dict((c, i) for i, c in enumerate(_BASE_LIST))
_NON_ALPHANUM_PATTERN = re.compile("[^a-zA-Z0-9]+")
# Number of bytes in the tailing chunk
_TAIL_LEN = {"_0": 0, "_1": 1, "_2": 2, "_3": 3}
_RESERVED_NAMES = (MODIN_UNNAMED_SERIES_LABEL, ROWID_COL_NAME)

if (sys.version_info[0] >= 3) and (sys.version_info[1] >= 10):
    _COL_TYPES = str | int | float | Timestamp | None
    _COL_NAME_TYPE = _COL_TYPES | Tuple[_COL_TYPES, ...]
else:
    _COL_NAME_TYPE = Any


def encode_col_name(
    name: _COL_NAME_TYPE,
    ignore_reserved: bool = True,
) -> str:
    """
    Encode column name, using the alphanumeric and underscore characters only.

    The supported name types are specified in the type hints. Non-string names
    are converted to string and prefixed with a corresponding tag. The strings
    are encoded in the following way:
      - All alphanum characters are left as is. I.e., if the column name
        consists from the alphanum characters only, the original name is
        returned.
      - Non-alphanum parts of the name are encoded, using a customized
        version of the base64 algorithm, that allows alphanum characters only.

    Parameters
    ----------
    name : str, int, float, Timestamp, None, tuple
        Column name to be encoded.
    ignore_reserved : bool, default: True
        Do not encode reserved names.

    Returns
    -------
    str
        Encoded name.
    """
    if name is None:
        return "_N"
    if isinstance(name, int):
        return f"_I{str(name)}"
    if isinstance(name, float):
        return f"_F{str(name)}"
    if isinstance(name, Timestamp):
        return f"_D{encode_col_name((name.timestamp(), str(name.tz)))}"
    if isinstance(name, tuple):
        dst = ["_T"]
        count = len(name)
        for n in name:
            dst.append(encode_col_name(n))
            count -= 1
            if count != 0:
                dst.append("_S")  # Separator
        return "".join(dst)
    if len(name) == 0:
        return "_E"
    if ignore_reserved and (name.startswith(IDX_COL_NAME) or name in _RESERVED_NAMES):
        return name

    non_alpha = _NON_ALPHANUM_PATTERN.search(name)
    if not non_alpha:
        # If the name consists only from alphanum characters, return it as is.
        return name

    dst = []
    off = 0
    while non_alpha:
        start = non_alpha.start()
        end = non_alpha.end()
        dst.append(name[off:start])
        _quote(name[start:end], dst)
        off = end
        non_alpha = _NON_ALPHANUM_PATTERN.search(name, off)
    dst.append(name[off:])
    return "".join(dst)


def decode_col_name(name: str) -> _COL_NAME_TYPE:
    """
    Decode column name, previously encoded with encode_col_name().

    Parameters
    ----------
    name : str
        Encoded name.

    Returns
    -------
    str, int, float, Timestamp, None, tuple
        Decoded name.
    """
    if name.startswith("_"):
        if name.startswith(IDX_COL_NAME) or name in _RESERVED_NAMES:
            return name
        char = name[1]
        if char == "N":
            return None
        if char == "I":
            return int(name[2:])
        if char == "F":
            return float(name[2:])
        if char == "D":
            stamp = decode_col_name(name[2:])
            return Timestamp.fromtimestamp(stamp[0], tz=stamp[1])
        if char == "T":
            dst = [decode_col_name(n) for n in name[2:].split("_S")]
            return tuple(dst)
        if char == "E":
            return ""

    idx = name.find("_Q")
    if idx == -1:
        return name

    dst = []
    off = 0
    end = len(name)
    while idx != -1:
        dst.append(name[off:idx])
        off = _unquote(name, dst, idx, end)
        idx = name.find("_Q", off)
    dst.append(name[off:])
    return "".join(dst)


def _quote(src: str, dst: List[str]):  # noqa: GL08
    base = _BASE_LIST
    chars = src.encode("UTF-8")
    off = 0
    end = len(chars)
    nbytes = 0

    dst.append("_Q")
    while off < end:
        nbytes = min(end - off, 3)
        n = 0

        # Put 8-bit integers into 24-bit integer
        for i in range(0, nbytes):
            n |= chars[off + i] << (8 * (2 - i))
        # For each 6-bit integer append the corresponding chars
        for i in range(0, nbytes + 1):
            dst.append(base[(n >> (6 * (3 - i))) & 0x3F])

        off += nbytes
    dst.extend(("_", str(nbytes)))


def _unquote(src: str, dst: List[str], off, end) -> int:  # noqa: GL08
    assert src[off : off + 2] == "_Q"
    base = _BASE_DICT
    off += 2
    raw = bytearray()

    while off < end:
        nchars = min(end - off, 4)
        nbytes = 3
        n = 0

        for i in range(0, nchars):
            char = src[off]
            off += 1

            if char == "_":
                off += 1
                char = src[off - 2 : off]
                tail = _TAIL_LEN.get(char, None)
                if tail is not None:
                    nbytes = 0 if tail == 3 else tail
                    end = off
                    break

            n |= base[char] << (6 * (3 - i))

        for i in range(0, nbytes):
            raw.append((n >> (8 * (2 - i))) & 0xFF)

    assert src[off - 2 : off] in _TAIL_LEN
    dst.append(raw.decode("UTF-8"))
    return off


class LazyProxyCategoricalDtype(pandas.CategoricalDtype):
    """
    Proxy class for lazily retrieving categorical dtypes from arrow tables.

    Parameters
    ----------
    table : pyarrow.Table
        Source table.
    column_name : str
        Column name.
    """

    def __init__(self, table: pyarrow.Table, column_name: str):
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=table is None,
            extra_log="attempted to bind 'None' pyarrow table to a lazy category",
        )
        self._table = table
        self._column_name = column_name
        self._ordered = False
        self._lazy_categories = None

    def _new(self, table: pyarrow.Table, column_name: str) -> pandas.CategoricalDtype:
        """
        Create a new proxy, if either table or column name are different.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.
        column_name : str
            Column name.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._table is None:
            # The table has been materialized, we don't need a proxy anymore.
            return pandas.CategoricalDtype(self.categories)
        elif table is self._table and column_name == self._column_name:
            return self
        else:
            return LazyProxyCategoricalDtype(table, column_name)

    @property
    def _categories(self):  # noqa: GL08
        if self._table is not None:
            chunks = self._table.column(self._column_name).chunks
            cat = pandas.concat([chunk.dictionary.to_pandas() for chunk in chunks])
            self._lazy_categories = self.validate_categories(cat.unique())
            self._table = None  # The table is not required any more
        return self._lazy_categories

    @_categories.setter
    def _set_categories(self, categories):  # noqa: GL08
        self._lazy_categories = categories
        self._table = None
