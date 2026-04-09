from __future__ import annotations

import base64
import contextlib
import ctypes
import json
import os
from ctypes import POINTER, Structure, byref, c_char, c_void_p, c_wchar_p
from pathlib import Path
from typing import Any, Final

windll: Any = getattr(ctypes, "windll", None) if os.name == "nt" else None

_DESCRIPTION: Final[str] = "YouTubeIntel Desktop Secrets"


def _win_error() -> OSError:
    return OSError("Windows DPAPI call failed")


class DATA_BLOB(Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", POINTER(c_char))]


def _blob_from_bytes(data: bytes) -> DATA_BLOB:
    buffer = ctypes.create_string_buffer(data)
    return DATA_BLOB(len(data), ctypes.cast(buffer, POINTER(c_char)))


def _bytes_from_blob(blob: DATA_BLOB) -> bytes:
    return ctypes.string_at(blob.pbData, blob.cbData)


def _protect_bytes(data: bytes) -> bytes:
    if os.name != "nt":
        return data
    in_blob = _blob_from_bytes(data)
    out_blob = DATA_BLOB()
    if not windll.crypt32.CryptProtectData(
        byref(in_blob), c_wchar_p(_DESCRIPTION), None, None, None, 0, byref(out_blob)
    ):
        raise _win_error()
    try:
        return _bytes_from_blob(out_blob)
    finally:
        windll.kernel32.LocalFree(c_void_p(ctypes.addressof(out_blob.pbData.contents)))


def _unprotect_bytes(data: bytes) -> bytes:
    if os.name != "nt":
        return data
    in_blob = _blob_from_bytes(data)
    out_blob = DATA_BLOB()
    if not windll.crypt32.CryptUnprotectData(
        byref(in_blob), None, None, None, None, 0, byref(out_blob)
    ):
        raise _win_error()
    try:
        return _bytes_from_blob(out_blob)
    finally:
        windll.kernel32.LocalFree(c_void_p(ctypes.addressof(out_blob.pbData.contents)))


def save_secret_payload(path: Path, payload: dict[str, str]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    encrypted = _protect_bytes(raw)
    encoded = base64.b64encode(encrypted).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ciphertext": encoded}, indent=2), encoding="utf-8")
    with contextlib.suppress(Exception):
        os.chmod(path, 0o600)


def load_secret_payload(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        encoded = raw.get("ciphertext", "")
        encrypted = base64.b64decode(encoded.encode("ascii"))
        decrypted = _unprotect_bytes(encrypted)
        payload = json.loads(decrypted.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
