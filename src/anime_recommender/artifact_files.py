"""Local format helpers. No network or ML imports; historical bytes unchanged."""
import ctypes
import errno
import os
import sys
import hashlib
import json
from pathlib import Path

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

def atomic_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def publish_directory(staging, destination):
    """Atomic rename WITHOUT replacement; fail closed on unsupported systems."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows refuses an existing destination.
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError(errno.ENOTSUP, "Atomic no-replace directory rename unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
    else:
        raise OSError(errno.ENOTSUP, "Atomic immutable publication supported on Windows/Linux only")
