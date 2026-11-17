"""
remote_zip.py - Download only SOME files from a huge zip archive on the web.

Some datasets are distributed as one giant zip (SFHQ-T2I is 23 GB) but we only
need a few thousand images from it. A zip file keeps its table of contents
(the "central directory") at the END of the file, and every member can be
located from it. HTTP "Range" requests let us download any byte range, so we:
  1. download just the central directory (a few MB) to list all members,
  2. download each wanted member's bytes individually (one request each),
  3. decompress it and check its CRC checksum.
Total download = only the files we actually use.
"""

import http.client
import io
import os
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

USER_AGENT = "deepfake-face-project/1.0 (university coursework)"


def _hf_token() -> str | None:
    """Hugging Face token if the user is logged in (optional: higher rate limits)."""
    token = os.environ.get("HF_TOKEN")
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text().strip()
    return token or None


class _RangeFile(io.RawIOBase):
    """A read-only, seekable 'file' whose reads become HTTP Range requests.

    Only used to let Python's zipfile module parse the central directory.
    """

    def __init__(self, remote: "RemoteZip"):
        self.remote, self.pos = remote, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.remote.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.remote.size - self.pos
        if n == 0:
            return b""
        data = self.remote.get_range(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


class RemoteZip:
    def __init__(self, url: str, use_hf_token: bool = True):
        self.url = url
        self.headers = {"User-Agent": USER_AGENT}
        token = _hf_token() if use_hf_token and "huggingface.co" in url else None
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self._local = threading.local()  # per-thread keep-alive connections
        self._resolve()
        self.zip = zipfile.ZipFile(_RangeFile(self))  # reads only the central directory
        self.members = {i.filename: i for i in self.zip.infolist() if not i.is_dir()}

    def _resolve(self, retries: int = 5):
        """Follow redirects once (Hugging Face sends us to a CDN) and get the file size."""
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.url, method="HEAD", headers=self.headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    self.final_url = resp.url
                    self.size = int(resp.headers["Content-Length"])
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _connection(self) -> http.client.HTTPSConnection:
        """One persistent HTTPS connection per thread (keep-alive).

        Opening a new encrypted connection for every file doubled the time per
        file (measured 1.2 s vs 0.5 s), so each download thread reuses its own.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None or getattr(self._local, "netloc", None) != urllib.parse.urlsplit(self.final_url).netloc:
            conn = http.client.HTTPSConnection(urllib.parse.urlsplit(self.final_url).netloc, timeout=120)
            self._local.conn, self._local.netloc = conn, urllib.parse.urlsplit(self.final_url).netloc
        return conn

    def get_range(self, start: int, end: int, retries: int = 6) -> bytes:
        """Bytes start..end (inclusive), retrying on network errors.

        The CDN link is temporary, so on 401/403/404 we resolve the URL again.
        """
        for attempt in range(retries):
            url = urllib.parse.urlsplit(self.final_url)
            path = url.path + (f"?{url.query}" if url.query else "")
            try:
                conn = self._connection()
                conn.request("GET", path, headers={"User-Agent": USER_AGENT,
                                                   "Range": f"bytes={start}-{end}"})
                resp = conn.getresponse()
                data = resp.read()
                if resp.status == 206:
                    return data
                if resp.status in (401, 403, 404):
                    self._resolve()
                raise IOError(f"HTTP {resp.status}")
            except (OSError, http.client.HTTPException):
                self._local.conn = None  # drop the broken connection, open a new one
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def read_member(self, name: str) -> bytes:
        """Download and decompress one member with a single range request."""
        info = self.members[name]
        # Local file header = 30 fixed bytes + file name + "extra" field (whose
        # length can differ from the central directory, so allow some slack).
        slack = 30 + len(info.filename.encode()) + 1024
        blob = self.get_range(info.header_offset, info.header_offset + slack + info.compress_size)
        signature, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", blob[:30])
        if signature != 0x04034B50:
            raise IOError(f"bad local header for {name}")
        start = 30 + name_len + extra_len
        raw = blob[start:start + info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            data = raw
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(raw)
        else:
            raise IOError(f"unsupported compression {info.compress_type} for {name}")
        if zlib.crc32(data) != info.CRC:
            raise IOError(f"CRC mismatch for {name}")
        return data

    def download(self, names: list[str], out_dir: Path, workers: int = 16,
                 progress_every: int = 500) -> list[Path]:
        """Download members in parallel into out_dir (keeping their paths); skip existing."""
        out_dir.mkdir(parents=True, exist_ok=True)

        def fetch(name):
            dst = out_dir / name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + ".part")
                tmp.write_bytes(self.read_member(name))
                tmp.rename(dst)  # only complete files ever get the final name
            return dst

        done = []
        with ThreadPoolExecutor(workers) as pool:
            for i, path in enumerate(pool.map(fetch, names), 1):
                done.append(path)
                if i % progress_every == 0 or i == len(names):
                    print(f"      {i}/{len(names)}", flush=True)
        return done
