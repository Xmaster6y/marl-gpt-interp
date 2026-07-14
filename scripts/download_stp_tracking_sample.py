"""Download a bounded tracking CSV sample from a large remote STP ZIP using HTTP ranges."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import hydra
from loguru import logger
from omegaconf import DictConfig


class HTTPRangeReader(io.RawIOBase):
    """Minimal cached seekable reader backed by HTTP byte-range requests."""

    def __init__(self, url: str, block_size: int = 4 * 1024 * 1024):
        self.url = url
        self.block_size = block_size
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
                raise ValueError(f"Server does not advertise byte ranges: {url}")
        self.position = 0
        self.cache_start = 0
        self.cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.length)
        return self.position

    def _fill_cache(self, requested: int) -> None:
        start = self.position
        end = min(self.length - 1, start + max(self.block_size, requested) - 1)
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise OSError(f"Expected HTTP 206 for range {start}-{end}, got {response.status}")
            self.cache = response.read()
        self.cache_start = start

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        remaining = self.length - self.position if size is None or size < 0 else min(size, self.length - self.position)
        chunks = []
        while remaining:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fill_cache(remaining)
                cache_end = self.cache_start + len(self.cache)
            offset = self.position - self.cache_start
            count = min(remaining, cache_end - self.position)
            chunks.append(self.cache[offset : offset + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


@hydra.main(config_path="../configs/download_stp_tracking_sample", version_base=None)
def main(cfg: DictConfig) -> dict:
    script_cfg = cfg
    url = str(script_cfg.archive_url)
    member_name = str(script_cfg.member_name)
    max_data_rows = int(script_cfg.max_data_rows)
    output_path = _resolve(_repo_root(), str(script_cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remote = HTTPRangeReader(url, block_size=int(script_cfg.range_block_bytes))
    with zipfile.ZipFile(remote) as archive:
        matches = [info for info in archive.infolist() if PurePosixPath(info.filename).name == member_name]
        if len(matches) != 1:
            raise ValueError(f"Expected one ZIP member named {member_name!r}, found {len(matches)}")
        info = matches[0]
        digest = hashlib.sha256()
        rows_written = 0
        with archive.open(info) as source, output_path.open("wb") as output:
            for line_number, line in enumerate(source):
                if line_number > max_data_rows:
                    break
                output.write(line)
                digest.update(line)
                rows_written += 1

    manifest = {
        "archive_url": url,
        "archive_content_length": remote.length,
        "archive_etag": remote.etag,
        "member_name": info.filename,
        "member_crc32": f"{info.CRC:08x}",
        "member_compressed_size": info.compress_size,
        "member_uncompressed_size": info.file_size,
        "output_path": str(output_path),
        "lines_written": rows_written,
        "data_rows_written": max(0, rows_written - 1),
        "sample_sha256": digest.hexdigest(),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info(f"Wrote {rows_written} lines from {info.filename} to {output_path}")
    return manifest


if __name__ == "__main__":
    main()
