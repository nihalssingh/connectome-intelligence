"""Fetch the authors' archived Figure 1D outputs without downloading the 4.5 GB results.zip.

The Edmond (Max Planck Dataverse) archive of Shiu et al., doi:10.17617/3.CZODIW, holds results.zip.
The server honours HTTP Range requests, so a seekable file object over HTTP lets zipfile read the
central directory and extract single members (CRC-checked by zipfile).

    .venv/bin/python stage1/fetch_archive.py
"""
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

API = 'https://edmond.mpg.de/api'
DOI = 'doi:10.17617/3.CZODIW'
VERSION = 3  # the dataset version every comparison here was made against
OUT = Path(__file__).resolve().parents[1] / 'data' / 'shiu_archive'
# Fig. 1D sweep, plus the sugar-only conditions (second stimulus at 0 Hz) of Figs. 3a, 3b and 4e:
# four independent replicate sweeps of the same 21-GRN stimulus by the authors
WANT = re.compile(r'^results/(figure_1/(data/sugarR_\d+Hz\.parquet|fig_1d_rate(_std)?\.csv)'
                  r'|figure_3/data/Sugar_(Bitter|Ir94e)_\d+_Hz_0_Hz\.parquet'
                  r'|figure_4/data/Sugar_Water_\d+_Hz_0_Hz\.parquet)$')


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file; each cache miss fetches >= `block` bytes."""

    def __init__(self, url, size, block=4 << 20):
        self.url, self.size, self.block, self.pos = url, size, block, 0
        self.cache_start, self.cache = 0, b''
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def _fetch(self, start, n):
        end = min(self.size, start + max(n, self.block)) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise IOError(f'server ignored Range request (HTTP {r.status})')
            self.cache_start, self.cache = start, r.read()
        self.requests += 1

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b''
        lo = self.pos - self.cache_start
        if lo < 0 or lo + n > len(self.cache):
            self._fetch(self.pos, n)
            lo = 0
        out = self.cache[lo:lo + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def results_zip():
    with urllib.request.urlopen(f'{API}/datasets/:persistentId/versions/{VERSION}?persistentId={DOI}', timeout=60) as r:
        meta = json.load(r)['data']
    f = next(f['dataFile'] for f in meta['files'] if f['dataFile']['filename'] == 'results.zip')
    return f"{API}/access/datafile/{f['id']}", f['filesize'], meta.get('versionNumber')


def main():
    url, size, version = results_zip()
    rf = HTTPRangeFile(url, size)
    manifest = {'doi': DOI, 'dataset_version': version, 'url': url, 'zip_bytes': size, 'files': {}}
    with zipfile.ZipFile(rf) as z:
        members = [i for i in z.infolist() if WANT.match(i.filename)]
        print(f'results.zip: {len(z.infolist())} entries; extracting {len(members)}')
        for info in members:
            dest = OUT / info.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not (dest.is_file() and dest.stat().st_size == info.file_size):
                dest.write_bytes(z.read(info))  # zipfile verifies the CRC-32
            manifest['files'][info.filename] = {'bytes': info.file_size, 'crc32': f'{info.CRC:08x}'}
            print(f'  {info.filename} ({info.file_size:,} bytes)')
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    print(f'{rf.requests} range requests; manifest at {OUT / "MANIFEST.json"}')


if __name__ == '__main__':
    main()
