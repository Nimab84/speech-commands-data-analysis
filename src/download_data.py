"""Fetch the official mini archive; check integrity and extract only dataset files."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase1.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    archive_dir = ROOT / 'data/archives'
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / 'mini_speech_commands.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        print('Downloading official archive (182 MB)...', flush=True)
        with urllib.request.urlopen(cfg['dataset_url'], timeout=90) as response:
            with partial.open('wb') as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != cfg['archive_bytes']:
            raise ValueError('Archive size differs from the registered source.')
        partial.replace(archive)
    if archive.stat().st_size != cfg['archive_bytes']:
        raise ValueError('Unexpected archive size.')
    digest = sha256(archive)
    expected = cfg.get('archive_sha256')
    if expected and digest != expected:
        raise ValueError('SHA-256 mismatch. Refusing to extract changed data.')
    target = (ROOT / 'data/raw').resolve()
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts = PurePosixPath(entry.filename).parts
            if not parts or parts[0] != 'mini_speech_commands' or entry.is_dir():
                continue
            if any(p in ('..', '.') or ':' in p or '\\' in p for p in parts):
                raise ValueError('Unsafe archive member.')
            dest = target.joinpath(*parts).resolve()
            if not dest.is_relative_to(target):
                raise ValueError('Archive member escapes the extraction directory.')
            # Reading each member also verifies its ZIP CRC.
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as source, dest.open('wb') as output:
                shutil.copyfileobj(source, output)
            if dest.suffix.lower() == '.wav':
                count += 1
    metadata = {
        'url': cfg['dataset_url'], 'archive_bytes': archive.stat().st_size,
        'archive_sha256': digest, 'wav_files_extracted': count,
        'retrieved_at_utc': datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
        'identity': 'Official TensorFlow mini archive; not the full TFDS dataset',
    }
    manifest_path = ROOT / 'data/source_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('archive_sha256') == digest:
            metadata['retrieved_at_utc'] = previous['retrieved_at_utc']
    (ROOT / 'data/source_manifest.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    shutil.copyfile(target/'mini_speech_commands/README.md', ROOT/'data/mini_source_README.md')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
