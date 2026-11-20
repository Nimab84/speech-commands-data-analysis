"""Package Phase 2 for CW and prepare a separate GitHub upload folder."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT/'reports/phase2.pdf'
    verification = json.loads((ROOT/'tmp/verification.json').read_text())
    if hashlib.sha256(report.read_bytes()).hexdigest()!=verification['pdf_sha256']:
        raise ValueError('Verify the current report before packaging')
    names = ['README.md','FEATURES.md','AI_USAGE.md','PROJECT_CARD.md','requirements.txt','.gitignore','.gitattributes',
             'configs/phase2.json','data/phase1_inventory.csv','data/source_manifest.json','data/mini_source_README.md',
             'reports/phase2.pdf','reports/summary.json']
    files = [ROOT/name for name in names]
    for folder,pattern in [('src','*.py'),('data/processed','*.csv'),('reports/tables','*.csv'),
                           ('figures','*.png'),('artifacts','*.joblib'),('artifacts','*.npz')]:
        files.extend(sorted((ROOT/folder).glob(pattern)))
    members = {path.relative_to(ROOT).as_posix():path for path in files}
    for required in ['AI_USAGE.md','reports/phase2.pdf','src/run_phase2.py','FEATURES.md']:
        if required not in members or not members[required].is_file():
            raise ValueError('Missing delivery item: '+required)
    target = ROOT/'output/phase2_path_d.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in members.items():
            archive.write(path,name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())==set(members)
        assert all(archive.read(name)==path.read_bytes() for name,path in members.items())
    upload = ROOT/'output/github_upload'
    for name,path in members.items():
        dest = upload/'phase2'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    # Updated root documents let later phases share one course repository.
    for name in ['README.md','PROJECT_CARD.md','AI_USAGE.md']:
        source = ROOT.parent/name
        if source.is_file() and (ROOT.parent/'phase2').resolve()==ROOT.resolve():
            shutil.copy2(source,upload/name)
    print(f'Created {target.name}: {len(members)} files, {target.stat().st_size:,} bytes; CRC and content checks passed.')
    print('Prepared output/github_upload with the phase2 folder and available root documentation updates.')


if __name__ == '__main__':
    main()
