#!/usr/bin/env python3
"""Authenticate using the official browser flow, then upload the frozen public dataset."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'tools/dataset_python'))
# Persistent credentials/manifests stay on D:. Xet's rename-heavy workspace uses
# tmpfs: DrvFs can report ENOENT opening a file immediately after a successful rename.
os.environ['HF_HOME']='/mnt/d/robot_tools/huggingface'
from prepare_hf_release import OUTPUT,REPO,digest,dump,append_membership
PERSISTENT_XET=Path('/mnt/d/robot_tools/huggingface/xet')
RAM_XET=Path('/dev/shm/aic-hf-xet')
UPDATE_METADATA=frozenset(('README.md','meta/info.json','meta/episodes.jsonl','meta/release.json',
    'meta/snapshot.json','meta/files.json','meta/READY.json','environment/README.md'))


def authorize(api):
    from huggingface_hub import get_token
    if not get_token():
        from huggingface_hub.utils._oauth_device import request_device_code,poll_device_token
        from huggingface_hub._login import _save_oauth_token
        code=request_device_code()
        print(json.dumps({'type':'browser_authorization_required',
            'url':code['verification_uri_complete'],'user_code':code['user_code'],
            'expires_in_seconds':code['expires_in']}),flush=True)
        response=poll_device_token(code)
        _save_oauth_token(response)
        del response
    user=api.whoami()['name']
    if user.lower()!=REPO.split('/')[0].lower():
        raise ValueError('The logged-in account is not the requested dataset owner')
    print(json.dumps({'type':'authenticated','account':user}),flush=True)


def inventory(folder):
    ready=json.loads((folder/'meta/READY.json').read_text())
    if not ready.get('complete'):
        raise ValueError('Release preparation is incomplete')
    files=json.loads((folder/'meta/files.json').read_text())
    for name in ('meta/files.json','meta/READY.json'):
        p=folder/name
        files.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
    for item in files:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix in ('.pt','.pth','.db3','.mcap'):
            raise ValueError('Unexpected upload path')
        p=folder/relative
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes']:
            raise ValueError('Frozen release file changed')
        # Small public metadata is rehashed. Large Parquet hashes are checked
        # against the upload service's LFS SHA-256 after the transfer.
        if not item['path'].startswith('data/') and digest(p)!=item['sha256']:
            raise ValueError('Release metadata checksum changed')
        if p.stat().st_size<10*1024*1024:
            content=p.read_bytes()
            item['git_blob_sha1']=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    return files


def verify_remote(api,files):
    info=api.dataset_info(REPO,files_metadata=True)
    if info.private:
        raise ValueError('Requested a public dataset, but remote is private')
    remote={f.rfilename:f for f in info.siblings}
    for expected in files:
        actual=remote.get(expected['path'])
        if actual is None or actual.size!=expected['bytes']:
            raise ValueError(f"Remote file missing or incomplete: {expected['path']}")
        if actual.lfs is not None and actual.lfs.sha256!=expected['sha256']:
            raise ValueError(f"Remote checksum mismatch: {expected['path']}")
        if actual.lfs is None and expected.get('git_blob_sha1') and actual.blob_id!=expected['git_blob_sha1']:
            raise ValueError(f"Remote Git blob checksum mismatch: {expected['path']}")
    return {'repo_id':REPO,'revision':info.sha,'public':True,
            'verified_files':len(files),'parquet_files':sum(f['path'].startswith('data/') for f in files),
            'bytes':sum(f['bytes'] for f in files)}


def verify_partial_release(info,files):
    """A cancelled first upload may not have committed snapshot.json yet."""
    if info.private:
        raise ValueError('Existing repository is private')
    expected={f['path']:f for f in files}
    for actual in info.siblings:
        if actual.rfilename=='.gitattributes':
            continue
        item=expected.get(actual.rfilename)
        if item is None or actual.size!=item['bytes']:
            raise ValueError('Existing repository is not this frozen release')
        if actual.lfs is not None:
            matches=actual.lfs.sha256==item['sha256']
        else:
            matches=bool(item.get('git_blob_sha1')) and actual.blob_id==item['git_blob_sha1']
        if not matches:
            raise ValueError('Existing remote file checksum differs from this release')


def remote_matches(actual,expected):
    if actual.size!=expected['bytes']:
        return False
    if actual.lfs is not None:
        return actual.lfs.sha256==expected['sha256']
    return bool(expected.get('git_blob_sha1')) and actual.blob_id==expected['git_blob_sha1']


def validate_update_files(previous,files):
    after={f['path']:f for f in files}
    for item in previous:
        new=after.get(item['path'])
        if new is None:
            raise ValueError('An update cannot delete a published file: '+item['path'])
        if item['path'] not in UPDATE_METADATA and (new['bytes'],new['sha256'])!=(item['bytes'],item['sha256']):
            raise ValueError('An update cannot replace a published data/asset file: '+item['path'])


def verify_update_release(info,previous,files):
    validate_update_files(previous,files)
    if info.private:
        raise ValueError('Existing repository is private')
    before={f['path']:f for f in previous};after={f['path']:f for f in files}
    for actual in info.siblings:
        name=actual.rfilename
        if name=='.gitattributes':
            continue
        if name not in after or not (remote_matches(actual,after[name]) or
                name in before and remote_matches(actual,before[name])):
            raise ValueError('Remote file does not match either frozen release: '+name)


def configure_xet_cache(cache):
    cache=cache.resolve()
    if not cache.is_relative_to(Path('/dev/shm')):
        raise ValueError('Use a RAM Xet workspace to avoid DrvFs rename/open failures')
    cache.mkdir(parents=True,exist_ok=True)
    if not any(cache.iterdir()) and PERSISTENT_XET.exists():
        for path in PERSISTENT_XET.iterdir():
            if path.is_dir() and path.name!='logs':
                shutil.copytree(path,cache/path.name,ignore=shutil.ignore_patterns('.tmp*'),dirs_exist_ok=True)
    os.environ['HF_XET_CACHE']=str(cache)
    os.environ.setdefault('HF_XET_SHARD_CACHE_SIZE_LIMIT',str(64*1024*1024))
    return cache


def preserve_xet_shards(cache):
    """Persist immutable deduplication hints, without returning active work to DrvFs."""
    for source in cache.rglob('*.mdb'):
        if any(part.startswith('.tmp') for part in source.relative_to(cache).parts):
            continue
        target=PERSISTENT_XET/source.relative_to(cache)
        if target.exists():
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.mdb.partial')
        shutil.copyfile(source,temporary)
        temporary.replace(target)


def upload_batches(api,folder,files,existing,*,workers=4,batch_size=8,cache=None,previous=None):
    if batch_size<1:
        raise ValueError('upload batch size must be positive')
    if existing is not None:
        if previous is None:verify_partial_release(existing,files)
        else:verify_update_release(existing,previous,files)
    expected={f['path']:f for f in files}
    committed={s.rfilename for s in existing.siblings if s.rfilename in expected and
        remote_matches(s,expected[s.rfilename])} if existing else set()
    pending=[f for f in files if f['path'] not in committed]
    metadata=[f for f in pending if f['path'] in UPDATE_METADATA] if previous is not None else []
    if previous is not None:pending=[f for f in pending if f['path'] not in UPDATE_METADATA]
    progress_path=folder.parent/(folder.name+'-progress.json')
    def progress(kind,**extra):
        report={'type':kind,'repo_id':REPO,'committed_files':len(committed),'total_files':len(files),
            'committed_episodes':sum(p.startswith('data/') for p in committed),
            'total_episodes':sum(p.startswith('data/') for p in expected),
            'committed_bytes':sum(expected[p]['bytes'] for p in committed),
            'total_bytes':sum(f['bytes'] for f in files),'updated_at_unix':time.time(),**extra}
        dump(progress_path,report)
        print(json.dumps(report),flush=True)
    progress('upload_progress')
    batches=(len(pending)+batch_size-1)//batch_size
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        number=start//batch_size+1
        progress('upload_batch_start',batch=number,batches=batches,batch_files=[f['path'] for f in batch])
        api.upload_large_folder(repo_id=REPO,repo_type='dataset',folder_path=folder,
            allow_patterns=[f['path'] for f in batch],num_workers=workers,
            print_report=True,print_report_every=30)
        verify_remote(api,batch)
        committed.update(f['path'] for f in batch)
        progress('upload_batch_complete',batch=number,batches=batches)
        if cache is not None:
            preserve_xet_shards(cache)
    if metadata:
        # Publish the card, counts and manifests together after all new data exists.
        from huggingface_hub import CommitOperationAdd
        current=api.dataset_info(REPO,files_metadata=True)
        verify_update_release(current,previous,files)
        progress('upload_metadata_start',metadata_files=[f['path'] for f in metadata])
        api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=current.sha,
            commit_message='Add collected insertion episodes and update dataset documentation',
            operations=[CommitOperationAdd(path_in_repo=f['path'],path_or_fileobj=folder/f['path']) for f in metadata])
        verify_remote(api,metadata)
        committed.update(f['path'] for f in metadata)
        progress('upload_metadata_complete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=OUTPUT)
    p.add_argument('--update-from',type=Path,
        help='Append to this previous frozen release; changed metadata is committed after new episodes')
    p.add_argument('--previous-readme',type=Path,
        help='Reviewed public README if it was edited after the previous release; only valid with --update-from')
    p.add_argument('--login-only',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--upload-batch-size',type=int,default=8)
    p.add_argument('--xet-cache',type=Path,default=RAM_XET)
    p.add_argument('--finish-after-pid',type=int,
        help='Wait for an existing publisher, then resume any missing files and verify the full release')
    a=p.parse_args()
    folder=a.folder.resolve()
    if not folder.is_relative_to(Path('/mnt/d')):
        p.error('Release folder must be on D:')
    if a.upload_batch_size<1:
        p.error('upload-batch-size must be positive')
    if a.previous_readme and not a.update_from:
        p.error('previous-readme requires update-from')
    cache=configure_xet_cache(a.xet_cache) if not (a.login_only or a.verify_only) else None
    from huggingface_hub import HfApi,hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError,EntryNotFoundError
    api=HfApi()
    if a.login_only:
        authorize(api);return
    if a.finish_after_pid:
        if a.finish_after_pid==os.getpid():
            p.error('Cannot wait for this same process')
        process=Path('/proc')/str(a.finish_after_pid)
        announced=False
        while True:
            try:
                command=(process/'cmdline').read_bytes().split(b'\0')
                state=(process/'stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:
                break
            if state=='Z' or not any(arg.endswith(b'/publish_hf_dataset.py') for arg in command):
                break
            if not announced:
                print(json.dumps({'type':'waiting_for_existing_upload','pid':a.finish_after_pid}),flush=True)
                announced=True
            time.sleep(10)
    locks=[]
    lock_folders={folder}
    if a.update_from:
        a.update_from=a.update_from.resolve()
        if not a.update_from.is_relative_to(Path('/mnt/d')) or a.update_from==folder:
            p.error('update-from must be a separate frozen release directory on D:')
        lock_folders.add(a.update_from)
    for directory in sorted(lock_folders):
        lock=(directory.parent/(directory.name+'-publish.lock')).open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An upload/verification process already owns this release') from None
        locks.append(lock)
    files=inventory(folder)
    previous_files=None
    if a.update_from:
        previous_files=inventory(a.update_from)
        if a.previous_readme:
            content=a.previous_readme.read_bytes()
            replacement={'path':'README.md','bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),
                'git_blob_sha1':hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()}
            previous_files=[replacement if f['path']=='README.md' else f for f in previous_files]
        append_membership(json.loads((a.update_from/'meta/snapshot.json').read_text()),
            json.loads((folder/'meta/snapshot.json').read_text()))
        validate_update_files(previous_files,files)
    authorize(api)
    if not a.verify_only:
        try:
            existing=api.dataset_info(REPO,files_metadata=True)
        except RepositoryNotFoundError:
            existing=None
        if existing:
            if existing.private:
                raise ValueError('Existing repository is private; do not change visibility implicitly')
            if previous_files is not None:
                verify_update_release(existing,previous_files,files)
            else:
                try:
                    previous=Path(hf_hub_download(REPO,'meta/snapshot.json',repo_type='dataset'))
                except EntryNotFoundError:
                    verify_partial_release(existing,files)
                else:
                    if json.loads(previous.read_text())!=json.loads((folder/'meta/snapshot.json').read_text()):
                        raise ValueError('Remote has a different frozen snapshot')
        elif previous_files is not None:
            raise ValueError('Cannot append: the published repository does not exist')
        api.create_repo(REPO,repo_type='dataset',private=False,exist_ok=True)
        print(json.dumps({'type':'upload_start','repo_id':REPO,'files':len(files),
                          'bytes':sum(f['bytes'] for f in files)}),flush=True)
        upload_batches(api,folder,files,existing,workers=a.workers,batch_size=a.upload_batch_size,
            cache=cache,previous=previous_files)
    report=verify_remote(api,files)
    dump(folder.parent/(folder.name+'-upload-result.json'),report)
    print(json.dumps({'type':'upload_verified',**report}),flush=True)


if __name__=='__main__':
    main()
