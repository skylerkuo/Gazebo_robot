#!/usr/bin/env python3
"""Resume a prepared dataset update until uploaded, then verify anonymous access."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from publish_hf_dataset import REPO, PROJECT, inventory, verify_remote
from prepare_hf_release import dump


def matching_uploader(pid, folder):
    try:
        proc=Path('/proc')/str(pid)
        args=(proc/'cmdline').read_bytes().split(b'\0')
        state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
        return state!='Z' and os.fsencode(str(folder)) in args and any(
            arg.endswith(b'/publish_hf_dataset.py') for arg in args)
    except (FileNotFoundError,ProcessLookupError):
        return False


def valid_report(folder, expected):
    path=folder.parent/(folder.name+'-upload-result.json')
    if not path.exists():return False
    data=json.loads(path.read_text())
    return (data.get('repo_id')==REPO and data.get('public') is True and bool(data.get('revision'))
        and data.get('verified_files')==len(expected)
        and data.get('parquet_files')==sum(f['path'].startswith('data/') for f in expected)
        and data.get('bytes')==sum(f['bytes'] for f in expected))


def verify_public(folder, files):
    import httpx
    from huggingface_hub import HfApi
    report=verify_remote(HfApi(token=False),files)
    revision=report['revision']
    url=f'https://huggingface.co/datasets/{REPO}'
    with httpx.Client(follow_redirects=True,timeout=90) as client:
        response=client.get(f'{url}/raw/{revision}/meta/info.json');response.raise_for_status()
        info=response.json()
        local=json.loads((folder/'meta/info.json').read_text())
        if any(info[k]!=local[k] for k in ('total_episodes','total_frames')):
            raise ValueError('Public dataset counts differ from the prepared update')
        response=client.get(f'{url}/raw/{revision}/README.md');response.raise_for_status()
        if response.content!=(folder/'README.md').read_bytes():
            raise ValueError('Public README differs from the reviewed dataset card')
        latest=sorted(f['path'] for f in files if f['path'].startswith('data/'))[-1]
        with client.stream('GET',f'{url}/resolve/{revision}/{latest}',headers={'Range':'bytes=0-3'}) as response:
            response.raise_for_status()
            magic=next(response.iter_bytes(chunk_size=4))
            if magic!=b'PAR1':raise ValueError('Anonymous Parquet download has an invalid header')
            status=response.status_code
    report.update(episodes=info['total_episodes'],frames=info['total_frames'],
        anonymous_access=True,readme_matches=True,download_checked=latest,
        download_http_status=status,parquet_header='PAR1')
    dump(folder.parent/(folder.name+'-public-validation.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--update-from',type=Path,required=True)
    parser.add_argument('--previous-readme',type=Path)
    parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('--adopt-pid',type=int)
    parser.add_argument('--max-retries',type=int,default=20)
    args=parser.parse_args()
    args.folder=args.folder.resolve();args.update_from=args.update_from.resolve()
    args.job_dir=args.job_dir.resolve()
    if not all(p.is_relative_to(Path('/mnt/d')) for p in (args.folder,args.update_from,args.job_dir)):
        parser.error('Keep the release and job files on D:')
    if args.max_retries<1:parser.error('max-retries must be positive')
    args.job_dir.mkdir(parents=True,exist_ok=True)
    lock=(args.job_dir/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=inventory(args.folder)
    def state(phase,**extra):
        data={'phase':phase,'repo_id':REPO,'pid':os.getpid(),'updated_at_unix':time.time(),**extra}
        progress=args.folder.parent/(args.folder.name+'-progress.json')
        if progress.exists():data['upload']=json.loads(progress.read_text())
        dump(args.job_dir/'state.json',data)
        print(json.dumps(data),flush=True)
    if args.adopt_pid:
        while matching_uploader(args.adopt_pid,args.folder):
            state('uploading',uploader_pid=args.adopt_pid,adopted=True)
            time.sleep(30)
    command=[sys.executable,'-u',str(PROJECT/'scripts/publish_hf_dataset.py'),
        '--folder',str(args.folder),'--update-from',str(args.update_from),
        '--upload-batch-size','32','--workers','4']
    if args.previous_readme:command.extend(('--previous-readme',str(args.previous_readme)))
    env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1',HF_XET_HIGH_PERFORMANCE='1')
    force_upload=False
    for attempt in range(1,args.max_retries+1):
        try:
            if force_upload or not valid_report(args.folder,files):
                with (args.job_dir/f'upload-{attempt:02d}.log').open('a') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    while process.poll() is None:
                        state('uploading',uploader_pid=process.pid,attempt=attempt)
                        time.sleep(30)
                if process.returncode or not valid_report(args.folder,files):
                    raise RuntimeError(f'Uploader exited {process.returncode}; next attempt will resume')
                force_upload=False
            state('verifying_public_access',attempt=attempt)
            result=verify_public(args.folder,files)
            state('complete',result=result)
            return
        except Exception as error:
            force_upload=True
            state('retry_wait',attempt=attempt,error=str(error))
            if attempt==args.max_retries:
                state('failed',attempt=attempt,error=str(error));raise
            time.sleep(min(attempt*30,300))


if __name__=='__main__':
    main()
