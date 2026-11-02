#!/usr/bin/env python3
"""Collect 100 direct successes, scratch ACT, frozen ACT + independently random Flow.

All job state/data/checkpoints live on D:. No upload code is invoked.
Safe to restart: completed trials, frozen indexes and last.pt are retained.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
JOB=Path('/mnt/d/robot_tools/aic_direct100_fresh_20261030')
SOURCE=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1')
DIRECT=Path('/mnt/d/robot_datasets/aic_sfp0_direct_success_100_v1')
RECOVERY=Path('/mnt/d/robot_datasets/aic_sfp0_recovery_only_v1')
ACT=Path('/mnt/d/robot_models/aic_sfp0_act_direct100_scratch_v1')
FLOW=Path('/mnt/d/robot_models/aic_sfp0_flow_recovery_direct100_scratch_v1')
sys.path.insert(0,str(ROOT/'tools/dataset_python'))


def read(path):
    for attempt in range(8):
        try:return json.loads(Path(path).read_text())
        except (OSError,json.JSONDecodeError):
            if attempt==7:raise
            time.sleep(.2)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def alive(pid,script=None):
    try:
        proc=Path('/proc')/str(pid)
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        args=(proc/'cmdline').read_bytes().split(b'\0')
        return state!='Z' and (script is None or any(a.endswith(('/'+script).encode()) for a in args))
    except (FileNotFoundError,ProcessLookupError):return False


def counts(root=SOURCE):
    manifest=read(root/'_collection/manifest.json')
    accepted=[e for e in manifest['attempts'] if e.get('accepted')]
    direct=[e for e in accepted if e.get('recorder',{}).get('controller')=='official_cheatcode_only']
    return {'accepted':len(accepted),'direct':len(direct),'attempts':len(manifest['attempts'])}


def recent_metrics(root):
    path=root/'metrics.jsonl'
    if not path.exists():return {}
    with path.open('rb') as f:
        f.seek(0,2);f.seek(max(0,f.tell()-300000));lines=f.read().decode(errors='replace').splitlines()
    result={}
    for line in lines:
        try:item=json.loads(line)
        except ValueError:continue
        if item.get('type') in ('train','resume','start'):result['latest_step']=item.get('step')
        if item.get('type')=='validation':
            result['validation']={k:item[k] for k in ('step','loss','position_l2_mm','position_xyz_mae_mm','rotation_deg','samples') if k in item}
        if item.get('type')=='best_checkpoint' and item.get('checkpoint')=='best.pt':result['best']=item
    return result


def stage_config(policy,job=JOB):
    name='sfp0_act_direct100_scratch.json' if policy=='act' else 'sfp0_flow_recovery_direct100_scratch.json'
    value=read(ROOT/'configs'/name)
    if policy=='flow':value['init_from']=str(job/'frozen_act/act.pt')
    return value


class Pipeline:
    def __init__(self,job,monitor_seconds=1200):
        self.job=job;self.monitor_seconds=monitor_seconds;self.child=None
        self.state=read(job/'state.json') if (job/'state.json').exists() else {'status':'running','phase':'starting','stages':{}}
        self.last_monitor=0
        self.state.update(pid=os.getpid(),status='running',heartbeat_unix=time.time())
        self.state.pop('error',None)

    def event(self,kind,**data):
        item={'time_utc':datetime.now(timezone.utc).isoformat(),'type':kind,**data}
        with (self.job/'events.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)

    def tick(self,force=False):
        now=time.time();self.state['heartbeat_unix']=now
        write(self.job/'state.json',self.state)
        if force or now-self.last_monitor>=self.monitor_seconds:
            snapshot={'time_utc':datetime.now(timezone.utc).isoformat(),**self.state,
                'collection':counts(),'act':recent_metrics(ACT),'flow':recent_metrics(FLOW),
                'c_free_gib':round(shutil.disk_usage('/mnt/c').free/2**30,2),
                'd_free_gib':round(shutil.disk_usage('/mnt/d').free/2**30,2)}
            try:
                gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
                snapshot['gpu']=gpu.stdout.strip()
            except (OSError,subprocess.TimeoutExpired):snapshot['gpu']='unavailable'
            write(self.job/'monitor_latest.json',snapshot)
            with (self.job/'monitor.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
            self.last_monitor=now
        if (self.job/'STOP').exists():raise InterruptedError('Operator STOP file requested')

    def phase(self,name):
        self.state.update(phase=name,child_pid=None)
        self.event('phase',phase=name);self.tick(force=True)

    def run(self,command,log,*,timeout=None):
        started=time.monotonic()
        with log.open('ab',buffering=0) as output:
            self.child=subprocess.Popen(command,cwd=WORKSPACE,stdin=subprocess.DEVNULL,
                stdout=output,stderr=output,start_new_session=True)
            self.state.update(child_pid=self.child.pid,command=command,log=str(log))
            self.event('child_started',phase=self.state['phase'],pid=self.child.pid,command=command)
            try:
                while self.child.poll() is None:
                    self.tick()
                    if timeout and time.monotonic()-started>timeout:raise TimeoutError('Stage wall-time exceeded')
                    if time.monotonic()-started>3600 and log.exists() and time.time()-log.stat().st_mtime>3600:
                        raise TimeoutError('No stage log activity for more than one hour')
                    time.sleep(10)
                status=self.child.wait()
            except BaseException:
                if self.child.poll() is None:
                    os.killpg(self.child.pid,signal.SIGINT)
                    try:self.child.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid,signal.SIGTERM);self.child.wait(timeout=30)
                raise
            finally:self.child=None
        self.event('child_finished',phase=self.state['phase'],exit_code=status)
        self.state['child_pid']=None;self.tick(force=True)
        return status

    def recover_pending(self):
        from collect_sfp0_direct import pending_direct_runs,import_direct_run
        from sideview_engine_done import completed
        from expert_batch import prune_scoring_bags
        with (ROOT/'runtime/session.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for run in pending_direct_runs(SOURCE):
                if not (run/'launcher.json').exists():
                    try:finished=completed(run)
                    except (OSError,ValueError,RuntimeError):finished=False
                    status=0 if finished else 130
                    write(run/'launcher.json',{'exit_code':status,'interrupted_or_timed_out':not finished,
                        'recovered':True,'evidence':'official complete engine and scores' if finished else
                        'no complete official result; preserve raw and reject interrupted trials'})
                status=read(run/'launcher.json')['exit_code']
                entries=import_direct_run(SOURCE,run)
                if status==0:prune_scoring_bags(run)
                self.event('recovered_collection',run=str(run),accepted=sum(e['accepted'] for e in entries),exit_code=status)

    def collect(self):
        self.phase('collection')
        receipt=self.job/'initial_collection.json'
        if receipt.exists():
            pid=read(receipt)['pid']
            try:
                while alive(pid,'collect_sfp0_direct.py'):
                    self.state.update(child_pid=pid,log=str(self.job/'collection.log'),adopted_initial_collector=True)
                    self.tick();time.sleep(10)
            except InterruptedError:
                if alive(pid,'collect_sfp0_direct.py'):
                    os.kill(pid,signal.SIGINT)
                    deadline=time.monotonic()+180
                    while alive(pid,'collect_sfp0_direct.py') and time.monotonic()<deadline:time.sleep(1)
                raise
        for attempt in range(1,11):
            current=counts()
            if current['direct']==100:
                self.state['stages']['collection']=current;return
            if current['direct']>100:raise ValueError('More than 100 direct episodes; do not silently change training membership')
            try:self.recover_pending()
            except BlockingIOError:
                self.event('waiting_for_simulator_lock');time.sleep(60);continue
            current=counts();target=current['accepted']+100-current['direct']
            command=[sys.executable,'-u',str(ROOT/'scripts/collect_sfp0_direct.py'),
                '--output-dir',str(SOURCE),'--episodes',str(target),'--max-attempts',str(max(500,current['attempts']+200)),
                '--batch-size','5','--real-time-factor','0.5','--gui','false']
            status=self.run(command,self.job/'collection.log')
            if status:self.event('collection_retry',attempt=attempt,exit_code=status)
        raise RuntimeError('Collection retry budget exhausted; raw data and scores retained')

    def curate(self):
        self.phase('prepare_direct100')
        command=[sys.executable,'-u',str(ROOT/'scripts/prepare_sfp0_direct_act.py'),
                 '--dataset-root',str(SOURCE),'--output-dir',str(DIRECT)]
        if self.run(command,self.job/'prepare.log')!=0:raise RuntimeError('Direct curation failed')
        from prepare_sfp0_direct_act import select_direct,select_recovery
        direct=select_direct(read(DIRECT/'_collection/manifest.json'))
        if len(direct)!=100:raise ValueError('ACT requires exactly 100 valid direct episodes')
        recovery=select_recovery(read(RECOVERY/'_collection/manifest.json'))
        if len(recovery)<2:raise ValueError('Recovery set is missing')
        self.state['stages']['curation']={'direct_episodes':100,'recovery_episodes':len(recovery),
            'direct_episode_ids':[e['episode_index'] for e in direct],
            'recovery_episode_ids':[e['episode_index'] for e in recovery]}
        self.tick(force=True)

    def checkpoint_status(self,root,steps,policy):
        import torch
        last=root/'checkpoints/last.pt'
        if not last.exists():return None
        value=torch.load(last,map_location='cpu',weights_only=True)
        if value.get('smoke_test') or value['config']['policy_type']!=policy:
            raise ValueError('Unexpected checkpoint in the new experiment output')
        if value['config']['output_dir']!=str(root):raise ValueError('Checkpoint output identity differs')
        expected_data=DIRECT if policy=='act' else RECOVERY
        if value['config']['dataset_root']!=str(expected_data):raise ValueError('Checkpoint dataset identity differs')
        if policy=='act':
            init=value.get('initialization') or {}
            if init.get('mode')!='from_scratch' or value['config']['pretrained_backbone']:
                raise ValueError('ACT is not a true random initialization experiment')
            if value['config'].get('init_from') is not None:raise ValueError('Scratch ACT has an initialization checkpoint')
        else:
            if value['config'].get('flow_encoder_init_from_act',True) or value['config']['pretrained_backbone']:
                raise ValueError('Flow did not start with fully random trainable weights')
            origin=read(self.job/'frozen_act/act.origin.json')
            if value['config'].get('base_act_sha256')!=origin['sha256']:
                raise ValueError('Flow is not paired with this newly trained ACT')
        if value['step']>=steps and (value.get('validation_step')!=value['step'] or
                value.get('metrics',{}).get('samples')!=1500):
            raise ValueError('Final checkpoint lacks the expected complete validation')
        return {'step':value['step'],'complete':value['step']>=steps,'validation_step':value.get('validation_step')}

    def train(self,policy):
        self.phase('train_'+policy)
        config=stage_config(policy,self.job)
        config_path=self.job/(policy+'_config.json');write(config_path,config)
        root=ACT if policy=='act' else FLOW
        script='train_sfp0_act_direct.py' if policy=='act' else 'train_sfp0_flow_recovery.py'
        for attempt in range(1,6):
            checkpoint=self.checkpoint_status(root,config['steps'],config['policy_type'])
            if checkpoint and checkpoint['complete']:
                best=root/'checkpoints/best.pt'
                if not best.exists():raise RuntimeError('Training finished without BEST')
                self.state['stages'][policy]={'output':str(root),'last':str(root/'checkpoints/last.pt'),
                    'best':str(best),**checkpoint};self.tick(force=True);return
            if attempt==5:break
            if checkpoint is None and root.exists() and any(root.iterdir()):
                archive=root.with_name(root.name+'.interrupted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
                root.rename(archive);self.event('preserved_incomplete_start',archive=str(archive))
            command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--config',str(config_path)]
            if checkpoint:command+=['--resume',str(root/'checkpoints/last.pt')]
            elif policy=='act':command+=['--from-scratch']
            status=self.run(command,self.job/(policy+'.log'))
            if status:self.event('training_retry',policy=policy,attempt=attempt,exit_code=status)
        raise RuntimeError(policy+' training retry budget exhausted')

    def execute(self):
        confirmation=read(Path('/mnt/d/robot_datasets/_hf_release/aic-cable-insertion-gazebo-sfp0-upload-result.json'))
        if not confirmation.get('public') or confirmation.get('parquet_files')!=211 or confirmation.get('verified_files')!=650:
            raise ValueError('Published dataset verification receipt missing')
        self.state['published_dataset_verified']=confirmation
        self.collect();self.curate();self.train('act')
        self.phase('freeze_new_act')
        from prepare_sfp0_direct_act import freeze_act_seed
        seed=freeze_act_seed(ACT/'checkpoints/best.pt',self.job/'frozen_act/act.pt')
        self.state['stages']['frozen_act']=seed
        self.train('flow')
        self.state.update(status='complete',phase='complete',child_pid=None)
        self.event('pipeline_complete',act=self.state['stages']['act'],flow=self.state['stages']['flow'])
        self.tick(force=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir',type=Path,default=JOB)
    parser.add_argument('--monitor-seconds',type=int,default=1200)
    args=parser.parse_args();job=args.job_dir.resolve()
    if not job.is_relative_to(Path('/mnt/d')) or args.monitor_seconds<10:parser.error('Use a D: job folder and monitoring >=10s')
    job.mkdir(parents=True,exist_ok=True)
    lock=(job/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This pipeline is already running') from None
    pipeline=Pipeline(job,args.monitor_seconds)
    # Per-process/inherited setting only; piped WSL core captures ignore ulimit -c.
    # Keep unattended runtime crashes from generating GiB dumps on Windows C:.
    try:Path('/proc/self/coredump_filter').write_text('0')
    except OSError:pass
    def stop(signum,_):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:pipeline.execute()
    except BaseException as error:
        pipeline.state.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',
                              error=f'{type(error).__name__}: {error}',child_pid=None)
        pipeline.event('pipeline_error',error=pipeline.state['error'],traceback=traceback.format_exc())
        # Write even when an operator STOP exists; do not raise again from tick.
        pipeline.state['heartbeat_unix']=time.time();write(job/'state.json',pipeline.state)
        raise


if __name__=='__main__':main()
