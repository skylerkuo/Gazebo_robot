"""Independent scene/recording identities for a shared expert simulator session."""
import copy
import json
from pathlib import Path
import shutil


def make_expert_trials(source,profile,max_task_seconds,count,scene_attempt=None):
    from run_sfp0_multiview_gazebo import inference_scene
    if not isinstance(count,int) or count<1:raise ValueError('Batch size must be positive')
    engine=copy.deepcopy(source);engine['trials']={};episodes=[]
    for i in range(count):
        seed=None if scene_attempt is None else scene_attempt+i
        single,scene=inference_scene(source,profile,max_task_seconds,seed)
        trial=single['trials']['trial_1'];task=trial['tasks']['task_1']
        trial_id='trial_1' if count==1 else f'episode_{i:06d}'
        task_id='task_1' if count==1 else trial_id
        trial['tasks']={task_id:task};engine['trials'][trial_id]=trial
        episodes.append(dict(scene,trial_id=trial_id,task_id=task_id))
    return engine,episodes


def import_expert_session(root,run,*,status=None):
    """Only official whole-session results qualify; re-import is trial-idempotent."""
    from human_dataset import import_human_run
    run=Path(run).resolve()
    manifest=json.loads((run/'run_manifest.json').read_text())
    settings=json.loads((run/'settings.json').read_text())
    if manifest.get('method')!='AutoExpertACT' or manifest.get('prepare_only'):
        raise ValueError('Select an executed automatic-expert session')
    if status is None:
        status=manifest.get('launcher_exit_code')
        if status is None:
            from sideview_engine_done import completed
            try:finished=completed(run)
            except (OSError,ValueError):finished=False
            if not finished:raise ValueError('Session has no recorded exit status or complete official scores; inspect the interrupted session first')
            status=0
    episodes=settings.get('recording_episodes',[{'trial_id':'trial_1','task_id':'task_1'}])
    entries=[]
    for episode in episodes:
        entry=import_human_run(root,run,settings['collection_settings'],
            manifest['source_contract']['collection_root'],status=status,expert_source='expert',
            trial_id=episode['trial_id'],task_id=episode['task_id'])
        entries.append(entry)
        print(json.dumps({'type':'expert_episode_import','run':str(run),'trial':entry['trial'],
            'accepted':entry['accepted'],'outcome':entry.get('outcome'),
            'controller_version':entry.get('controller_version'),
            'expert_anchors':len(entry.get('filtered_expert_anchor_frames',[])),
            'rollout_training_anchors':len(entry.get('rollout_train_anchor_frames',[])),
            'official_insertion_score':entry.get('official_score',{}).get('tier_3',{}).get('score'),
            'reason':entry.get('rejection_reason'),'dataset':str(Path(root).resolve())}),flush=True)
    return entries


def pending_expert_batches(root):
    """Do not silently skip a batch that finished before its import was committed."""
    root=Path(root);path=root/'_collection/manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
    known={(str(Path(e['run']).resolve()),e.get('trial','trial_1')) for e in manifest['attempts']}
    pending=[]
    for path in sorted((root/'runs').glob('*/run_manifest.json')):
        value=json.loads(path.read_text())
        if value.get('method')!='AutoExpertACT' or value.get('batch_size',1)<=1 or value.get('prepare_only'):continue
        episodes=value.get('recording_episodes',[])
        if any((str(path.parent.resolve()),e['trial_id']) not in known for e in episodes):pending.append(path.parent)
    return pending


def prune_scoring_bags(run):
    """After verified official completion keep scores/proof, discard bulky bags."""
    from sideview_engine_done import completed
    from sideview_data import digest_file,json_write
    run=Path(run).resolve()
    if not completed(run):raise ValueError('Cannot remove scoring bags before all official trial scores are complete')
    scoring=run/'scoring';removed=[]
    for bag in sorted(scoring.glob('bag_*')):
        if not bag.is_dir():continue
        if bag.is_symlink():raise ValueError('Refusing a symlinked scoring bag')
        size=sum(p.stat().st_size for p in bag.rglob('*') if p.is_file())
        shutil.rmtree(bag);removed.append({'name':bag.name,'bytes':size})
    report={'official_scores_sha256':digest_file(scoring/'scoring.yaml'),
        'removed_bags':removed,'reclaimed_bytes':sum(p['bytes'] for p in removed),
        'score_yaml_and_raw_recordings_preserved':True}
    json_write(scoring/'bag_cleanup.json',report)
    return report
