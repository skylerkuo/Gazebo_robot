"""Audit/export human-assisted rollouts; model/hold actions never become expert labels."""
import fcntl
import json
import re
from pathlib import Path

import numpy as np
import yaml

from sideview_data import TASK,export_episode,json_write,successful_score
from human_control import expert_anchor_frames
from idle_sampling import DEFAULT_IDLE_FILTER,filter_idle_anchors

HUMAN_FORMAT='aic-sfp0-human-interventions-v1'
EXPERT_FORMAT='aic-sfp0-official-expert-interventions-v1'
DEFAULT_HUMAN_ROOT='/mnt/d/robot_datasets/aic_sfp0_human_v1'
DATA_DRIVE_ROOT=Path('/mnt/d')


def import_receipt_path(run,expert_source,trial_id='trial_1'):
    name='human' if expert_source=='human' else 'expert'
    if trial_id=='trial_1':return Path(run)/f'{name}_import.json'
    return Path(run)/f'{name}_imports'/f'{trial_id}.json'


def import_human_run(root,run,profile,parent_collection,*,status,operator_discarded=False,test_run=False,expert_source='human',retry_rejected=False,trial_id='trial_1',task_id='task_1'):
    from collect_sfp0_sideview import refresh_metadata
    import pyarrow.parquet as pq
    if expert_source not in ('human','expert'):raise ValueError('Unknown correction provider')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',s) for s in (trial_id,task_id)):raise ValueError('Invalid trial/task recording name')
    format_name=HUMAN_FORMAT if expert_source=='human' else EXPERT_FORMAT
    root=Path(root).resolve();run=Path(run).resolve()
    if not root.is_relative_to(DATA_DRIVE_ROOT):raise ValueError('Human dataset must be on D:')
    directory=root/'_collection';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'import.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    path=directory/'manifest.json'
    settings=dict(profile,minimum_frames=21)
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['format']!=format_name or manifest['parent_collection_root']!=str(Path(parent_collection).resolve()):
            raise ValueError('Human dataset belongs to another collection')
        if any(manifest['settings'].get(k)!=v for k,v in settings.items() if k!='real_time_factor'):
            raise ValueError('Human recording camera/timing settings changed')
    else:
        manifest={'format':format_name,'expert_source':expert_source,'settings':settings,'task':TASK,'attempts':[],
            'ground_truth_supervision':expert_source=='expert','model_ground_truth_inputs':False,
            'parent_collection_root':str(Path(parent_collection).resolve()),
            'action_frame':'base_link','action_format':'absolute TCP xyz+xyzw',
            'training_rule':f'Only successful, non-test {expert_source} future-20 chunks in one control segment; no model/hold labels.'}
    known=next((e for e in manifest['attempts'] if e['run']==str(run) and e.get('trial','trial_1')==trial_id),None)
    if known and known.get('recording_task_id','task_1')!=task_id:raise ValueError('Trial recording task changed')
    if known and (known.get('accepted') or not retry_rejected):return known
    accepted=[e for e in manifest['attempts'] if e.get('accepted')]
    episode=len(accepted);offset=sum(e['validation']['frames'] for e in accepted)
    entry={'attempt':known['attempt'] if known else len(manifest['attempts']),'run':str(run),'trial':trial_id,
        'recording_task_id':task_id,'accepted':False,
        'operator_discarded':operator_discarded,'test_run':test_run,'recording_type':'model_and_'+expert_source}
    destination=None;own_export=False
    try:
        scoring=yaml.safe_load((run/'scoring/scoring.yaml').read_text()) or {}
        entry['official_score']=scoring.get(trial_id,{})
        summary=json.loads((run/'raw'/f'{task_id}.json').read_text());entry['recorder']=summary
        if summary.get('controller_version'):
            entry['controller_version']=summary['controller_version']
            manifest['active_controller_version']=summary['controller_version']
        direct_expert=expert_source=='expert' and summary.get('controller')=='official_cheatcode_only'
        if direct_expert:
            entry['recording_type']='official_cheatcode_only'
            entry['controller_version']='official_cheatcode_only_v1'
        if expert_source=='expert' and summary.get('controller') not in ('model_and_official_expert','official_cheatcode_only'):
            raise ValueError('Expected explicitly recorded official-expert supervision')
        if status!=0:raise ValueError('Session did not finish normally; raw recording retained')
        if operator_discarded or summary.get('operator_discarded'):raise ValueError('Operator discarded this episode')
        if test_run:raise ValueError('Automated test recording is excluded from training')
        if not successful_score(entry['official_score']):raise ValueError('Official evaluator did not confirm successful insertion')
        if not summary.get('policy_return') or summary.get('writer_error'):
            raise ValueError('Human recording did not finish successfully')
        calibration=json.loads((run/'raw'/f'{task_id}_calibration.json').read_text())
        if (calibration['side_camera']!=settings['camera'] or
                (calibration['record_width'],calibration['record_height'])!=(settings['record_width'],settings['record_height'])):
            raise ValueError('Human recording camera calibration differs from the source collection')
        destination=root/f'data/chunk-000/file-{episode:03d}.parquet'
        destination.parent.mkdir(parents=True,exist_ok=True)
        origin=destination.with_suffix('.origin.json')
        if destination.exists():
            provenance=json.loads(origin.read_text()) if origin.exists() else {}
            if (provenance.get('run')!=str(run) or provenance.get('trial','trial_1')!=trial_id
                    or provenance.get('task_id','task_1')!=task_id):
                raise ValueError('Uncommitted export exists from another run; recover that run first')
            recovery=destination.with_suffix('.recovery.parquet')
            report=export_episode(run/'raw'/f'{task_id}.sqlite3',recovery,episode,offset,settings,
                                  command_provenance=True,allow_gaps=True)
            from sideview_data import digest_file
            if digest_file(destination)!=digest_file(recovery):raise ValueError('Existing export differs from this raw recording')
            recovery.unlink()
        else:
            json_write(origin,{'run':str(run),'trial':trial_id,'task_id':task_id,'episode_index':episode,'offset':offset})
            own_export=True
            report=export_episode(run/'raw'/f'{task_id}.sqlite3',destination,episode,offset,settings,
                              command_provenance=True,allow_gaps=True)
        table=pq.read_table(destination,columns=['action_source','action_segment','human_active','simulation_time',
                                               'action','observation.environment_state']).to_pydict()
        if direct_expert and (set(table['action_source'])!={'expert'} or not all(table['human_active'])):
            raise ValueError('Controller-only recording must contain exclusively active official-expert targets')
        anchors=expert_anchor_frames(table['action_source'],table['action_segment'],table['human_active'],table['simulation_time'],source=expert_source)
        report['expert_anchor_count']=len(anchors)
        report['control_source_counts']={name:table['action_source'].count(name) for name in set(table['action_source'])}
        entry['validation']=report
        filtered,sampling=filter_idle_anchors(anchors,table['action'],table['observation.environment_state'],table['simulation_time'])
        entry.update(filtered_expert_anchor_frames=filtered,idle_filter=dict(DEFAULT_IDLE_FILTER),sampling_report=sampling)
        report['training_anchor_count']=len(filtered)
        if expert_source=='expert':
            from rollout_sampling import ROLLOUT_FILTER,sample_rollout_file
            manifest['rollout_filter']=dict(ROLLOUT_FILTER)
            manifest['training_rule']='Treat moving model and expert recovery targets equally; boolean mask excludes stationary interiors and hold. VAL uses expert-only windows.'
            _,_,full_train,expert_val,full_report=sample_rollout_file(destination)
            entry.update(rollout_train_anchor_frames=full_train,expert_validation_frames=expert_val,
                rollout_sampling_report=full_report,rollout_filter=dict(ROLLOUT_FILTER),model_motion_preserved=True)
            report['rollout_training_anchor_count']=len(full_train)
        if expert_source=='expert':
            if not full_train:raise ValueError('Successful episode has no usable moving future-20 windows')
            entry['outcome']=('expert_direct_success' if direct_expert else
                'expert_assisted_success' if report['control_source_counts'].get('expert',0) else 'model_success')
            entry['validation_eligible']=bool(expert_val)
        elif not anchors:
            # Keep the audit Parquet beside the run, outside trainable data/.
            destination.replace(run/('no_expert_windows.parquet' if trial_id=='trial_1' else f'no_expert_windows_{trial_id}.parquet'))
            raise ValueError('No complete human future-20 windows; keep human mode for >1 simulated second')
        entry.update(accepted=True,episode_index=episode,parquet=str(destination.relative_to(root)),
            expert_anchor_frames=anchors)
    except Exception as error:
        if own_export and destination is not None and destination.exists():
            destination.replace(run/('rejected_export.parquet' if trial_id=='trial_1' else f'rejected_export_{trial_id}.parquet'))
        entry['rejection_reason']=f'{type(error).__name__}: {error}'
    if known:
        entry['previous_rejection_reason']=known.get('rejection_reason')
        manifest['attempts'][manifest['attempts'].index(known)]=entry
    else:manifest['attempts'].append(entry)
    refresh_metadata(root,manifest,settings)
    info=json.loads((root/'meta/info.json').read_text())
    info['features'].update(action_source={'dtype':'string','shape':[1],'names':None},
        action_segment={'dtype':'int64','shape':[1],'names':None},human_active={'dtype':'bool','shape':[1],'names':None})
    json_write(root/'meta/info.json',info)
    json_write(import_receipt_path(run,expert_source,trial_id),entry)
    return entry


def recover_successful_runs(root):
    """Re-audit official successes rejected solely for a missing expert segment."""
    root=Path(root).resolve();path=root/'_collection/manifest.json'
    if not path.exists():return []
    manifest=json.loads(path.read_text())
    if manifest['format']!=EXPERT_FORMAT:raise ValueError('Recovery requires the model+official-expert collection')
    candidates=[e for e in manifest['attempts'] if not e.get('accepted')
        and 'No complete human future-20 windows' in e.get('rejection_reason','')
        and successful_score(e.get('official_score',{}))]
    return [import_human_run(root,e['run'],manifest['settings'],manifest['parent_collection_root'],
        status=0,expert_source='expert',retry_rejected=True,trial_id=e.get('trial','trial_1'),
        task_id=e.get('recording_task_id','task_1')) for e in candidates]
