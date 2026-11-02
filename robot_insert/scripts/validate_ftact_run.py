#!/usr/bin/env python3
"""Validate actual learned-policy observations, predictions, commands, and official scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def temporal_ensemble_checks(predictions, coefficient, rate_hz=20):
    """Independently verify the time alignment and ACT weighted mean in a run log."""
    checks = {key: bool(predictions) for key in ('act_exponential_weights',
        'ensemble_same_execution_time', 'ensemble_matches_weighted_mean',
        'new_prediction_for_each_command')}
    checks['overlapping_predictions_combined'] = any(r.get('ensemble_count', 0) > 1 for r in predictions)
    last_ticks = {}
    for row in predictions:
        try:
            count = row['ensemble_count']
            weights = np.asarray(row['ensemble_weights'], dtype=float)
            poses = np.asarray(row['ensemble_source_poses'], dtype=float)
            stamps = np.asarray(row['ensemble_observation_stamps'], dtype=float)
            indices = np.asarray(row['ensemble_source_indices'], dtype=int)
            ids = np.asarray(row['ensemble_inference_ids'], dtype=int)
            if not (1 <= count <= 20 and weights.shape == stamps.shape == indices.shape == ids.shape == (count,)
                    and poses.shape == (count, 7) and np.isfinite(poses).all()
                    and ((0 <= indices) & (indices < 20)).all()):
                raise ValueError('Invalid ensemble evidence dimensions')
            expected_weights = np.exp(-coefficient * np.arange(count))
            expected_weights /= expected_weights.sum()
            checks['act_exponential_weights'] &= bool(np.allclose(weights, expected_weights, rtol=1e-10, atol=1e-12))
            target_time = row['ensemble_target_time']
            expected_tick = int(np.floor((row['simulation_time']-row['ensemble_origin_time'])*rate_hz+1e-6)) + 1
            group = (row['policy_instance'], row['task_id'],row.get('control_segment',0))
            checks['ensemble_same_execution_time'] &= bool(
                np.allclose(stamps+(indices+1)/rate_hz, target_time, rtol=0, atol=.005)
                and (np.diff(stamps)>0).all() and (np.diff(ids)>0).all()
                and row['ensemble_target_tick'] == expected_tick
                and np.isclose(target_time, row['ensemble_origin_time']+expected_tick/rate_hz, rtol=0, atol=1e-8)
                and expected_tick > last_ticks.get(group, -1))
            last_ticks[group] = expected_tick
            aligned = poses.copy()
            aligned[(aligned[:, 3:]*aligned[0, 3:]).sum(1)<0, 3:] *= -1
            expected = (aligned*expected_weights[:, None]).sum(0)
            expected[3:] /= np.linalg.norm(expected[3:])
            checks['ensemble_matches_weighted_mean'] &= bool(np.allclose(expected, row['raw_target'], rtol=1e-8, atol=1e-9))
            checks['new_prediction_for_each_command'] &= bool(row['new_chunk'] and ids[-1] == row['inference_id'])
        except (KeyError, ValueError, TypeError, IndexError):
            for key in checks:
                if key != 'overlapping_predictions_combined':
                    checks[key] = False
    return checks


def flow_receding_horizon_checks(predictions, chunks, loaded, manifest):
    """Check the generated sample, its physical target time, and FM sampler contract."""
    settings = manifest['settings']
    checks = dict(only_sfp0_trial=manifest['trials'] == ['trial_1'],
                  flow_sampler_contract=bool(loaded), timed_future_actions=bool(predictions),
                  generated_samples_match_commands=bool(predictions),
                  force_and_rgb_replanned=len(chunks) > 1,
                  new_prediction_for_each_command=bool(predictions))
    lookup = {(r['policy_instance'], r['task_id'], r['inference_id']): r for r in chunks}
    for row in loaded:
        checks['flow_sampler_contract'] &= (row.get('execution_mode') == 'flow_receding_horizon'
            and row.get('ode_steps') == settings.get('ode_steps')
            and row.get('ode_solver') == settings.get('ode_solver')
            and row.get('noise_seed') == settings.get('noise_seed')
            and row.get('replan_every_steps') == settings.get('n_action_steps'))
    for row in predictions:
        try:
            index = row['chunk_index']
            chunk = lookup[(row['policy_instance'], row['task_id'], row['inference_id'])]
            expected = max(0, int(np.floor((row['simulation_time']-chunk['observation_stamp'])*20+1e-6)))
            checks['timed_future_actions'] &= bool(0 <= index < 20 and index == expected
                and np.isclose(row['chunk_observation_stamp'], chunk['observation_stamp'], rtol=0, atol=1e-9))
            sample = np.asarray(chunk['actions'], dtype=float)
            checks['generated_samples_match_commands'] &= bool(sample.shape == (20, 7)
                and np.isfinite(sample).all()
                and np.allclose(sample[index], row['raw_target'], rtol=1e-7, atol=1e-8))
            if settings.get('n_action_steps') == 1:
                checks['new_prediction_for_each_command'] &= bool(row['new_chunk'])
        except (KeyError, ValueError, IndexError, TypeError):
            checks['timed_future_actions'] = checks['generated_samples_match_commands'] = False
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',nargs='?',type=Path)
    parser.add_argument('--require-insertion',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=args.run or root/'results'/(root/'runtime/latest-run.txt').read_text().strip()
    manifest=json.loads((run/'run_manifest.json').read_text())
    evidence=json.loads((run/'evidence.json').read_text())
    lines=[json.loads(line) for line in (run/'inference.jsonl').read_text().splitlines() if line]
    predictions=[r for r in lines if r['type']=='prediction']
    loaded=[r for r in lines if r['type']=='model_loaded']
    contracts=[r for r in lines if r['type']=='observation_contract']
    endings=[r for r in lines if r['type']=='task_end']
    chunks=[r for r in lines if r['type']=='chunk_prediction']
    scoring=yaml.safe_load((run/'scoring/scoring.yaml').read_text()) if (run/'scoring/scoring.yaml').exists() else {}
    scores={name:value for name,value in scoring.items() if name!='total'}
    commands=[r for r in predictions if r['command_sent']]
    insertion={name:r['tier_3'].get('message')=='Cable insertion successful.' and r['tier_3']['score']>0 for name,r in scores.items()}
    multiview=manifest.get('method')=='MultiViewACT'
    expected_cameras=4 if multiview else 3
    checks={
        'evaluator_exit_zero':(run/'evaluator-exit-code.txt').exists() and (run/'evaluator-exit-code.txt').read_text().strip()=='0',
        'frozen_checkpoint_loaded':bool(loaded) and len({r['checkpoint_step'] for r in loaded})==1,
        'no_ground_truth':not manifest['ground_truth'] and all(not r['ground_truth_used'] for r in loaded),
        ('four_rgb_state_and_force_received' if multiview else 'three_rgb_state_and_force_received'):
            bool(contracts) and all(len(r['source_state'])==6 and len(r['wrench'])==6
                                   and len(r['camera_shapes'])==expected_cameras for r in contracts),
        'predictions_received':len(predictions)>0,
        'all_requested_trials_scored':set(scores)==set(manifest['trials']),
        'all_task_windows_finished':len(endings)==len(manifest['trials']) and all(r['policy_execution_completed'] for r in endings),
        'commands_finite_and_unit_quaternions':bool(predictions) and all(np.isfinite(r['command']).all() and abs(np.linalg.norm(r['command'][3:])-1)<1e-5 for r in predictions),
        'observations_not_stale':all(r['observation_age_seconds']<=manifest['settings']['max_observation_age_seconds'] for r in predictions),
    }
    if not manifest['dry_run']:
        checks.update(actual_pose_commands_sent=bool(commands) and evidence['pose_commands']>0,
            all_trials_model_valid=bool(scores) and all(r['tier_1']['score']==1 for r in scores.values()))
    if multiview:
        profile=manifest['settings']['collection_settings']
        camera_order=manifest['settings']['camera_order']
        checks['four_camera_order_matches']=bool(loaded) and all(r.get('camera_order')==camera_order for r in loaded+contracts)
        checks['four_camera_causal_sync']=bool(predictions)
        checks['joint_and_wrench_causal_sync']=bool(predictions)
        for row in predictions:
            try:
                stamp=row['observation_stamp']
                times=row['camera_stamps']
                ages=[stamp-times[name] for name in ('left','center','right','side')]
                checks['four_camera_causal_sync'] &= all(-1e-8<=age<=profile['camera_tolerance_s']+1e-8 for age in ages)
                checks['joint_and_wrench_causal_sync'] &= all(-1e-8<=stamp-row[key]<=profile['state_tolerance_s']+1e-8
                                                            for key in ('source_joint_stamp','wrench_stamp'))
            except (KeyError,TypeError,ValueError):
                checks['four_camera_causal_sync']=checks['joint_and_wrench_causal_sync']=False
    if any(r.get('checkpoint_format') in ('aic-sfp0-act-v2', 'aic-sfp0-act-pace-v1','aic-sfp0-act-multiview-v1') for r in loaded):
        checks.update(
            only_sfp0_trial=manifest['trials']==['trial_1'],
            timed_future_actions=bool(predictions) and all(0<=r['chunk_index']<20 and
                r['chunk_index']==max(0,int(np.floor((r['simulation_time']-r['chunk_observation_stamp'])*20+1e-6))) for r in predictions),
            force_and_rgb_replanned=bool(chunks) and len(chunks)>1)
        if any(r.get('execution_mode') == 'act_temporal_ensemble' for r in loaded):
            checks.update(temporal_ensemble_checks(predictions, manifest['settings']['temporal_ensemble_coeff'],
                                                 manifest['settings']['rate_hz']))
        elif any(r.get('execution_mode') == 'act_receding_horizon' for r in loaded):
            checks['new_prediction_for_each_command'] = all(r['new_chunk'] for r in predictions)
        else:
            groups={}
            for row in predictions:
                groups.setdefault((row['policy_instance'],row['task_id'],row['inference_id']),[]).append(row['chunk_index'])
            checks['actions_advance_within_chunks'] = any(len(indices)>1 and max(indices)>min(indices) for indices in groups.values()) \
                and all(all(b>=a for a,b in zip(indices,indices[1:])) for indices in groups.values())
    if any(r.get('checkpoint_format') == 'aic-sfp0-flow-matching-v1' for r in loaded):
        checks.update(flow_receding_horizon_checks(predictions, chunks, loaded, manifest))
    if args.require_insertion:
        checks['all_trials_inserted']=bool(insertion) and all(insertion.values())
    report={'passed':all(checks.values()),'checks':checks,'dry_run':manifest['dry_run'],
        'scope':'observations and learned inference' if manifest['dry_run'] else 'learned closed-loop observation/inference/command pipeline',
        'checkpoint_step':loaded[0]['checkpoint_step'] if loaded else None,
        'predictions':len(predictions),'commands':len(commands),'insertion_success_by_trial':insertion,
        'insertion_success_required':args.require_insertion,'task_endings':endings,
        'model_inferences':len(chunks) if chunks else len(predictions),
        'inference_seconds':{'median':float(np.median([r['inference_seconds'] for r in (chunks or predictions)])),
                             'max':max(r['inference_seconds'] for r in (chunks or predictions))} if predictions else None}
    if len(predictions)>1:
        span=predictions[-1]['simulation_time']-predictions[0]['simulation_time']
        report['observed_command_rate_hz']=(len(predictions)-1)/span if span>0 else None
    if predictions and 'ensemble_count' in predictions[0]:
        report['ensemble_count']={'mean':float(np.mean([r['ensemble_count'] for r in predictions])),
                                  'max':max(r['ensemble_count'] for r in predictions)}
    if multiview and predictions:
        xyz=np.asarray([row['actual_tcp_pose'][:3] for row in predictions])
        report['measured_tcp_excursion_m']=float(np.linalg.norm(xyz-xyz[0],axis=1).max())
        report['gazebo_gui_requested']=manifest.get('gui',False)
    (run/'ftact_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
