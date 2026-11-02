#!/usr/bin/env python3
"""Run the current four-camera ACT checkpoint with the Gazebo UI enabled."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import secrets
import time
import shutil
import errno

import torch
import yaml

from act_multiview import FORMAT,FOUR_CAMERAS,index_digest
from automatic_expert_control import EXPERT_CONTROL_VERSION,EXPERT_MAX_SECONDS
from sideview_data import (TASK,json_write,validate_settings,make_engine,make_world,
                           make_robot_description,digest_file)

ROOT=Path(__file__).resolve().parents[1]
SHARE=ROOT/'runtime/rootfs/ws_aic/install/share'


def collection_contract(payload,portable=None):
    if (payload.get('format') not in (FORMAT,'aic-sfp0-act-pace-multiview-v1','aic-sfp0-act-pace-pred-multiview-v1','aic-sfp0-flow-matching-multiview-v1','aic-sfp0-supervised-residual-flow-multiview-v1','aic-sfp0-stacked-residual-flow-multiview-v1','aic-sfp0-alternating-act-residual-flow-multiview-v1') or payload.get('camera_order')!=list(FOUR_CAMERAS)
            or payload.get('action_offset')!=1 or payload['config']['chunk_size']!=20
            or payload.get('task')!=TASK):
        raise ValueError('Select a four-view SFP0 ACT checkpoint with future-20 absolute TCP actions')
    if portable is not None:
        value=json.loads(Path(portable).read_text())
        checksum=value.pop('contract_sha256',None)
        if (value.get('format')!='aic-sfp0-portable-collection-contract-v1' or
                checksum!=index_digest(value) or value['data_index_sha256']!=payload['data_index_sha256']):
            raise ValueError('Portable collection contract checksum or checkpoint binding changed')
        validate_settings(value['profile'])
        return value['profile'],value['camera_info'],dict(value['source_contract'],
            portable_contract=str(Path(portable).resolve()),portable_contract_sha256=checksum)
    root=Path(payload['config']['dataset_root']).resolve()
    index=json.loads(Path(payload['config']['index_path']).read_text())
    if index.get('index_content_sha256')!=payload['data_index_sha256'] or index_digest(index)!=payload['data_index_sha256']:
        raise ValueError('Checkpoint training-index checksum differs from its source index')
    manifest=json.loads((root/'_collection/manifest.json').read_text())
    profile=manifest['settings']
    validate_settings(profile)
    if manifest['task']!=TASK: raise ValueError('Collection is for a different task')
    entries={e['episode_index']:e for e in manifest['attempts'] if e.get('accepted')}
    for ep in index['episodes']:
        entry=entries[ep['episode_index']]
        if entry['attempt']!=ep['source_attempt'] or entry['validation']['parquet_sha256']!=ep['source_sha256']:
            raise ValueError('Source collection membership/checksum changed since training')
    first=entries[index['episodes'][0]['episode_index']]
    calibration_name=(f"{first.get('recording_task_id','task_1')}_calibration.json" if manifest.get('format') in
        ('aic-sfp0-official-expert-interventions-v1','aic-sfp0-human-interventions-v1') else
        f"episode_{first['attempt']:06d}_calibration.json")
    calibration=root/first['run']/'raw'/calibration_name
    recorded=json.loads(calibration.read_text())
    if (recorded['side_camera']!=profile['camera'] or
            (recorded['record_width'],recorded['record_height'])!=(profile['record_width'],profile['record_height'])):
        raise ValueError('Collection camera geometry/resolution differs from recorded calibration')
    info=recorded['source_camera_info']
    return profile,info,{'collection_root':str(root),'collection_settings_sha256':index_digest(profile),
        'calibration_source':str(calibration),'calibration_sha256':digest_file(calibration),
        'data_index_sha256':payload['data_index_sha256']}


def snapshot_checkpoint(source,destination,*,shared_cache=None):
    if shared_cache is not None:
        cache=Path(shared_cache);cache.mkdir(parents=True,exist_ok=True)
        with Path(source).open('rb') as reader:
            digest=hashlib.file_digest(reader,'sha256').hexdigest()
            frozen=cache/f'{digest}.pt'
            if not frozen.exists():
                temporary=cache/f'{digest}.{os.getpid()}.partial'
                reader.seek(0)
                with temporary.open('wb') as writer:shutil.copyfileobj(reader,writer,8*1024*1024)
                if digest_file(temporary)!=digest:raise ValueError('Shared checkpoint snapshot checksum mismatch')
                temporary.replace(frozen)
            elif digest_file(frozen)!=digest:raise ValueError('Shared checkpoint snapshot changed')
            try:os.link(frozen,destination)
            except OSError as error:
                if error.errno not in (errno.EXDEV,errno.EPERM,errno.EACCES,errno.EOPNOTSUPP):raise
                # Hard links may be unavailable on another filesystem. Preserve
                # the same immutable bytes and keep single-run behavior usable.
                shutil.copyfile(frozen,destination)
        return digest
    digest=hashlib.sha256()
    # Open once: best.pt is atomically replaced while training continues.
    with Path(source).open('rb') as reader,Path(destination).open('wb') as writer:
        for block in iter(lambda:reader.read(8*1024*1024),b''):
            writer.write(block);digest.update(block)
    return digest.hexdigest()


def inference_scene(source,profile,max_task_seconds,scene_attempt=None):
    randomized=scene_attempt is None
    if randomized: scene_attempt=secrets.randbelow(2**32-1)+1
    if scene_attempt<0: raise ValueError('scene-attempt must be nonnegative')
    engine,_=make_engine(source,[scene_attempt],profile)
    trial=next(iter(engine['trials'].values()));task=next(iter(trial['tasks'].values()))
    task['time_limit']=max(10,math.ceil(max_task_seconds+5))
    trial['tasks']={'task_1':task};engine['trials']={'trial_1':trial}
    pose=trial['scene']['task_board']['pose']
    reference=source['trials']['trial_1']['scene']['task_board']['pose']
    details={'mode':'random' if randomized else 'fixed','scene_attempt':scene_attempt,
        'seed':profile['seed'],'task_board_pose':dict(pose),
        'translation_offset_m':{axis:pose[axis]-reference[axis] for axis in ('x','y','z')},
        'yaw_offset_rad':pose['yaw']-reference['yaw'],'limits':dict(profile['variation'])}
    return engine,details


def main(default_checkpoint=None,default_config=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=default_checkpoint or Path('/mnt/d/robot_models/aic_sfp0_act_multiview_v1/checkpoints/best.pt'))
    parser.add_argument('--config',type=Path,default=default_config or ROOT/'configs/sfp0_multiview_gazebo.json')
    parser.add_argument('--collection-contract',type=Path,help='Portable camera/source metadata when old raw data was deleted')
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--human-control',action='store_true',help='Keyboard takeover window and synchronized correction recording')
    parser.add_argument('--auto-expert',action='store_true',help='ACT followed by official ground-truth expert; record corrective supervision')
    parser.add_argument('--batch-size',type=int,default=1,help='Automatic expert trials sharing one Gazebo and model process')
    parser.add_argument('--keep-scoring-bags',action='store_true',help='Keep large ROS scoring bags after expert results are committed')
    parser.add_argument('--expert-after-seconds',type=float,default=20.,help='Simulation seconds allowed to ACT before official expert takeover')
    parser.add_argument('--expert-max-seconds',type=float,default=EXPERT_MAX_SECONDS,help='Simulation seconds available for expert alignment/insertion')
    parser.add_argument('--expert-data-root',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_expert_v1'))
    parser.add_argument('--human-data-root',type=Path,default=Path('/mnt/d/robot_datasets/aic_sfp0_human_v1'))
    parser.add_argument('--gui',choices=('true','false'),default='true')
    parser.add_argument('--renderer',choices=('hardware','software'),default='hardware')
    parser.add_argument('--device',choices=('cuda','cpu'))
    parser.add_argument('--amp',choices=('none','bf16','fp16'))
    parser.add_argument('--ode-steps',type=int)
    parser.add_argument('--ode-solver',choices=('euler','heun'))
    parser.add_argument('--noise-seed',type=int)
    parser.add_argument('--real-time-factor',type=float,default=.5)
    parser.add_argument('--scene-attempt',type=int,
        help='Omit for a random small scene variation; 0: official sample; >0: fixed reproducible geometry')
    parser.add_argument('--max-task-seconds',type=float)
    parser.add_argument('--max-wall-seconds',type=float,default=1200.)
    parser.add_argument('--dry-run',action='store_true',help='Observe and predict without sending motion commands')
    parser.add_argument('--prepare-only',action='store_true',help='Freeze checkpoint and generate the exact scene/settings; do not launch')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    if args.batch_size>1 and not args.auto_expert:parser.error('Batch sessions currently require --auto-expert')
    if args.auto_expert and args.human_control:parser.error('Select keyboard assistance or automatic expert, not both')
    if args.auto_expert and args.dry_run:parser.error('Automatic expert collection sends simulated commands; omit --dry-run')
    if args.auto_expert and (not math.isfinite(args.expert_after_seconds) or args.expert_after_seconds<=0):
        parser.error('expert-after-seconds must be finite and positive')
    if args.auto_expert and (not math.isfinite(args.expert_max_seconds) or args.expert_max_seconds<=0):
        parser.error('expert-max-seconds must be finite and positive')
    if args.auto_expert and not args.expert_data_root.resolve().is_relative_to(Path('/mnt/d')):
        parser.error('Expert recordings must be stored on D:')
    if args.human_control and args.dry_run:parser.error('Human collection sends real simulated commands; omit --dry-run')
    if args.human_control and not args.human_data_root.resolve().is_relative_to(Path('/mnt/d')):
        parser.error('Human recordings must be stored on D:')
    operator_test=args.human_control and os.environ.get('ROBOT_INSERT_HUMAN_TEST')=='1'
    if not args.checkpoint.is_file(): parser.error('Checkpoint is missing; select --checkpoint or wait for best.pt')
    if not (ROOT/'runtime/rootfs.ready').exists(): parser.error('Prepare the official AIC image first')
    if (not 0<args.real_time_factor<=1 or (args.scene_attempt is not None and args.scene_attempt<0)
            or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=10):
        parser.error('Use 0 < real-time-factor <= 1, scene-attempt >= 0 and max-wall-seconds > 10')
    settings=json.loads(args.config.read_text())
    if (args.human_control or args.auto_expert) and args.max_task_seconds is None:settings['max_task_seconds']=180
    for key in ('device','amp','max_task_seconds','ode_steps','ode_solver','noise_seed'):
        if getattr(args,key) is not None: settings[key]=getattr(args,key)
    if args.auto_expert and args.expert_after_seconds+args.expert_max_seconds+5>settings['max_task_seconds']:
        parser.error('Allow expert-after-seconds + expert-max-seconds + 5 seconds for expert recovery')
    if settings['device']=='cpu' and args.amp is None: settings['amp']='none'
    flow_mode=settings['execution_mode']=='sfp0_flow_receding_horizon'
    if (settings['execution_mode'] not in ('sfp0_act_temporal_ensemble','sfp0_flow_receding_horizon') or settings['n_action_steps']!=1
            or settings['rate_hz']!=20 or settings['action_frame']!='base_link'):
        parser.error('Four-view policy requires matching execution mode, per-step replanning, 20 Hz and base_link')
    if flow_mode:
        from flow_matching import validate_solver
        try:validate_solver(settings['ode_steps'],settings['ode_solver'])
        except (KeyError,ValueError) as error:parser.error(str(error))
        if args.auto_expert or args.human_control:parser.error('Four-view Flow launcher supports autonomous inference only')
    elif any(getattr(args,key) is not None for key in ('ode_steps','ode_solver','noise_seed')):
        parser.error('ODE/noise overrides require the Flow Matching inference config')
    for key in ('max_task_seconds','max_observation_age_seconds','max_wall_stall_seconds',
                'max_translation_step_m','max_rotation_step_rad','max_force_n','max_torque_nm'):
        if not math.isfinite(settings[key]) or settings[key]<=0: parser.error(f'{key} must be finite and positive')
    if not math.isfinite(settings['temporal_ensemble_coeff']) or settings['temporal_ensemble_coeff']<0:
        parser.error('temporal-ensemble coefficient must be finite and nonnegative')
    if settings['device']=='cpu' and settings['amp']!='none': parser.error('CPU inference requires amp=none')
    if settings['device']=='cuda' and not args.prepare_only:
        if not torch.cuda.is_available(): parser.error('CUDA is unavailable; use --device cpu')
        # Gazebo rendering and ACT share this laptop GPU with training.
        free,_=torch.cuda.mem_get_info()
        required=(3.5 if args.renderer=='hardware' else 1.5)*1024**3
        if free<required:
            parser.error(f'Only {free/1024**3:.2f} GiB GPU memory free; this mode needs about {required/1024**3:g} GiB. '
                         'Stop training with Ctrl+C before GPU/Gazebo inference, or select --device cpu --renderer software.')
    default_results=Path('/mnt/d/robot_models/aic_sfp0_flow_expert_v1/gazebo_results' if flow_mode else
                         '/mnt/d/robot_models/aic_sfp0_act_multiview_v1/gazebo_results')
    if default_checkpoint is not None:default_results=default_checkpoint.parent.parent/'gazebo_results'
    results=(args.results_dir or (args.expert_data_root/'runs' if args.auto_expert else args.human_data_root/'runs' if args.human_control else
             default_results)).expanduser().resolve()
    if not results.is_relative_to(Path('/mnt/d')): parser.error('Inference results/checkpoint snapshots must be on D:')
    lock=(ROOT/'runtime/session.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: parser.error('Another robot_insert Gazebo session is running')
    began=time.perf_counter()
    method='AutoExpertACT' if args.auto_expert else 'HumanAssistedACT' if args.human_control else 'MultiViewFlow' if flow_mode else 'MultiViewACT'
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{method}_{os.getpid()}'
    output=results/name;output.mkdir(parents=True)
    checkpoint=output/'checkpoint.pt'
    digest=snapshot_checkpoint(args.checkpoint,checkpoint,
        shared_cache=Path('/mnt/d/robot_models/_shared_checkpoints') if args.auto_expert else None)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (payload.get('format') in ('aic-sfp0-flow-matching-multiview-v1','aic-sfp0-supervised-residual-flow-multiview-v1','aic-sfp0-stacked-residual-flow-multiview-v1','aic-sfp0-alternating-act-residual-flow-multiview-v1'))!=flow_mode:
        parser.error('Checkpoint and inference config disagree about Flow Matching execution')
    portable=args.collection_contract or Path('/mnt/d/robot_models/_collection_contracts')/f"{payload['data_index_sha256']}.json"
    try: profile,camera_info,contract=collection_contract(payload,portable if portable.is_file() else None)
    except (ValueError,KeyError,FileNotFoundError) as error: parser.error(str(error))
    profile=dict(profile,real_time_factor=args.real_time_factor)
    source=SHARE/'aic_engine/config/sample_config.yaml'
    from expert_batch import make_expert_trials
    engine,episodes=make_expert_trials(yaml.safe_load(source.read_text()),profile,settings['max_task_seconds'],args.batch_size,args.scene_attempt)
    scene=episodes[0]
    (output/'engine_config.yaml').write_text(yaml.safe_dump(engine,sort_keys=False))
    make_world(SHARE/'aic_description/world/aic.sdf',output/'sideview_world.sdf',profile)
    make_robot_description(SHARE/'aic_description/urdf/ur_gz.urdf.xacro',
        SHARE/'aic_assets/models/Basler Camera/basler_camera_macro.xacro',
        output/'recording_robot.xacro',output/'recording_camera.xacro',profile,f'/workspace/results/{name}')
    bridge=yaml.safe_load((SHARE/'aic_bringup/config/ros_gz_bridge_config.yaml').read_text())
    for topic,ros_type,gz_type in [('image','sensor_msgs/msg/Image','gz.msgs.Image'),
                                  ('camera_info','sensor_msgs/msg/CameraInfo','gz.msgs.CameraInfo')]:
        bridge.append({'ros_topic_name':f'/side_camera/{topic}','gz_topic_name':f'/side_camera/{topic}',
                       'ros_type_name':ros_type,'gz_type_name':gz_type,'direction':'GZ_TO_ROS','lazy':True})
    (output/'bridge.yaml').write_text(yaml.safe_dump(bridge,sort_keys=False))
    settings.update(checkpoint=f'/workspace/results/{name}/checkpoint.pt',output_dir=f'/workspace/results/{name}',
        dry_run=args.dry_run,camera_order=list(FOUR_CAMERAS),collection_settings=profile,
        expected_camera_info=camera_info,trial_ids=[e['trial_id'] for e in episodes],
        recording_task_ids=[e['task_id'] for e in episodes],recording_episodes=episodes)
    operator_server=operator_window=None
    if args.auto_expert:
        settings.update(expert_assisted=True,expert_after_seconds=args.expert_after_seconds,
            expert_max_seconds=args.expert_max_seconds,expert_controller_version=EXPERT_CONTROL_VERSION,
            expert_z_stiffness=600.,expert_z_damping=70.,
            raw_dir=f'/workspace/results/{name}/raw')
    if args.human_control:
        settings.update(human_assisted=True,raw_dir=f'/workspace/results/{name}/raw')
        if not args.prepare_only:
            from human_control import OperatorServer
            from human_operator_ui import OperatorWindow
            operator_server=OperatorServer()
            try:operator_window=OperatorWindow(operator_server.state)
            except Exception:
                operator_server.close();raise
            settings['human_operator']=operator_server.endpoint()
            if operator_test:operator_window.root.title('TEST — AIC human controls (excluded from training)')
    json_write(output/'settings.json',settings)
    manifest={'method':method,'source_checkpoint':str(args.checkpoint.resolve()),
        'checkpoint_sha256':digest,'checkpoint_step':payload['step'],'checkpoint_format':payload['format'],
        'ground_truth':args.auto_expert,'model_ground_truth_inputs':False,
        'ground_truth_supervision':args.auto_expert,'gui':args.gui=='true','renderer':args.renderer,'scene_attempt':scene['scene_attempt'],
        'scene_variation':scene,
        'trials':settings['trial_ids'],'recording_episodes':episodes,'batch_size':args.batch_size,
        'dry_run':args.dry_run,'settings':settings,'source_contract':contract,
        'world_sha256':digest_file(output/'sideview_world.sdf'),
        'wall_timeout_seconds':args.max_wall_seconds,'prepare_only':args.prepare_only,
        'human_data_root':str(args.human_data_root.resolve()) if args.human_control else None,
        'operator_test':operator_test}
    if args.auto_expert:
        manifest.update(expert_data_root=str(args.expert_data_root.resolve()),expert_controller_version=EXPERT_CONTROL_VERSION)
    json_write(output/'run_manifest.json',manifest)
    local=ROOT/'results';local.mkdir(exist_ok=True)
    (local/name).symlink_to(output,target_is_directory=True)
    if not args.prepare_only: (ROOT/'runtime/latest-run.txt').write_text(name+'\n')
    policy_name=('Alternating-trained ACT + Residual Flow' if payload.get('format')=='aic-sfp0-alternating-act-residual-flow-multiview-v1'
                 else 'ACT + first Flow + second Residual Flow' if payload.get('format')=='aic-sfp0-stacked-residual-flow-multiview-v1'
                 else 'ACT + supervised Residual Flow' if payload.get('format')=='aic-sfp0-supervised-residual-flow-multiview-v1'
                 else 'Flow Matching' if flow_mode else 'ACT/PACE')
    print(f'Four-view {policy_name} checkpoint step {payload["step"]}; Gazebo GUI={args.gui}',flush=True)
    if payload.get('format')=='aic-sfp0-alternating-act-residual-flow-multiview-v1':
        print(f'ACT initialization step {payload["config"]["base_act_step"]}; '
              f'ACT updates {payload["act_updates"]}, Flow updates {payload["flow_updates"]}; '
              f'ODE {settings["ode_solver"]}/{settings["ode_steps"]}; expert assistance disabled.',flush=True)
    if payload.get('format')=='aic-sfp0-stacked-residual-flow-multiview-v1':
        base=payload['config']['base_policy_config']
        print(f'Frozen ACT step {base["base_act_step"]} + first Flow step {payload["config"]["base_policy_step"]}; '
              f'first Flow ODE {base["ode_solver"]}/{base["ode_steps"]}; '
              f'second Flow ODE {settings["ode_solver"]}/{settings["ode_steps"]}; expert assistance disabled.',flush=True)
    if payload.get('format')=='aic-sfp0-supervised-residual-flow-multiview-v1':
        print(f'Frozen ACT base step {payload["config"]["base_act_step"]}; '
              f'ACT + residual scale {payload["config"]["residual_scale"]:g}; '
              f'ODE {settings["ode_solver"]}/{settings["ode_steps"]}; expert assistance disabled.',flush=True)
    offsets=scene['translation_offset_m']
    print(f'Scene {scene["mode"]}; --scene-attempt {scene["scene_attempt"]}; board offset '
          f'XYZ=({offsets["x"]*1000:.2f}, {offsets["y"]*1000:.2f}, {offsets["z"]*1000:.2f}) mm, '
          f'yaw={math.degrees(scene["yaw_offset_rad"]):.3f} deg',flush=True)
    print(f'Results: {output}',flush=True)
    if args.auto_expert:
        print(f'ACT first {args.expert_after_seconds:g}s, then official ground-truth expert '
              f'({EXPERT_CONTROL_VERSION}, budget {args.expert_max_seconds:g}s); data provenance=expert.',flush=True)
        print(f'Shared simulator/model session: {args.batch_size} independent trial(s).',flush=True)
    del payload
    if args.prepare_only: return 0
    env=dict(os.environ,INSERT_WITH_ML='true',INSERT_RESULTS_DIR=str(results))
    if args.renderer=='software': env['GALLIUM_DRIVER']='llvmpipe'
    command=[str(ROOT/'scripts/enter.sh'),'/bin/bash','/workspace/scripts/session_multiview.sh',
             name,args.gui,str(args.max_wall_seconds-10),'policies.AutoExpertFTACT' if args.auto_expert else 'policies.HumanFTACT' if args.human_control else 'policies.FTACT']
    try:process=subprocess.Popen(command,env=env,start_new_session=True)
    except Exception:
        if operator_window is not None:operator_window.root.destroy()
        if operator_server is not None:operator_server.close()
        raise
    interrupted=False
    try:
        if operator_window is not None:
            interrupted=operator_window.run(process,args.max_wall_seconds)
            status=process.wait(timeout=15)
        else:status=process.wait(timeout=args.max_wall_seconds)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        interrupted=True
        if process.poll() is None: os.killpg(process.pid,signal.SIGINT)
        try: status=process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);status=process.wait()
    manifest.update(launcher_exit_code=status,interrupted_or_timed_out=interrupted,
        session_wall_seconds=time.perf_counter()-began)
    if operator_server is not None:
        manifest['operator_discarded']=operator_server.state.discard
        operator_server.close()
    json_write(output/'run_manifest.json',manifest)
    print(f'Session exited {status}; inspect {output/"model.log"}',flush=True)
    if args.human_control:
        from human_dataset import import_human_run
        entry=import_human_run(args.human_data_root,output,profile,contract['collection_root'],status=status,
                               operator_discarded=manifest.get('operator_discarded',False),test_run=operator_test)
        print(json.dumps({'type':'human_episode_import','accepted':entry['accepted'],
            'expert_anchors':entry.get('validation',{}).get('training_anchor_count',entry.get('validation',{}).get('expert_anchor_count',0)),
            'reason':entry.get('rejection_reason'),'dataset':str(args.human_data_root.resolve())}),flush=True)
    if args.auto_expert:
        from expert_batch import import_expert_session,prune_scoring_bags
        importing=time.perf_counter()
        entries=import_expert_session(args.expert_data_root,output,status=status)
        if status==0 and not args.keep_scoring_bags:
            manifest['scoring_bag_cleanup']=prune_scoring_bags(output)
        manifest.update(import_wall_seconds=time.perf_counter()-importing,total_wall_seconds=time.perf_counter()-began)
        json_write(output/'run_manifest.json',manifest)
        print(json.dumps({'type':'expert_session_complete','trials':len(entries),
            'accepted':sum(bool(e['accepted']) for e in entries),
            'session_wall_seconds':manifest['session_wall_seconds'],
            'import_wall_seconds':manifest['import_wall_seconds'],'total_wall_seconds':manifest['total_wall_seconds']}),flush=True)
    return status if status>=0 else 128-status


if __name__=='__main__': raise SystemExit(main())
