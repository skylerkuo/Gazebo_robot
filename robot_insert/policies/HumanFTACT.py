"""Four-view ACT with explicit operator takeover and synchronized action provenance."""
import base64
from contextlib import nullcontext
import time

import cv2
import numpy as np
import torch

from policies.FTACT import FTACT
from ftact_control import ACTTemporalPoseEnsembler,bounded_pose,pose_array,unit_pose,stamp_seconds
from human_control import OperatorLink,manual_pose_step
from multiview_control import multiview_arrays,ObservationNotReady
from sideview_data import camera_geometry
from sideview_recorder import Recorder


class HumanFTACT(FTACT):
    recording_controller='model_and_human'

    def create_operator(self):
        return OperatorLink(self.settings['human_operator'])

    def correction_source(self):
        return 'human'

    def correction_target(self,previous,control,ticks):
        return manual_pose_step(previous,control,self.minimum,self.maximum,ticks)

    def __init__(self,parent_node):
        super().__init__(parent_node)
        if not self.multiview or self.settings['dry_run']:raise ValueError('Human collection requires a real four-view ACT session')
        profile=dict(self.settings['collection_settings'],raw_dir=self.settings['raw_dir'],
                     trial_ids=self.settings.get('recording_task_ids',['task_1']),
                     camera_geometry=camera_geometry(self.settings['collection_settings']['camera']))
        if not hasattr(parent_node,'_human_ftact_recorder'):
            parent_node._human_ftact_recorder=Recorder(parent_node,profile)
        self.recorder=parent_node._human_ftact_recorder
        self.operator=self.create_operator()

    def insert_cable(self,task,get_observation,move_robot,send_feedback):
        if (task.target_module_name!=self.task_contract['target_module'] or
                task.port_name!=self.task_contract['port'] or task.plug_type!='sfp'):
            raise ValueError('Human collection remains scoped to SFP0')
        self.model.policy.reset();self.side_images.clear();self.source_joints.clear()
        if self.pace_context is not None:self.pace_context.reset()
        self.recorder.begin(task)
        ensemble=ACTTemporalPoseEnsembler(20,20,self.settings['temporal_ensemble_coeff'])
        start=self.time_now().nanoseconds*1e-9
        budget=min(float(task.time_limit),self.settings['max_task_seconds'])
        last_stamp=-float('inf');last_fresh=time.monotonic();next_tick=start
        count=inferences=stale=segment=0
        current=None;previous=None;manual_stamp=None;last_preview=-float('inf')
        status={'message':'模型已載入，等待同步觀測','remaining_seconds':budget,'frames':0}
        reason='canceled';completed=False;discarded=False
        self.write({'type':'task_start','task_id':task.id,'simulation_time':start,
                    'budget_seconds':budget,'human_assisted':self.correction_source()=='human',
                    'expert_assisted':self.correction_source()=='expert'})
        try:
            while self.active():
                now=self.time_now().nanoseconds*1e-9
                control=self.operator.poll(status)
                if control['finish']:
                    discarded=control['discard'];reason='operator_discarded' if discarded else 'operator_finished'
                    completed=True;break
                if now-start>=budget:reason='control_window_completed';completed=True;break
                if time.monotonic()-last_fresh>self.settings['max_wall_stall_seconds']:
                    raise TimeoutError('No usable synchronized observation')
                observation=get_observation()
                if observation is None or now<next_tick:time.sleep(.01);continue
                stamp=stamp_seconds(observation.center_image.header.stamp)
                if stamp<=last_stamp or stamp<start-.1 or now-stamp>self.settings['max_observation_age_seconds']:
                    time.sleep(.01);continue
                try:
                    arrays,joints,sync=multiview_arrays(observation,tuple(self.side_images),tuple(self.source_joints),
                        self._parent_node._ftact_side_info,self.settings['collection_settings'],self.settings['expected_camera_info'])
                    actual=unit_pose(pose_array(observation.controller_state.tcp_pose))
                except ObservationNotReady as error:
                    status.update(message=str(error));time.sleep(.01);continue
                last_stamp=stamp;last_fresh=time.monotonic()
                wrench=arrays['observation.environment_state'];force=float(np.linalg.norm(wrench[:3]))
                if self.pace_context is not None:arrays.update(self.pace_context.update(wrench,stamp))
                status={'message':'已收到插入事件，可按「完成本回合」送評分' if self.recorder.insertion_events else '模型或人工控制中',
                    'simulation_time':now,'elapsed_seconds':now-start,'insertion_event':bool(self.recorder.insertion_events),
                    'actual_tcp':actual.tolist(),'force_n':force,'frames':self.recorder.active.frames,
                    'remaining_seconds':max(0.,budget-(now-start)),
                    'request_hold':bool(force>self.settings['max_force_n'] or np.linalg.norm(wrench[3:])>self.settings['max_torque_nm'])}
                if self.correction_source()=='human' and time.monotonic()-last_preview>=.3:
                    last_preview=time.monotonic();status['preview_id']=count
                    status['previews']={}
                    for camera in self.camera_names:
                        image=arrays[f'observation.images.{camera}_camera']
                        ok,jpeg=cv2.imencode('.jpg',image[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,65])
                        if ok:status['previews'][camera]=base64.b64encode(jpeg).decode()
                control=self.operator.poll(status)
                if control['finish']:
                    discarded=control['discard'];reason='operator_discarded' if discarded else 'operator_finished'
                    completed=True;break
                key=(control['mode'],control['version'])
                if key!=current:
                    current=key;segment+=1;previous=actual.copy();manual_stamp=None
                    ensemble.reset();self.model.policy.reset()
                    self.write({'type':'control_switch','task_id':task.id,
                                'control_source':self.correction_source() if key[0]=='human' else key[0],
                                'correction_source':self.correction_source(),'takeover_reason':control.get('takeover_reason'),
                                'control_segment':segment,'simulation_time':now,'anchor_actual_tcp':actual.tolist()})
                mode=key[0];details={};elapsed=0.;chunk_index=None
                if count==0:
                    self.write({'type':'observation_contract','task_id':task.id,
                        'source_state':arrays['observation.state'].tolist(),'wrench':wrench.tolist(),
                        'camera_shapes':{k:list(v.shape) for k,v in arrays.items() if 'images' in k},**sync})
                if mode=='model':
                    batch={}
                    for name,array in arrays.items():
                        value=torch.from_numpy(array.copy())
                        if 'images' in name:value=value.permute(2,0,1).float()/255
                        batch[name]=value.unsqueeze(0).to(self.device)
                    precision=(torch.autocast('cuda',dtype=torch.bfloat16 if self.settings['amp']=='bf16' else torch.float16)
                               if self.settings['amp']!='none' else nullcontext())
                    began=time.monotonic()
                    with torch.inference_mode(),precision:chunk=self.model.predict_action_chunk(batch)[0].cpu().numpy()
                    elapsed=time.monotonic()-began;inferences+=1
                    self.write({'type':'chunk_prediction','task_id':task.id,'inference_id':inferences,
                        'observation_stamp':stamp,'inference_seconds':elapsed,'control_segment':segment})
                    latest=self.operator.poll(status)
                    if latest['finish']:
                        discarded=latest['discard'];reason='operator_discarded' if discarded else 'operator_finished'
                        completed=True;break
                    if (latest['mode'],latest['version'])!=current:
                        ensemble.reset();continue  # Never send an old model command after takeover.
                    now=self.time_now().nanoseconds*1e-9
                    if now-stamp>self.settings['max_observation_age_seconds']:
                        stale+=1;continue
                    ensemble.add(chunk,stamp,inferences);target,details=ensemble.target(now)
                    chunk_index=details['ensemble_source_indices'][-1]
                elif mode=='human':
                    ticks=1. if manual_stamp is None else max(0.,(stamp-manual_stamp)*20)
                    target=self.correction_target(previous,control,ticks)
                    details=dict(getattr(self,'correction_details',{}))
                    manual_stamp=stamp
                else:target=previous.copy()
                try:
                    command,limited=bounded_pose(target,previous,self.minimum,self.maximum,
                        self.settings['max_translation_step_m'],self.settings['max_rotation_step_rad'])
                except ValueError as error:
                    status.update(request_hold=True,message=f'輸出範圍限制：{error}；請人工接手')
                    self.operator.poll(status);ensemble.reset();continue
                motion=self.motion(command)
                if move_robot(motion_update=motion) is False:raise RuntimeError('Motion command was rejected')
                human_active=mode=='human' and bool(control.get('correction_active',np.any(control['axes'])))
                source=self.correction_source() if mode=='human' else mode
                self.recorder.command(motion,source=source,segment=segment,human_active=human_active)
                previous=command;count+=1
                now=self.time_now().nanoseconds*1e-9
                next_tick=details['ensemble_target_time'] if mode=='model' else max(stamp+.05,now+.001)
                self.write({'type':'prediction','task_id':task.id,'step':count,'simulation_time':now,
                    'observation_stamp':stamp,'observation_age_seconds':now-stamp,'actual_tcp_pose':actual.tolist(),
                    'raw_target':target.tolist(),'command':command.tolist(),'command_sent':True,
                    'control_source':source,'control_segment':segment,'human_active':human_active,
                    'inference_seconds':elapsed,'inference_id':inferences,'chunk_index':chunk_index,
                    'chunk_observation_stamp':stamp,'new_chunk':mode=='model','force_norm':force,**sync,**details,**limited})
                if count%20==0:send_feedback(f'{mode}: {count} commands, {self.recorder.active.frames} recorded frames')
        except Exception as error:
            reason=f'{type(error).__name__}: {error}';self.get_logger().error(reason)
        finally:
            if self.active():
                try:
                    actual=unit_pose(pose_array(get_observation().controller_state.tcp_pose))
                    hold=self.motion(actual)
                    if move_robot(motion_update=hold) is not False:self.recorder.command(hold,source='hold',segment=segment+1)
                except Exception:pass
            try:
                summary=self.recorder.finish(completed,controller=self.recording_controller)
                summary.update(operator_discarded=discarded,control_commands=count)
                if hasattr(self,'recording_control_version'):summary['controller_version']=self.recording_control_version
                if hasattr(self,'recording_control_parameters'):summary['control_parameters']=self.recording_control_parameters
                from sideview_data import json_write
                json_write(self.recorder.output/f'{task.id}.json',summary)
            finally:self.operator.close()
        self.write({'type':'task_end','task_id':task.id,'predictions':count,'model_inferences':inferences,
            'stale_predictions_dropped':stale,'reason':reason,'policy_execution_completed':completed and count>0,
            'human_assisted':self.correction_source()=='human','expert_assisted':self.correction_source()=='expert',
            'operator_discarded':discarded,'insertion_success':'determined_by_official_scoring'})
        return completed and count>0
