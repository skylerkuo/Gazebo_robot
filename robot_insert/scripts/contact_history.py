"""One causal 20 Hz history implementation for recorded data and deployment."""
from collections import deque
import threading
import numpy as np
import torch
from torch import nn

from contact_se3 import residual,rotate

TCP_HISTORY='contact.tcp_history'
FT_HISTORY='contact.ft_history'
VALID='contact.history_valid'
SOURCE_TIMES='contact.source_times'
GRID='contact.grid_times'
TRACKING='contact.tracking_error'
TRACKING_VALID='contact.tracking_valid'
COMMAND_TIMES='contact.command_times'
TRACKING_VERSION='tcp_ft_tracking_v1'
POSE_PAIR_VERSION='tcp_command_ft_pose_pair_v1'
ACTUAL_POSE_VERSION='tcp_actual_ft_v1'
COMMAND_HISTORY='contact.command_history'


def aligned_history(camera_times,tcp,wrench,wrench_times,anchor,horizon,*,tare=None,ema_alpha=1.,max_age=.1):
    if horizon not in (1,10,20):raise ValueError('H must be 1, 10 or 20')
    if not 0<ema_alpha<=1:raise ValueError('EMA must be causal with 0 < alpha <= 1')
    times=np.asarray(camera_times,dtype=np.float64);ftimes=np.asarray(wrench_times,dtype=np.float64)
    tcp=np.asarray(tcp,dtype=np.float32);wrench=np.asarray(wrench,dtype=np.float32)
    if (times.ndim!=1 or not len(times) or tcp.shape!=(len(times),7) or wrench.shape!=(len(times),6)
            or ftimes.shape!=times.shape or np.any(np.diff(times)<=0)):
        raise ValueError('Invalid or unordered history')
    stop=np.searchsorted(times,float(anchor)+1e-8,side='right')
    if stop<1:raise ValueError('No past observation available')
    # Never filter using rows after the observation anchor.
    times,tcp,ftimes=times[:stop],tcp[:stop],ftimes[:stop]
    values=wrench[:stop]-np.asarray(tare if tare is not None else [0.]*6,dtype=np.float32)
    # Values after anchor are never inspected, even for normalization/filtering.
    if (not all(np.isfinite(v).all() for v in (times,ftimes,tcp,values))
            or np.any(ftimes>times+1e-8) or np.any(times-ftimes>max_age+1e-8)):
        raise ValueError('Future/stale F/T timestamp or invalid history value')
    filtered=np.empty_like(values);filtered[0]=values[0]
    for i in range(1,len(values)):
        if ftimes[i]<ftimes[i-1]-1e-8:raise ValueError('F/T timestamps moved backwards')
        filtered[i]=(ema_alpha*values[i]+(1-ema_alpha)*filtered[i-1]
                     if ftimes[i]>ftimes[i-1]+1e-8 else filtered[i-1])
    grid=float(anchor)-np.arange(horizon-1,-1,-1)/20
    indices=np.searchsorted(times,grid+1e-8,side='right')-1
    valid=indices>=0;indices=np.maximum(indices,0)
    # Prefix padding repeats the earliest available past observation and has a
    # false validity bit. It never uses a frame later than the current anchor.
    if np.any(grid[valid]-times[indices[valid]]>max_age+1e-8):raise ValueError('History has a camera gap')
    return {TCP_HISTORY:tcp[indices].copy(),FT_HISTORY:filtered[indices].copy(),VALID:valid,
            SOURCE_TIMES:np.stack((times[indices],ftimes[indices]),-1),GRID:grid}


def tracking_errors(tcp, commands):
    """External published target minus actual pose, expressed in base_link (m/rad)."""
    from contact_se3 import unit, multiply, so3_log
    actual=torch.as_tensor(np.asarray(tcp),dtype=torch.float32)
    target=torch.as_tensor(np.asarray(commands),dtype=torch.float32)
    q=unit(actual[...,3:]);inverse=torch.cat((-q[...,:3],q[...,3:]),-1)
    return torch.cat((target[...,:3]-actual[...,:3],
                      so3_log(multiply(unit(target[...,3:]),inverse))),-1).numpy()


def add_tracking_history(history, command_times, commands, *, raw_commands=False, preserve_history_valid=False):
    """Use the last published command at each selected TCP source timestamp."""
    times=np.asarray(command_times,dtype=np.float64).reshape(-1)
    poses=np.asarray(commands,dtype=np.float32).reshape(-1,7)
    if len(times)!=len(poses) or not np.isfinite(times).all() or np.any(np.diff(times)<0):
        raise ValueError('Invalid or unordered published command timestamps')
    source=history[SOURCE_TIMES][:,0]
    # A tiny tolerance covers roundoff between recorded simulation clocks only.
    ids=np.searchsorted(times,source+1e-8,side='right')-1
    valid=history[VALID] & (ids>=0)
    errors=np.zeros((len(source),6),dtype=np.float32)
    targets=np.zeros((len(source),7),dtype=np.float32);targets[:,6]=1.
    stamps=np.zeros(len(source),dtype=np.float64)
    if valid.any():
        selected=poses[ids[valid]]
        if not np.isfinite(selected).all():raise ValueError('Nonfinite published command')
        if raw_commands:targets[valid]=selected
        else:errors[valid]=tracking_errors(history[TCP_HISTORY][valid],selected)
        stamps[valid]=times[ids[valid]]
    feature={COMMAND_HISTORY:targets} if raw_commands else {TRACKING:errors}
    if preserve_history_valid:
        # A missing command does not invalidate the observed TCP and F/T.
        return {**history,**feature,COMMAND_TIMES:stamps,TRACKING_VALID:valid}
    return {**history,**feature,COMMAND_TIMES:stamps,VALID:valid}


class LiveHistory:
    def __init__(self,config):
        self.config=config;self.lock=threading.RLock();self.reset()

    def reset(self,start=None):
        with self.lock:
            self.commands=deque(maxlen=4096);self.rows=deque(maxlen=4096);self.pending=deque(maxlen=400);self.last_error=None;self.min_stamp=start

    def append(self,stamp,tcp,wrench,wrench_stamp):
        with self.lock:
            if self.min_stamp is not None and stamp<self.min_stamp-1e-8:return
            if self.rows and stamp<=self.rows[-1][0]:return
            self.rows.append((float(stamp),np.asarray(tcp,dtype=np.float32),np.asarray(wrench,dtype=np.float32),float(wrench_stamp)))

    def append_command(self,stamp,pose):
        # Called only after a non-dry-run command was accepted by move_robot.
        if self.config.get('history_features') not in (TRACKING_VERSION,POSE_PAIR_VERSION,ACTUAL_POSE_VERSION):return
        pose=np.asarray(pose,dtype=np.float32).copy()
        if not np.isfinite(stamp) or pose.shape!=(7,) or not np.isfinite(pose).all():
            raise ValueError('Invalid published TCP command')
        with self.lock:
            if self.min_stamp is not None and stamp<self.min_stamp-1e-8:return
            if self.commands and stamp<self.commands[-1][0]:raise ValueError('Command clock moved backwards')
            self.commands.append((float(stamp),pose))

    def sample(self,stamp):
        with self.lock:
            rows=[r for r in self.rows if r[0]<=stamp+1e-8]
            commands=list(self.commands)
        if not rows:raise ValueError('Waiting for causal TCP/FT history')
        if abs(rows[-1][0]-stamp)>1e-5:raise ValueError('Waiting for TCP at the camera timestamp')
        history=aligned_history([r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows],
            [r[3] for r in rows],stamp,self.config['history_length'],tare=self.config['tare'],
            ema_alpha=self.config['ema_alpha'],max_age=self.config['max_sensor_age_s'])
        if self.config.get('history_features') in (TRACKING_VERSION,POSE_PAIR_VERSION,ACTUAL_POSE_VERSION):
            history=add_tracking_history(history,[r[0] for r in commands],[r[1] for r in commands],
                raw_commands=self.config.get('history_features') in (POSE_PAIR_VERSION,ACTUAL_POSE_VERSION),
                preserve_history_valid=self.config.get('tracking_preserve_history_valid',False))
        return history


class HistoryEncoder(nn.Module):
    def __init__(self,stats,width=64,output_dim=128):
        super().__init__()
        for name in ('ft_mean','ft_std','tcp_mean','tcp_std'):
            self.register_buffer(name,torch.tensor(stats[name],dtype=torch.float32))
        self.conv1=nn.Conv1d(13,width,3);self.conv2=nn.Conv1d(width,width,3)
        self.out=nn.Sequential(nn.Linear(width+9,output_dim),nn.SiLU())

    def forward(self,batch):
        tcp=batch[TCP_HISTORY].float();current=tcp[:,-1]
        anchor=current[:,None].expand_as(tcp)
        delta=residual(anchor,tcp)/tcp.new_tensor([.01,.01,.01,.1,.1,.1])
        ft=(batch[FT_HISTORY].float()-self.ft_mean)/self.ft_std
        x=torch.cat((delta,ft,batch[VALID].float()[...,None]),-1).transpose(1,2)
        # Left padding only: there is no convolutional access to a future sample.
        x=torch.nn.functional.silu(self.conv1(torch.nn.functional.pad(x,(2,0))))
        x=torch.nn.functional.silu(self.conv2(torch.nn.functional.pad(x,(2,0))))
        # Pool all H causal features, rather than a 5-frame receptive field only.
        temporal=x.mean(-1)
        axes=current.new_tensor([[1.,0,0],[0,1.,0]])[None].expand(len(current),-1,-1)
        rotation=rotate(current[:,None,3:].expand(-1,2,-1),axes).flatten(1)
        pose=torch.cat(((current[:,:3]-self.tcp_mean)/self.tcp_std,rotation),-1)
        return self.out(torch.cat((temporal,pose),-1))


def ros_enqueue(store,observation):
    """Retain only timestamped small sensor values, never buffered RGB messages."""
    from ftact_control import stamp_seconds
    wrench=observation.wrist_wrench;w=wrench.wrench
    if wrench.header.frame_id!=store.config['wrench_frame']:
        store.last_error='F/T sensor frame changed';return
    row=(stamp_seconds(observation.center_image.header.stamp),
         [w.force.x,w.force.y,w.force.z,w.torque.x,w.torque.y,w.torque.z],stamp_seconds(wrench.header.stamp))
    with store.lock:
        if store.min_stamp is not None and row[0]<store.min_stamp-1e-8:return
        if not store.pending or row[0]>store.pending[-1][0]:store.pending.append(row)


def ros_flush(store,node,anchor):
    from rclpy.time import Time
    with store.lock:
        while store.pending and store.pending[0][0]<=anchor+1e-8:
            stamp,wrench,wrench_stamp=store.pending[0]
            try:
                tf=node._tf_buffer.lookup_transform('base_link','gripper/tcp',Time(seconds=stamp)).transform
            except Exception as error:
                store.last_error=str(error)
                if anchor-stamp<=store.config['max_sensor_age_s']:break
                store.pending.popleft();continue
            p,q=tf.translation,tf.rotation
            store.append(stamp,[p.x,p.y,p.z,q.x,q.y,q.z,q.w],wrench,wrench_stamp)
            store.pending.popleft()
