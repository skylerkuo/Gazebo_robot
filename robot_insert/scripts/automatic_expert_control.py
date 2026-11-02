"""Deterministic, simulation-time handoff. No plug/port ground truth enters ACT."""
import math
import numpy as np

from ftact_control import unit_pose

EXPERT_CONTROL_VERSION='in_place_progress_v2'
EXPERT_MAX_SECONDS=60.


class InPlaceExpertAlignment:
    """Align at the takeover TCP height; descend only while the tip is aligned."""
    def __init__(self,actual,now):
        actual=unit_pose(actual)
        self.height=float(actual[2]);self.floor=self.height-.08
        self.last_time=float(now);self.aligned_since=None;self.phase='align'
        self.geometry_initialized=False;self.force_baseline=None;self.force_paused=False
        self.progress_z=float(actual[2]);self.progress_time=float(now)

    def target(self,official,actual,plug,port,now,*,force_n=None):
        target,actual,plug,port=map(unit_pose,(official,actual,plug,port))
        dt=max(0.,min(.1,float(now)-self.last_time));self.last_time=float(now)
        if not self.geometry_initialized:
            # Stop the TCP reference when the tip is 10 mm below the port plane,
            # with an absolute 80 mm descent bound from takeover. Never retract.
            self.floor=max(self.floor,min(self.height,float(actual[2]-(plug[2]-port[2])-.01)))
            self.geometry_initialized=True
        if force_n is not None:
            if not math.isfinite(force_n) or force_n<0:raise ValueError('Invalid expert force norm')
            if self.force_baseline is None:self.force_baseline=float(force_n)
            if force_n>self.force_baseline+15:self.force_paused=True
            elif force_n<self.force_baseline+8:self.force_paused=False
        xy=float(np.linalg.norm(plug[:2]-port[:2]))
        angle=float(np.rad2deg(2*np.arccos(np.clip(abs(np.dot(plug[3:],port[3:])),0,1))))
        aligned=xy<=.0015 and angle<=2.
        # Hysteresis avoids stopping every descent at the 1.5 mm entry boundary.
        if xy>.003 or angle>4 or (self.phase!='insert' and not aligned):
            self.aligned_since=None;self.phase='align'
        elif self.aligned_since is None:self.aligned_since=float(now)
        elif now-self.aligned_since>=.25:self.phase='insert'
        if self.force_paused:
            self.phase='contact_pause';self.aligned_since=None
        # Advance a time-based reference even when the impedance servo lags.
        # The old actual_z - 1 mm clamp could stall at static force equilibrium.
        self.height=min(self.height,float(actual[2]))
        if self.phase=='insert' and self.height>self.floor:
            self.height=max(self.floor,self.height-.003*dt)
        if actual[2]<=self.progress_z-.00025:
            self.progress_z=float(actual[2]);self.progress_time=float(now)
        target[2]=self.height
        return target,{'phase':self.phase,'tip_xy_error_mm':xy*1000,'tip_rotation_error_deg':angle,
            'target_tcp_z_m':self.height,'minimum_target_tcp_z_m':self.floor,
            'tip_height_above_port_mm':float((plug[2]-port[2])*1000),
            'target_lead_mm':max(0.,float(actual[2]-self.height))*1000,
            'force_paused':self.force_paused,'force_baseline_n':self.force_baseline,
            'stalled':bool(self.phase=='insert' and now-self.progress_time>=3 and actual[2]-self.height>.005),
            'descent_limit_reached':self.height<=self.floor}


class AutomaticExpertOperator:
    def __init__(self,after_seconds,max_seconds=EXPERT_MAX_SECONDS):
        if not math.isfinite(after_seconds) or after_seconds<=0:raise ValueError('Expert handoff time must be positive')
        if not math.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Expert time budget must be positive')
        self.after=after_seconds;self.mode='model';self.version=0
        self.max_seconds=max_seconds;self.force_baseline=None
        self.expert_started=None;self.inserted=None;self.reason=None

    def poll(self,status):
        elapsed=float(status.get('elapsed_seconds',0))
        finish=discard=False
        if status.get('request_hold'):
            self.reason='force_or_output_limit';mode='hold';finish=discard=True
        elif status.get('insertion_event'):
            if self.inserted is None:self.inserted=elapsed
            mode='hold';finish=elapsed-self.inserted>=2.
            self.reason='insertion_event_stabilizing'
        elif elapsed>=self.after:
            if self.expert_started is None:
                self.expert_started=elapsed;self.reason='model_time_budget_exhausted'
                self.force_baseline=status.get('force_n')
            age=elapsed-self.expert_started
            mode='human' if age<self.max_seconds else 'hold'
            finish=age>=self.max_seconds+2
            if age>=self.max_seconds:self.reason='expert_time_budget_exhausted'
            if self.force_baseline is not None and status.get('force_n',0)>self.force_baseline+25:
                self.reason='expert_contact_force_limit';mode='hold';finish=discard=True
        else:mode='model'
        if mode!=self.mode:self.mode=mode;self.version+=1
        return {'mode':self.mode,'version':self.version,'axes':[0.]*6,
            'translation_step_m':.0005,'rotation_step_rad':math.radians(.5),
            'finish':finish,'discard':discard,'correction_active':self.mode=='human',
            'takeover_reason':self.reason,'force_n':status.get('force_n')}

    def close(self):pass
