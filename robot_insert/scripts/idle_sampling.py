"""Remove redundant interior idle anchors, preserving continuous action chunks."""
import math
import numpy as np

DEFAULT_IDLE_FILTER={'version':1,'position_span_m':.0002,'rotation_span_deg':.2,
    'force_span_n':1.,'torque_span_nm':.05,'minimum_duration_s':2.,'boundary_samples':5}


def filter_idle_anchors(anchors,actions,wrench,times,settings=None):
    settings=dict(DEFAULT_IDLE_FILTER if settings is None else settings)
    if settings.get('version')!=1:raise ValueError('Unsupported idle filter version')
    for key in ('position_span_m','rotation_span_deg','force_span_n','torque_span_nm','minimum_duration_s'):
        if not math.isfinite(settings[key]) or settings[key]<=0:raise ValueError('Invalid idle filter threshold')
    edge=settings['boundary_samples']
    if not isinstance(edge,int) or edge<1:raise ValueError('Idle boundaries must retain at least one sample')
    actions=np.asarray(actions,dtype=float);wrench=np.asarray(wrench,dtype=float);times=np.asarray(times,dtype=float)
    if actions.shape!=(len(times),7) or wrench.shape!=(len(times),6):raise ValueError('Idle-filter signal lengths differ')
    idle=[]
    for f in anchors:
        a=actions[f:f+21];w=wrench[f:f+21]
        if len(a)!=21 or not np.allclose(np.diff(times[f:f+21]),.05,rtol=0,atol=1e-5):
            raise ValueError('Idle filter cannot bridge a gap or episode boundary')
        q=a[:,3:];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
        angle=np.rad2deg(2*np.arccos(np.clip(np.abs(q@q[0]),0,1))).max()
        if (np.linalg.norm(np.ptp(a[:,:3],axis=0))<=settings['position_span_m'] and
                angle<=settings['rotation_span_deg'] and
                np.linalg.norm(np.ptp(w[:,:3],axis=0))<=settings['force_span_n'] and
                np.linalg.norm(np.ptp(w[:,3:],axis=0))<=settings['torque_span_nm']):idle.append(f)
    runs=[]
    for f in idle:
        if not runs or f!=runs[-1][-1]+1:runs.append([f])
        else:runs[-1].append(f)
    removed=set()
    for run in runs:
        if times[run[-1]+20]-times[run[0]]+1e-8>=settings['minimum_duration_s']:
            removed.update(run[edge:-edge])
    kept=[f for f in anchors if f not in removed]
    return kept,{'before':len(anchors),'after':len(kept),'removed_idle_anchors':len(removed),
        'rule':'Only long near-constant command AND force/torque windows; keep boundary samples. No timestamp splicing.'}
