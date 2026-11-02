"""Keep learner motion and expert recovery, masking redundant stationary targets."""
import math
import numpy as np

from human_control import expert_anchor_frames
from idle_sampling import DEFAULT_IDLE_FILTER,filter_idle_anchors

ROLLOUT_FILTER={'version':2,'supervision_mode':'equal_valid_steps',
    'tcp_position_span_m':.0005,'tcp_rotation_span_deg':.2,
    'force_trend_n':2.,'torque_trend_nm':.5,'minimum_duration_s':2.,'boundary_samples':5}


def sample_rollout_file(path):
    import pyarrow.parquet as pq
    a=pq.read_table(path,columns=['action_source','action_segment','human_active','simulation_time',
        'action','observation.environment_state','observation.tcp_pose']).to_pydict()
    weights,train,val,report=rollout_windows(a['action_source'],a['action_segment'],a['human_active'],a['simulation_time'],
        a['action'],a['observation.environment_state'],a['observation.tcp_pose'])
    return np.asarray(a['observation.tcp_pose'],dtype=np.float32),weights,train,val,report


def rollout_windows(sources,segments,active,times,actions,wrench,tcp,settings=None):
    cfg=dict(ROLLOUT_FILTER if settings is None else settings)
    if cfg.get('version') not in (1,2):raise ValueError('Unsupported rollout-filter version')
    model_weight=cfg['model_weight'] if cfg['version']==1 else 1.
    expert_weight=cfg['expert_weight'] if cfg['version']==1 else 1.
    if not all(math.isfinite(v) and v>0 for v in (model_weight,expert_weight)):raise ValueError('Invalid legacy action weights')
    for key in ('tcp_position_span_m','tcp_rotation_span_deg','force_trend_n','torque_trend_nm','minimum_duration_s'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('Invalid rollout sampling setting')
    edge=cfg['boundary_samples']
    if not isinstance(edge,int) or edge<1:raise ValueError('Keep at least one boundary sample')
    sources=np.asarray(sources);segments=np.asarray(segments);times=np.asarray(times,dtype=float)
    actions=np.asarray(actions,dtype=float);wrench=np.asarray(wrench,dtype=float);tcp=np.asarray(tcp,dtype=float)
    n=len(times)
    if (sources.shape!=(n,) or segments.shape!=(n,) or actions.shape!=(n,7) or wrench.shape!=(n,6)
            or tcp.shape!=(n,7) or not np.isfinite(tcp).all()):raise ValueError('Invalid rollout signals')
    if not np.isin(sources,['model','expert','hold']).all():raise ValueError('Unexpected rollout action source')
    weights=np.where(sources=='expert',expert_weight,np.where(sources=='model',model_weight,0.)).astype(np.float32)
    continuous=[f for f in range(max(0,n-20)) if np.allclose(np.diff(times[f:f+21]),.05,rtol=0,atol=1e-5)]
    stationary=[]
    for f in continuous:
        window=slice(f,f+21)
        if not (np.all(sources[window]=='model') and np.all(segments[window]==segments[f])):continue
        q=tcp[window,3:];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
        rotation=np.rad2deg(2*np.arccos(np.clip(np.abs(q@q[0]),0,1))).max()
        trend=np.median(wrench[f+16:f+21],axis=0)-np.median(wrench[f:f+5],axis=0)
        if (np.linalg.norm(np.ptp(tcp[window,:3],axis=0))<=cfg['tcp_position_span_m'] and rotation<=cfg['tcp_rotation_span_deg']
                and np.linalg.norm(trend[:3])<=cfg['force_trend_n'] and np.linalg.norm(trend[3:])<=cfg['torque_trend_nm']):
            stationary.append(f)
    runs=[]
    for f in stationary:
        if not runs or f!=runs[-1][-1]+1:runs.append([f])
        else:runs[-1].append(f)
    removed_model=[]
    for run in runs:
        if times[run[-1]+20]-times[run[0]]+1e-8>=cfg['minimum_duration_s']:removed_model+=run[edge:-edge]
    weights[removed_model]=0
    expert=expert_anchor_frames(sources,segments,active,times,source='expert')
    kept_expert,_=filter_idle_anchors(expert,actions,wrench,times,DEFAULT_IDLE_FILTER)
    removed_expert=sorted(set(expert)-set(kept_expert));weights[removed_expert]=0
    train=[f for f in continuous if np.any(weights[f+1:f+21]>0)]
    val=[f for f in expert if np.all(weights[f+1:f+21]==expert_weight)]
    return weights,train,val,{'model_frames':int(np.sum(sources=='model')),
        'retained_model_targets':int(np.sum((sources=='model')&(weights>0))),
        'masked_stationary_model_targets':len(removed_model),'masked_stationary_expert_targets':len(removed_expert),
        'expert_targets':int(np.sum((sources=='expert')&(weights>0))),
        'train_windows':len(train),'expert_validation_windows':len(val),
        'handoff_windows':sum(bool(np.any(sources[f+1:f+21]=='model') and np.any(sources[f+1:f+21]=='expert')) for f in train)}
