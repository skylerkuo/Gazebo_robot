"""Moving model prefixes and handoff targets survive; idle targets have no gradient."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rollout_sampling import ROLLOUT_FILTER,rollout_windows
from act_objectives import common_scale_loss


def signals():
    n=140;s=np.array(['model']*80+['expert']*40+['hold']*20)
    segments=np.array([1]*80+[2]*40+[3]*20);active=s=='expert'
    t=np.arange(n)*.05;a=np.zeros((n,7));a[:,6]=1;a[:,0]=np.arange(n)*.0001
    tcp=a.copy();tcp[20:80,0]=.002
    return s,segments,active,t,a,np.zeros((n,6)),tcp


def test_keep_model_motion_and_cross_handoff_without_splicing_idle():
    args=signals();weights,train,val,report=rollout_windows(*args)
    assert weights[0]==1 and weights[30]==0 and weights[90]==1 and weights[130]==0
    assert 30 not in train and 75 in train and report['handoff_windows']>0
    np.testing.assert_array_equal(weights[76:96],[1.]*20)
    assert all(np.all(args[0][f+1:f+21]=='expert') for f in val)
    assert all(np.all(weights[f+1:f+21]==1) for f in val)
    assert np.allclose(np.diff(args[3]),.05) and len(args[3])==140
    assert report['masked_stationary_model_targets']>0 and report['retained_model_targets']>0


def test_contact_force_progress_is_retained_even_with_stationary_tcp():
    args=list(signals());args[5][:,0]=np.arange(140)*.2
    weights,_,_,report=rollout_windows(*args)
    assert report['masked_stationary_model_targets']==0 and weights[30]==1


def test_no_future_chunk_bridges_recording_gap():
    args=list(signals());args[3][70:]+=.1
    _,train,val,_=rollout_windows(*args)
    assert all(not f<70<=f+20 for f in train+val)


def test_equal_model_expert_gradients_and_stationary_mask():
    class Normalizer:
        def unnormalize_actions(self,x):return x
        def actions(self,x):return x
    target=torch.ones(1,3,7)*.01
    prediction=torch.zeros_like(target,requires_grad=True)
    loss,_=common_scale_loss(prediction,target,Normalizer(),action_weights=torch.tensor([[True,True,False]]))
    loss.backward();g=prediction.grad
    torch.testing.assert_close(g[:,0],g[:,1])
    assert torch.count_nonzero(g[:,2])==0 and torch.isfinite(loss)
    unweighted,_=common_scale_loss(prediction,target,Normalizer())
    ones,_=common_scale_loss(prediction,target,Normalizer(),action_weights=torch.ones(1,3))
    torch.testing.assert_close(unweighted,ones)
    with pytest.raises(ValueError):common_scale_loss(prediction,target,Normalizer(),action_weights=torch.zeros(1,3))


def test_old_frozen_indexes_remain_readable_without_changing_new_equal_sampling():
    settings=dict(ROLLOUT_FILTER,version=1,model_weight=.25,expert_weight=1.)
    legacy,*_=rollout_windows(*signals(),settings=settings)
    current,*_=rollout_windows(*signals())
    assert legacy[0]==.25 and current[0]==current[90]==1
