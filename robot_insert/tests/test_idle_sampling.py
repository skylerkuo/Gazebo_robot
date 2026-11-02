"""Idle pruning keeps contact, rotation, short pauses, and original frame timing."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from idle_sampling import filter_idle_anchors


def signals(n=100):
    a=np.zeros((n,7));a[:,6]=1
    return a,np.zeros((n,6)),np.arange(n)*.05,list(range(n-20))


def test_long_idle_keeps_edges_without_reindexing_or_splicing():
    a,w,t,anchors=signals()
    kept,report=filter_idle_anchors(anchors,a,w,t)
    assert kept==list(range(5))+list(range(75,80))
    assert report['removed_idle_anchors']==70
    assert np.allclose(np.diff(t),.05) and a.shape==(100,7)


def test_commands_moving_against_contact_and_changing_force_are_preserved():
    a,w,t,anchors=signals();a[:,2]=np.arange(100)*.00005
    assert filter_idle_anchors(anchors,a,w,t)[0]==anchors
    a[:,2]=0;w[:,0]=np.arange(100)*.1
    assert filter_idle_anchors(anchors,a,w,t)[0]==anchors
    w[:]=0;w[:,5]=np.arange(100)*.01
    assert filter_idle_anchors(anchors,a,w,t)[0]==anchors


def test_rotation_and_short_waits_preserved_and_quaternion_sign_is_equivalent():
    a,w,t,anchors=signals(35)
    assert filter_idle_anchors(anchors,a,w,t)[0]==anchors
    a,w,t,anchors=signals();angle=np.deg2rad(np.arange(100)*.05)
    a[:,3]=np.sin(angle/2);a[:,6]=np.cos(angle/2)
    assert filter_idle_anchors(anchors,a,w,t)[0]==anchors
    a[:]=[0,0,0,0,0,0,1];a[::2,3:]*=-1
    assert len(filter_idle_anchors(anchors,a,w,t)[0])==10
