"""Operator intent, takeover boundaries and human-only future action labels."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from human_control import OperatorState,OperatorServer,OperatorLink,manual_pose_step,expert_anchor_frames


def test_manual_keys_only_move_in_human_mode_and_mode_change_clears_keys():
    state=OperatorState();state.press('w')
    assert state.snapshot()['axes']==[0.]*6
    state.set_mode('human');version=state.snapshot()['version'];state.press('w');state.press('r');state.press('i')
    control=state.snapshot();assert control['axes']==[1.,0.,1.,1.,0.,0.]
    pose=manual_pose_step([0,0,.2,0,0,0,1],control,[-1]*3,[1]*3)
    np.testing.assert_allclose(pose[:3],[.0005,0,.2005])
    assert np.linalg.norm(pose[3:])==pytest.approx(1.) and pose[3]>0
    state.release('w');assert state.snapshot()['axes'][0]==0
    state.set_mode('hold');assert state.snapshot()['version']>version and not any(state.snapshot()['axes'])
    state.finish_episode(True);assert state.snapshot()['finish'] and state.snapshot()['discard']


def test_manual_jog_is_bounded_even_after_a_long_gap():
    state=OperatorState();state.set_mode('human');state.press('w')
    pose=manual_pose_step([.9999,0,0,0,0,0,1],state.snapshot(),[-1]*3,[1]*3,ticks=100)
    assert pose[0]==1.


def test_local_operator_link_delivers_mode_and_force_hold_without_model_control():
    server=OperatorServer();link=OperatorLink(server.endpoint())
    try:
        server.state.set_mode('human');server.state.press('f')
        control=link.poll({'frames':42})
        assert control['mode']=='human' and control['axes'][2]==-1
        assert server.state.telemetry['frames']==42
        control=link.poll({'request_hold':True})
        assert control['mode']=='hold' and not any(control['axes'])
        server.state.finish_episode();assert link.poll({})['finish']
    finally:link.close();server.close()


def test_operator_disconnect_falls_back_to_hold():
    server=OperatorServer();endpoint=server.endpoint();server.close()
    link=OperatorLink(endpoint)
    try:
        control=link.poll({})
        assert control['mode']=='hold' and control['link_lost'] and not any(control['axes'])
    finally:link.close()


def test_expert_windows_never_include_model_hold_or_multiple_takeover_segments():
    sources=np.array(['model']*5+['human']*30+['hold']*5)
    segments=np.array([0]*5+[1]*30+[2]*5)
    active=np.array([False]*5+[True]*30+[False]*5)
    times=np.arange(40)*.05
    anchors=expert_anchor_frames(sources,segments,active,times)
    assert anchors==list(range(4,15))
    for f in anchors:
        assert np.all(sources[f+1:f+21]=='human') and len(set(segments[f+1:f+21]))==1
    segments[20:35]=3
    assert expert_anchor_frames(sources,segments,active,times)==[]


def test_expert_windows_exclude_idle_only_and_dropped_frame_intervals():
    source=['human']*45;segments=[1]*45;active=[False]*45;times=np.arange(45)*.05
    assert expert_anchor_frames(source,segments,active,times)==[]
    active[22]=True
    anchors=expert_anchor_frames(source,segments,active,times)
    assert anchors and all(f+1<=22<=f+20 for f in anchors)
    times[23:]+=.05
    anchors=expert_anchor_frames(source,segments,active,times)
    assert all(not f<23<=f+20 for f in anchors)
