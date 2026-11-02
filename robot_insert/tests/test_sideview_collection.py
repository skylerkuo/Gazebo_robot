import copy
import json
from pathlib import Path
import sqlite3
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/dataset_python'))
from sideview_data import (CAMERAS, camera_geometry, causal_commands, export_episode,
    latest_causal, make_engine, make_robot_description, make_world, successful_score,
    validate_episode, validate_settings)


@pytest.fixture
def settings():
    value = json.loads((Path(__file__).resolve().parents[1]/'configs/sfp0_sideview.json').read_text())
    value['minimum_frames'] = 2
    return value


def records():
    return [{'simulation_time': t, 'camera_times': {name: t for name in CAMERAS},
        'joint_time': t-.002, 'wrench_time': t-.002, 'state': [0]*6,
        'arm_joint_positions': [0]*6, 'wrench': [0]*6, 'tcp_pose': [0, 0, .1, 0, 0, 0, 1]}
        for t in (1., 1.05, 1.10)]


def commands():
    return [{'simulation_time': t, 'pose': [0, 0, z, 0, 0, 0, 1]}
            for t,z in ((.99, .2), (1.04, .19), (1.09, .18))]


def test_camera_geometry_points_towards_target(settings):
    validate_settings(settings)
    c = settings['camera']
    matrix = np.asarray(camera_geometry(c)['world_from_optical'])
    direction = np.asarray(c['look_at_world'])-c['position_world']
    assert np.allclose(matrix[:3, 2], direction/np.linalg.norm(direction))
    assert np.allclose(matrix[:3,:3].T@matrix[:3,:3], np.eye(3))
    assert np.isclose(np.linalg.det(matrix[:3,:3]), 1)


def test_side_camera_addition_preserves_world(tmp_path, settings):
    source=tmp_path/'official.sdf'
    source.write_text('<sdf version="1.9"><world name="aic_world"><model name="existing"/></world></sdf>')
    destination=tmp_path/'modified.sdf'
    make_world(source, destination, settings)
    tree=ET.parse(destination)
    assert tree.find(".//model[@name='existing']") is not None
    assert tree.find(".//sensor[@name='side_camera']/topic").text == '/side_camera/image'
    assert tree.find(".//model[@name='side_camera']/static").text == 'true'
    assert 'side_camera' not in source.read_text()


def test_runtime_speed_can_change_without_changing_dataset_timing_or_physics_step(tmp_path, settings):
    from collect_sfp0_sideview import runtime_settings
    original=copy.deepcopy(settings)
    fast=runtime_settings(settings,1.)
    assert settings==original
    assert {k:v for k,v in fast.items() if k!='real_time_factor'} == {
        k:v for k,v in settings.items() if k!='real_time_factor'}
    source=tmp_path/'official.sdf'
    source.write_text('<sdf version="1.9"><world name="aic_world"><physics name="physics" type="ignored">'
        '<max_step_size>0.001</max_step_size><real_time_factor>0.5</real_time_factor>'
        '</physics></world></sdf>')
    destination=tmp_path/'fast.sdf'
    make_world(source,destination,fast)
    tree=ET.parse(destination)
    assert tree.find('.//physics/real_time_factor').text=='1.0'
    assert tree.find('.//physics/max_step_size').text=='0.001'
    assert tree.find(".//sensor[@name='side_camera']/update_rate").text=='20'
    assert ET.parse(source).find('.//physics/real_time_factor').text=='0.5'


@pytest.mark.parametrize('factor',[0.,-1.,2.,float('nan'),float('inf')])
def test_runtime_speed_rejects_invalid_rates(settings,factor):
    from collect_sfp0_sideview import runtime_settings
    with pytest.raises(ValueError): runtime_settings(settings,factor)


def test_robot_camera_copy_preserves_xacro_macro_namespace(tmp_path, settings):
    source=tmp_path/'robot.xacro'
    source.write_text('<robot><include filename="$(find aic_assets)/models/Basler Camera/basler_camera_macro.xacro"/></robot>')
    camera=tmp_path/'camera.xacro'
    camera.write_text('<robot xmlns:xacro="http://wiki.ros.org/xacro"><xacro:macro name="camera" params="use_sim_cam">'
        '<image><width>1152</width><height>1024</height></image><xacro:if value="${use_sim_cam}"/></xacro:macro></robot>')
    copied=tmp_path/'copied.xacro'
    target=tmp_path/'target.xacro'
    make_robot_description(source,camera,target,copied,settings,'/workspace/results/run')
    assert '<xacro:macro' in copied.read_text()
    assert '<width>576</width>' in copied.read_text()
    assert '/workspace/results/run/recording_camera.xacro' in target.read_text()


def test_causal_alignment_rejects_future_and_stale_sensor():
    assert latest_causal([(1.01, 'future')], 1., .02) is None
    assert latest_causal([(.90, 'stale')], 1., .02) is None
    assert latest_causal([(.99, 'past'), (1.01, 'future')], 1., .02) == (.99, 'past')


def test_command_labels_never_use_future():
    c=commands()
    indices, valid=causal_commands([.98,1.,1.03,1.05,1.1],
        [x['simulation_time'] for x in c], [x['pose'] for x in c])
    assert indices.tolist() == [-1,0,0,1,2]
    assert valid.tolist() == [False,True,True,True,True]


@pytest.mark.parametrize('problem', ['gap','future_camera','stale_force','bad_tcp'])
def test_corrupt_episode_rejected(settings, problem):
    r=records()
    if problem=='gap': r[-1]['simulation_time'] += .2
    if problem=='future_camera': r[1]['camera_times']['side'] += .01
    if problem=='stale_force': r[1]['wrench_time'] -= .2
    if problem=='bad_tcp': r[1]['tcp_pose'][-1]=0
    with pytest.raises(ValueError): validate_episode(r, commands(), settings)


def test_teacher_return_is_not_insertion_success():
    assert not successful_score({'tier_1': {'score':1}, 'tier_3':{'score':0, 'message':'No insertion'}})
    assert successful_score({'tier_1': {'score':1}, 'tier_3':{'score':75, 'message':'Cable insertion successful.'}})


def test_teardown_requires_engine_zero_exit_and_every_trial_scored(tmp_path):
    from sideview_engine_done import completed
    (tmp_path/'scoring').mkdir()
    (tmp_path/'settings.json').write_text(json.dumps({'trial_ids':['episode_000000','episode_000001']}))
    (tmp_path/'scoring/scoring.yaml').write_text('total: 75\nepisode_000000: {}\n')
    (tmp_path/'eval.log').write_text('[INFO] [aic_engine-7]: score saved\n')
    assert not completed(tmp_path)
    (tmp_path/'eval.log').write_text('[INFO] [aic_engine-7]: process has finished cleanly [pid 104]\n')
    assert not completed(tmp_path)
    (tmp_path/'scoring/scoring.yaml').write_text('total: 150\nepisode_000000: {}\nepisode_000001: {}\n')
    assert completed(tmp_path)
    (tmp_path/'eval.log').write_text('[ERROR] [aic_engine-7]: process has died [pid 104, exit code -6]\n')
    with pytest.raises(RuntimeError):completed(tmp_path)


def test_scene_variation_is_reproducible_and_single_task(settings):
    source={'trials':{'trial_1':{'scene':{'task_board':{'pose':{'x':.15,'y':-.2,'z':1.14,'yaw':3.1415}}},
        'tasks':{'task_1':{'target_module_name':'nic_card_mount_0','plug_name':'sfp_tip','port_name':'sfp_port_0'}}}}}
    original=copy.deepcopy(source)
    a,scenes=make_engine(source, [0,1,2], settings)
    b,_=make_engine(source, [0,1,2], settings)
    assert a==b and source==original
    assert scenes['episode_000000']==source['trials']['trial_1']['scene']
    assert scenes['episode_000001']!=scenes['episode_000002']
    for name,trial in a['trials'].items():
        assert list(trial['tasks'])==[name]
        assert trial['tasks'][name]['target_module_name']=='nic_card_mount_0'


def test_export_preserves_images_causal_actions_and_episode_boundaries(tmp_path, settings):
    import pyarrow.parquet as pq
    import cv2
    database=tmp_path/'episode.sqlite3'
    db=sqlite3.connect(database)
    db.execute('CREATE TABLE frames(stamp REAL PRIMARY KEY, metadata TEXT, left_png BLOB, center_png BLOB, right_png BLOB, side_png BLOB)')
    db.execute('CREATE TABLE commands(id INTEGER PRIMARY KEY, stamp REAL, metadata TEXT)')
    pngs=[cv2.imencode('.png', np.full((4,5,3), i*40, dtype=np.uint8))[1].tobytes() for i in range(4)]
    for r in records():db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?)',(r['simulation_time'],json.dumps(r),*pngs))
    for c in commands():db.execute('INSERT INTO commands(stamp,metadata) VALUES (?,?)',(c['simulation_time'],json.dumps(c)))
    db.commit();db.close()
    output=tmp_path/'episode.parquet'
    report=export_episode(database,output,episode=7,offset=100,settings=settings)
    rows=pq.read_table(output).to_pylist()
    assert report['frames']==3
    features=json.loads(pq.read_schema(output).metadata[b'huggingface'])['info']['features']
    assert features['observation.images.side_camera']['_type']=='Image'
    assert features['observation.tcp_pose']['feature']['dtype']=='float32'
    assert [r['episode_index'] for r in rows]==[7]*3
    assert [r['index'] for r in rows]==[100,101,102]
    assert np.allclose([r['action'][2] for r in rows],[.2,.19,.18])
    for r in rows:
        assert r['action_time']<=r['simulation_time']
        for i,name in enumerate(CAMERAS):assert r[f'observation.images.{name}_camera']['bytes']==pngs[i]


def test_buffered_writer_exports_all_four_rgb_streams_without_sqlite_image_blobs(tmp_path, settings):
    import cv2
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from sideview_writer import EpisodeWriter
    settings.update(record_width=5,record_height=4)
    messages=[]
    for camera in range(4):
        rgb=(np.arange(60,dtype=np.uint8).reshape(4,5,3)+camera*40)
        messages.append(SimpleNamespace(encoding='rgb8',width=5,height=4,step=15,data=rgb.tobytes()))
    writer=EpisodeWriter(tmp_path,'episode',settings)
    for c in commands():writer.put(('command',c))
    for r in records():writer.put(('frame',r,messages))
    writer.finish()
    assert writer.frames==3 and writer.commands==3 and writer.failure is None
    database=tmp_path/'episode.sqlite3'
    with sqlite3.connect(database) as db:
        raw=db.execute('SELECT left_png,center_png,right_png,side_png,image_offsets FROM frames').fetchall()
    assert all(row[:4]==(None,)*4 and len(json.loads(row[4]))==4 for row in raw)
    destination=tmp_path/'episode.parquet'
    report=export_episode(database,destination,episode=0,offset=0,settings=settings)
    assert report['frames']==3 and report['maximum_frame_gap_s']==pytest.approx(.05)
    rows=pq.read_table(destination).to_pylist()
    assert [row['index'] for row in rows]==[0,1,2]
    for row in rows:
        for camera,name in enumerate(CAMERAS):
            encoded=row[f'observation.images.{name}_camera']['bytes']
            bgr=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(bgr[...,::-1],np.frombuffer(messages[camera].data,np.uint8).reshape(4,5,3))
    with (tmp_path/'episode.images').open('r+b') as stream:stream.truncate(10)
    with pytest.raises(ValueError,match='Truncated packed PNG'):
        export_episode(database,tmp_path/'truncated.parquet',0,0,settings)


def test_collection_validator_checks_official_score_and_future_actions(tmp_path, settings):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import cv2
    from sideview_data import json_write
    from collect_sfp0_sideview import refresh_metadata
    from validate_sideview_collection import validate
    settings['record_width'],settings['record_height']=5,4
    database=tmp_path/'recording.sqlite3'
    db=sqlite3.connect(database)
    db.execute('CREATE TABLE frames(stamp REAL PRIMARY KEY, metadata TEXT, left_png BLOB, center_png BLOB, right_png BLOB, side_png BLOB)')
    db.execute('CREATE TABLE commands(id INTEGER PRIMARY KEY, stamp REAL, metadata TEXT)')
    png=cv2.imencode('.png',np.zeros((4,5,3),dtype=np.uint8))[1].tobytes()
    for r in records():db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?)',(r['simulation_time'],json.dumps(r),png,png,png,png))
    for c in commands():db.execute('INSERT INTO commands(stamp,metadata) VALUES (?,?)',(c['simulation_time'],json.dumps(c)))
    db.commit();db.close()
    output=tmp_path/'dataset'
    parquet=output/'data/chunk-000/file-000.parquet'
    report=export_episode(database,parquet,0,0,settings)
    entry={'accepted':True,'episode_index':0,'parquet':'data/chunk-000/file-000.parquet','validation':report,
           'official_score':{'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}}}
    manifest={'attempts':[entry],'settings':settings}
    refresh_metadata(output,manifest,settings)
    assert validate(output)['passed']
    manifest['attempts'][0]['official_score']['tier_3']['score']=0
    json_write(output/'_collection/manifest.json',manifest)
    with pytest.raises(ValueError,match='official'):validate(output)
    manifest['attempts'][0]['official_score']['tier_3']['score']=75
    json_write(output/'_collection/manifest.json',manifest)
    table=pq.read_table(parquet)
    table=table.set_column(table.schema.get_field_index('action_time'),'action_time',pa.array([5.,5.,5.]))
    pq.write_table(table,parquet)
    with pytest.raises(ValueError,match='Future command'):validate(output,verify_hashes=False)
