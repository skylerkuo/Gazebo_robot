"""Public releases must preserve provenance and exclude private/runtime files."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_hf_release import select_entries,assert_portable,dump,digest,append_membership,REPO
from publish_hf_dataset import (inventory,verify_remote,verify_partial_release,upload_batches,
    validate_update_files,verify_update_release)
from types import SimpleNamespace as NS


def entry(ep=0,controller='model_and_official_expert'):
    return {'episode_index':ep,'accepted':True,'parquet':f'data/chunk-000/file-{ep:03d}.parquet',
            'official_score':{'tier_1':{'score':1},'tier_3':{'score':75,'message':'Cable insertion successful.'}},
            'recorder':{'controller':controller,'policy_return':True,'writer_error':None}}


def test_release_preserves_model_prefix_and_also_includes_short_expert_or_act_only_success():
    assisted=entry();assisted['expert_validation_frames']=[]
    direct=entry(1,'official_cheatcode_only')
    rejected=entry(2);rejected['accepted']=False
    manifest={'attempts':[direct,rejected,assisted]}
    before=copy.deepcopy(manifest)
    assert select_entries(manifest)==[assisted,direct]
    assert manifest==before


@pytest.mark.parametrize('change',[
    {'parquet':'../secret.parquet'}, {'parquet':'/mnt/d/source.parquet'},
    {'operator_discarded':True}, {'test_run':True},
    {'official_score':{'tier_1':{'score':1},'tier_3':{'score':0}}},
    {'recorder':{'controller':'unknown','policy_return':True}},
    {'recorder':{'controller':'official_cheatcode_only','policy_return':False}},
])
def test_invalid_public_episode_rejected(change):
    e=entry();e.update(change)
    with pytest.raises(ValueError):select_entries({'attempts':[e]})


def test_private_host_paths_are_rejected_but_ros_frame_names_are_allowed():
    for p in ('/mnt/d/robot_models/best.pt','/home/user/data','D:\\private','/workspace/results/user'):
        with pytest.raises(ValueError):assert_portable({'nested':[{'path':p}]})
    assert_portable({'topic':'/fts_broadcaster/wrench','frame':'left_camera/optical','url':'https://huggingface.co'})


def test_upload_uses_frozen_allowlist_not_unknown_files_and_rejects_modified_metadata(tmp_path):
    dump(tmp_path/'meta/READY.json',{'complete':True})
    doc=tmp_path/'README.md';doc.write_text('public dataset card')
    dump(tmp_path/'meta/files.json',[{'path':'README.md','bytes':doc.stat().st_size,'sha256':digest(doc)}])
    (tmp_path/'token').write_text('must not be uploaded')
    (tmp_path/'checkpoint.pt').write_text('must not be uploaded')
    files=inventory(tmp_path)
    assert {f['path'] for f in files}=={'README.md','meta/files.json','meta/READY.json'}
    doc.write_text('changed dataset card')
    with pytest.raises(ValueError):inventory(tmp_path)


def test_remote_verification_checks_public_visibility_size_and_lfs_sha():
    expected=[{'path':'data/chunk-000/file-000.parquet','bytes':100,'sha256':'a'*64}]
    remote=NS(private=False,sha='revision',siblings=[NS(rfilename=expected[0]['path'],size=100,lfs=NS(sha256='a'*64))])
    api=NS(dataset_info=lambda *args,**kwargs:remote)
    assert verify_remote(api,expected)['parquet_files']==1
    remote.siblings[0].lfs.sha256='b'*64
    with pytest.raises(ValueError,match='checksum'):verify_remote(api,expected)
    remote.private=True
    with pytest.raises(ValueError,match='public'):verify_remote(api,expected)


def test_partial_resume_requires_matching_bytes_and_checksums_before_snapshot_commit():
    files=[{'path':'README.md','bytes':10,'sha256':'unused','git_blob_sha1':'a'*40},
           {'path':'data/one.parquet','bytes':100,'sha256':'b'*64}]
    info=NS(private=False,siblings=[NS(rfilename='README.md',size=10,lfs=None,blob_id='a'*40),
        NS(rfilename='data/one.parquet',size=100,lfs=NS(sha256='b'*64))])
    verify_partial_release(info,files)
    info.siblings[0].blob_id='c'*40
    with pytest.raises(ValueError,match='checksum'):verify_partial_release(info,files)
    info.siblings[0].blob_id='a'*40;info.siblings[1].size=101
    with pytest.raises(ValueError,match='not this frozen'):verify_partial_release(info,files)
    info.siblings[1].size=100;info.siblings.append(NS(rfilename='other.txt',size=1))
    with pytest.raises(ValueError,match='not this frozen'):verify_partial_release(info,files)


def test_batch_failure_keeps_committed_episodes_and_resume_only_uploads_missing_files(tmp_path):
    files=[{'path':f'data/{i}.parquet','bytes':100,'sha256':str(i)*64} for i in range(5)]
    def sibling(item):
        return NS(rfilename=item['path'],size=item['bytes'],lfs=NS(sha256=item['sha256']))
    remote=NS(private=False,sha='revision',siblings=[sibling(files[0])])
    calls=[]
    failing=True
    def upload(**kwargs):
        nonlocal failing
        selected=kwargs['allow_patterns'];calls.append(selected)
        assert len(selected)<=2 and files[0]['path'] not in selected
        if failing and files[3]['path'] in selected:
            raise RuntimeError('simulated interrupted transfer')
        remote.siblings.extend(sibling(f) for f in files if f['path'] in selected)
    api=NS(dataset_info=lambda *a,**k:remote,upload_large_folder=upload)
    folder=tmp_path/'release';folder.mkdir()
    with pytest.raises(RuntimeError,match='interrupted'):
        upload_batches(api,folder,files,remote,batch_size=2)
    saved=json.loads((tmp_path/'release-progress.json').read_text())
    assert saved['committed_episodes']==3 and saved['total_episodes']==5
    assert len(remote.siblings)==3
    failing=False
    upload_batches(api,folder,files,remote,batch_size=2)
    assert calls==[['data/1.parquet','data/2.parquet'],['data/3.parquet','data/4.parquet'],
                  ['data/3.parquet','data/4.parquet']]
    saved=json.loads((tmp_path/'release-progress.json').read_text())
    assert saved['committed_episodes']==5 and saved['committed_bytes']==500
    assert verify_remote(api,files)['verified_files']==5


def test_append_membership_preserves_existing_episode_ids_paths_and_hashes():
    old={'repo_id':REPO,'files':[{'episode_index':0,'path':'data/0.parquet','bytes':10,'sha256':'a'}]}
    new=copy.deepcopy(old)
    added={'episode_index':1,'path':'data/1.parquet','bytes':20,'sha256':'b'}
    new['files'].append(added)
    assert append_membership(old,new)==[added]
    for key,value in [('sha256','changed'),('path','data/moved.parquet'),('bytes',11)]:
        bad=copy.deepcopy(new);bad['files'][0][key]=value
        with pytest.raises(ValueError,match='remove or change'):append_membership(old,bad)
    bad=copy.deepcopy(new);bad['files']=bad['files'][1:]
    with pytest.raises(ValueError,match='remove or change'):append_membership(old,bad)
    bad=copy.deepcopy(new);bad['files'][1]['path']=bad['files'][0]['path']
    with pytest.raises(ValueError,match='Duplicate episode paths'):append_membership(old,bad)


def test_update_rejects_data_replacement_and_unrecognized_remote_files():
    old=[{'path':'data/0.parquet','bytes':100,'sha256':'a'*64}]
    new=copy.deepcopy(old)
    new[0]['sha256']='b'*64
    with pytest.raises(ValueError,match='cannot replace'):validate_update_files(old,new)
    with pytest.raises(ValueError,match='cannot delete'):validate_update_files(old,[])
    remote=NS(private=False,siblings=[NS(rfilename='unrelated.txt',size=1,lfs=None,blob_id='unknown')])
    with pytest.raises(ValueError,match='does not match'):verify_update_release(remote,old,old)


def test_update_uploads_only_new_content_then_commits_all_metadata_together_and_resumes(tmp_path):
    folder=tmp_path/'release';folder.mkdir()
    def item(path,content):
        p=folder/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        return {'path':path,'bytes':len(content),'sha256':digest(p)}
    old_data=item('data/0.parquet',b'old data')
    old_card=item('README.md',b'old card')
    old_info=item('meta/info.json',b'{"episodes":1}')
    previous=[old_data,old_card,old_info]
    new_data=item('data/1.parquet',b'new data')
    new_card=item('README.md',b'new card')
    new_info=item('meta/info.json',b'{"episodes":2}')
    files=[old_data,new_data,new_card,new_info]
    def sibling(f):
        return NS(rfilename=f['path'],size=f['bytes'],lfs=NS(sha256=f['sha256']))
    remote=NS(private=False,sha='parent-head',siblings=[sibling(f) for f in previous])
    uploads=[];commits=[];fail_commit=True
    def upload(**kw):
        assert kw['allow_patterns']==['data/1.parquet']
        uploads.append(kw['allow_patterns'])
        remote.siblings.append(sibling(new_data));remote.sha='data-head'
    def commit(**kw):
        assert remote.sha==kw['parent_commit']=='data-head'
        assert any(s.rfilename=='data/1.parquet' for s in remote.siblings)
        names={op.path_in_repo for op in kw['operations']}
        assert names=={'README.md','meta/info.json'}
        commits.append(names)
        if fail_commit:raise RuntimeError('connection lost before metadata commit')
        remote.siblings=[s for s in remote.siblings if s.rfilename not in names]
        remote.siblings.extend(sibling(f) for f in files if f['path'] in names)
        remote.sha='complete-head'
    api=NS(dataset_info=lambda *a,**kw:remote,upload_large_folder=upload,create_commit=commit)
    with pytest.raises(RuntimeError,match='connection lost'):
        upload_batches(api,folder,files,remote,previous=previous)
    assert next(s for s in remote.siblings if s.rfilename=='README.md').lfs.sha256==old_card['sha256']
    fail_commit=False
    upload_batches(api,folder,files,remote,previous=previous)
    assert len(uploads)==1 and len(commits)==2
    assert verify_remote(api,files)['revision']=='complete-head'
    upload_batches(api,folder,files,remote,previous=previous)
    assert len(uploads)==1 and len(commits)==2


def test_supervisor_requires_complete_matching_upload_report(tmp_path):
    from supervise_hf_update import valid_report
    folder=tmp_path/'release';folder.mkdir()
    files=[{'path':'data/1.parquet','bytes':100},{'path':'README.md','bytes':10}]
    assert not valid_report(folder,files)
    report={'repo_id':REPO,'public':True,'revision':'head','verified_files':2,'parquet_files':1,'bytes':110}
    path=tmp_path/'release-upload-result.json'
    dump(path,report);assert valid_report(folder,files)
    for key,value in [('repo_id','another/repo'),('public',False),('verified_files',1),('parquet_files',0),('bytes',109)]:
        dump(path,dict(report,**{key:value}))
        assert not valid_report(folder,files)
