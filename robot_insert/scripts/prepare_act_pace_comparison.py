"""Freeze one assisted-rollout index shared by ACT and four-view PACE."""
import copy
import fcntl
import json
import os
from pathlib import Path

from act_multiview import read_index,index_digest,file_digest
from prepare_expert_only import prepare_expert_only,inspect_expert_collection
from sideview_data import json_write
from storage_checks import ensure_free_space,GIB

DATA_DRIVE_ROOT=Path('/mnt/d')


def comparison_readiness(config):
    path=Path(config.comparison_index_path) if config.comparison_index_path else None
    if path is not None and path.is_file():
        index=read_index(path,verify_files=False)
        val=[e for e in index['episodes'] if e['split']=='validation']
        available=sum(len(e.get('expert_validation_frames',e.get('train_anchor_frames',[]))) for e in val)
        return {'accepted_episodes':len(index['episodes']),'train_episodes':index['counts']['train']['episodes'],
            'validation_episodes':len(val),'validation_episode_ids':[e['episode_index'] for e in val],
            'eligible_validation_anchors':available,'required_validation_anchors':config.eval_samples,
            'fixed_validation_anchors':index['validation_samples'],
            'ready':index['validation_samples']==config.eval_samples,
            'frozen_snapshot':True,'comparison_index':str(path),'data_index_sha256':index['index_content_sha256']}
    return dict(inspect_expert_collection(config.dataset_root,eval_samples=config.eval_samples,
        seed=config.seed,retain_model_motion=config.retain_model_motion)['report'],frozen_snapshot=False)

def prepare_comparison(config,initial):
    path=Path(config.comparison_index_path).resolve()
    if not path.is_relative_to(DATA_DRIVE_ROOT):raise ValueError('Comparison snapshot must be on D:')
    if not path.exists():ensure_free_space(path.parent,2*GIB)
    path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        contract=path.parent/'comparison_contract.json'
        digest=initial.get('_initialization_sha256') or file_digest(config.init_from)
        if path.exists():
            saved=json.loads(contract.read_text())
            if saved['initial_checkpoint_sha256']!=digest:raise ValueError('ACT/PACE comparison requires the exact same initialization checkpoint')
            index=read_index(path)
            if index['index_content_sha256']!=saved['index_content_sha256']:raise ValueError('Comparison snapshot checksum differs from its contract')
        else:
            original=prepare_expert_only(config.dataset_root,initial,eval_samples=config.eval_samples,
                seed=config.seed,retain_model_motion=config.retain_model_motion)
            value=copy.deepcopy({k:v for k,v in original.items() if k!='_directory'})
            for episode in value['episodes']:
                for kind in ('numeric','images','motion'):
                    if kind in episode:
                        episode[kind]['path']=os.path.relpath(Path(original['_directory'])/episode[kind]['path'],path.parent)
            value['index_content_sha256']=index_digest(value)
            json_write(contract,{'initial_checkpoint_sha256':digest,'initial_checkpoint':config.init_from,
                'index_content_sha256':value['index_content_sha256'],'policy_variants':['act','pace'],
                'data_policy':'equal model/expert valid targets; boolean stationary mask; expert-only VAL'})
            json_write(path,value)
            index=read_index(path,verify_files=False)
        if (index['stats']!=initial['stats'] or index['normalization_parent_index_sha256']!=initial['data_index_sha256']
                or index['validation_samples']!=config.eval_samples or index['preparation_settings']['seed']!=config.seed
                or Path(index['dataset_root'])!=Path(config.dataset_root).resolve()):
            raise ValueError('Comparison snapshot settings/statistics differ; use a new comparison_index_path for a new experiment')
        return index


def main():
    import argparse
    from ftact import INSERT_ROOT
    from train_sfp0_act_multiview import load_initialization
    from act_multiview import MultiViewConfig
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=INSERT_ROOT/'configs/sfp0_act_expert_finetune.json')
    parser.add_argument('--check-only',action='store_true',help='Readiness only; no model/data cache or training writes')
    args=parser.parse_args();values=json.loads(args.config.read_text())
    if values.get('policy_type')=='flow':
        from flow_multiview import MultiViewFlowConfig
        config=MultiViewFlowConfig(**values)
    elif values.get('policy_type')=='pace_pred':
        from pace_pred_multiview import MultiViewPACEPredConfig
        config=MultiViewPACEPredConfig(**values)
    elif values.get('policy_type')=='pace':
        from pace_multiview import MultiViewPACEConfig
        config=MultiViewPACEConfig(**values)
    else:config=MultiViewConfig(**values)
    from dataclasses import replace
    replace(config,device='cpu',amp='none').validate()
    if args.check_only:
        report=comparison_readiness(config)
        print(json.dumps(report,indent=2));return 0 if report['ready'] else 2
    if not config.comparison_index_path:parser.error('Set comparison_index_path in the config')
    initial,_=load_initialization(config.init_from)
    try:index=prepare_comparison(config,initial)
    except (ValueError,OSError) as error:parser.exit(2,f'Comparison preparation stopped: {error}\n')
    print(json.dumps({'comparison_index':str(Path(index['_directory'])/'index.json'),
        'data_index_sha256':index['index_content_sha256'],'counts':index['counts'],
        'validation_samples':index['validation_samples']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
