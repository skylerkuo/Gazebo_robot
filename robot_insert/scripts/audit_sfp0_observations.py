#!/usr/bin/env python3
"""Read-only ACT error/observation ambiguity audit; never modifies data or training.

Normalized clip time is NOT a ground-truth task phase. Similar observations
are approximate neighbors, not proof of multiple valid strategies or CVAE need.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

from ftact import CAMERAS, STATE, WRENCH, json_write, move_batch
from ftact_v2 import TaskDataset, read_index, load_sfp_checkpoint
from train_ftact import autocast_context
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


def matrix(column, width):
    return column.combine_chunks().values.to_numpy().reshape(-1, width).copy()


def balanced_train_anchors(index, count=1200, seed=1729):
    segments = sorted((s for s in index['segments'] if s['split'] == 'train'), key=lambda s:s['episode_index'])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(segments))
    quotas = np.full(len(segments), count//len(segments), dtype=int)
    quotas[order[:count % len(segments)]] += 1
    anchors = []
    for s, quota in zip(segments, quotas):
        frames = np.arange(s['start'], s['end']-20)
        if quota > len(frames):
            raise ValueError('Too many requested anchors in a short training segment')
        anchors.extend((s['episode_index'], int(f)) for f in sorted(rng.choice(frames, quota, replace=False)))
    return anchors


def numeric_scan(index, output):
    records, examples, counts = {}, [], {}
    root = Path(index['dataset_root'])
    for s in index['segments']:
        if s['split'] not in ('train', 'validation'):
            continue  # Reserve test for later final evaluation.
        ep = s['episode_index']
        table = pq.read_table(root/f'data/chunk-000/file-{ep:03d}.parquet', columns=[STATE,WRENCH,'action'])
        q, w, a = matrix(table[STATE],6), matrix(table[WRENCH],6), matrix(table['action'],7)
        records[ep] = dict(segment=s, state=q, wrench=w, action=a)
        n = s['end']-s['start']-40
        if n <= 0:
            continue
        first = np.arange(s['start'],s['start']+n)
        second = first+20  # Non-overlapping future chunks, one second apart.
        dq = np.max(np.abs(q[first]-q[second]),axis=1)
        df = np.linalg.norm(w[first,:3]-w[second,:3],axis=1)
        dt = np.linalg.norm(w[first,3:]-w[second,3:],axis=1)
        da = np.linalg.norm(a[first+1,:3]-a[second+1,:3],axis=1)
        near = (dq <= .005)&(df <= 1.)&(dt <= .1)
        divergent = near&(da >= .01)
        count = counts.setdefault(s['split'], dict(total_one_second_pairs=0,near_numeric_pairs=0,
                                                   near_numeric_next_target_diff_ge_10mm=0))
        count['total_one_second_pairs'] += n
        count['near_numeric_pairs'] += int(near.sum())
        count['near_numeric_next_target_diff_ge_10mm'] += int(divergent.sum())
        chosen = np.flatnonzero(divergent)
        if len(chosen):
            i = chosen[np.argmax(da[chosen])]
            examples.append(dict(episode=ep, source=s['source_episode_index'],split=s['split'],
                frame_a=int(first[i]),frame_b=int(second[i]),joint_max_rad=float(dq[i]),
                force_l2_n=float(df[i]),torque_l2_nm=float(dt[i]),next_target_diff_mm=float(da[i]*1000)))
    json_write(output/'numeric_scan.json',dict(counts=counts,examples=sorted(examples,key=lambda r:-r['next_target_diff_mm']),
        thresholds=dict(joint_max_rad=.005,force_l2_n=1.,torque_l2_nm=.1,separation_seconds=1.),
        meaning='Similar numeric inputs; images not yet checked; pairs overlap and are not independent.'))
    return records, examples


def freeze_model(checkpoint, device):
    # A single open descriptor survives training's atomic best.pt replacement.
    with checkpoint.open('rb') as source:
        sha = hashlib.file_digest(source,'sha256').hexdigest()
        source.seek(0)
        payload = torch.load(source,map_location='cpu',weights_only=True)
    if payload.get('format') != 'aic-sfp0-act-v2':
        raise ValueError('Audit requires the current single-task ACT checkpoint')
    model,_ = load_sfp_checkpoint(payload,device)
    return model,payload,sha


def collect(index, model, config, output, batch_size, train_samples, records, checkpoint_sha):
    path = output/'observations.npz'
    val = TaskDataset(index,'validation')
    train = TaskDataset(index,'train')
    lookup = {tuple(anchor):i for i,anchor in enumerate(train.anchors)}
    chosen = balanced_train_anchors(index,train_samples)
    data = ConcatDataset([val, Subset(train,[lookup[a] for a in chosen])])
    loader = DataLoader(data,batch_size=batch_size,num_workers=0,shuffle=False)
    values = {k:[] for k in ['episode','frame','state','wrench','target','prediction','embedding','rgb_small']}
    features = []
    def hook(_module,_inputs,result):
        pooled = result['feature_map'].detach().float().mean((-2,-1))
        features.append(torch.nn.functional.normalize(pooled,dim=-1).cpu().numpy())
    handle = model.policy.model.backbone.register_forward_hook(hook)
    model.eval()
    seen = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                features.clear()
                observations = move_batch({k:batch[k] for k in (*CAMERAS,STATE,WRENCH)},config.device)
                with autocast_context(config):
                    prediction = model.normalized_action_chunk(observations)
                physical = model.normalizer.unnormalize_actions(prediction).cpu().numpy()
                if len(features) != 3:
                    raise AssertionError('Expected one backbone invocation per camera')
                values['prediction'].append(physical)
                values['embedding'].append(np.stack(features,axis=1))
                for name,key in [('episode','episode_index'),('frame','frame_index'),('state',STATE),('wrench',WRENCH),('target','action')]:
                    values[name].append(batch[key].numpy())
                views=[]
                for key in CAMERAS:
                    images=(batch[key].permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
                    views.append(np.stack([cv2.resize(im,(48,48),interpolation=cv2.INTER_AREA) for im in images]))
                values['rgb_small'].append(np.stack(views,axis=1))
                seen += len(physical)
                if seen % 200 == 0 or seen == len(data):
                    print(json.dumps(dict(stage='predict',samples=seen,total=len(data))),flush=True)
    finally:
        handle.remove()
    values={k:np.concatenate(v) for k,v in values.items()}
    values['split'] = np.array(['validation']*len(val)+['train']*len(chosen))
    values['source'] = np.array([records[int(e)]['segment']['source_episode_index'] for e in values['episode']])
    values['progress'] = np.array([(int(f)-records[int(e)]['segment']['start']) /
        (records[int(e)]['segment']['end']-records[int(e)]['segment']['start']-1)
        for e,f in zip(values['episode'],values['frame'])])
    values['checkpoint_sha256']=np.array(checkpoint_sha)
    values['index_sha256']=np.array(index['index_content_sha256'])
    np.savez_compressed(path,**values)
    return values


def errors(values):
    prediction,target=values['prediction'].astype(np.float64),values['target'].astype(np.float64)
    difference=prediction[...,:3]-target[...,:3]
    position=np.linalg.norm(difference,axis=-1)*1000
    q=prediction[...,3:]
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    rotation=np.rad2deg(2*np.arccos(np.abs((q*target[...,3:]).sum(-1)).clip(0,1)))
    return position,rotation,difference*1000


def summarize(position,rotation,difference,mask):
    if not mask.any():return dict(samples=0)
    p=position[mask];r=rotation[mask];d=difference[mask]
    return dict(samples=int(mask.sum()),mean_position_mm=float(p.mean()),
        median_position_mm=float(np.median(p)),p90_position_mm=float(np.quantile(p,.9)),
        mean_rotation_deg=float(r.mean()),first_position_mm=float(p[:,0].mean()),
        last_position_mm=float(p[:,-1].mean()),xyz_mae_mm=np.abs(d).mean((0,1)).tolist(),
        xyz_bias_mm=d.mean((0,1)).tolist(),xyz_squared_error_share=(np.square(d).sum((0,1))/np.square(d).sum()).tolist())


def neighbor_pairs(values,mode,limits):
    """Select neighbors using OBSERVATIONS only, then measure label disagreement.

    Within-episode pairs are separated by >=1s. Across-episode pairs use distinct
    source demonstrations. No target threshold influences neighbor selection.
    """
    splits=values['split'];episodes=values['episode'];frames=values['frame']
    if mode=='train_cross_episode':queries=pool=np.flatnonzero(splits=='train')
    elif mode=='validation_to_train':queries=np.flatnonzero(splits=='validation');pool=np.flatnonzero(splits=='train')
    else:queries=pool=np.flatnonzero(splits=='validation')
    q,w,emb,rgb=values['state'],values['wrench'],values['embedding'],values['rgb_small']
    rows=[];eligible_queries=set()
    for i in queries:
        if mode=='validation_within_episode':valid=(episodes[pool]==episodes[i])&(frames[pool]>=frames[i]+20)
        else:valid=values['source'][pool]!=values['source'][i]
        if mode=='train_cross_episode':valid &= pool>i
        candidates=pool[valid]
        if not len(candidates):continue
        joint=np.max(np.abs(q[candidates]-q[i]),axis=1)
        force=np.linalg.norm(w[candidates,:3]-w[i,:3],axis=1)
        torque=np.linalg.norm(w[candidates,3:]-w[i,3:],axis=1)
        keep=(joint<=limits['joint_max_rad'])&(force<=limits['force_l2_n'])&(torque<=limits['torque_l2_nm'])
        candidates=candidates[keep];joint=joint[keep];force=force[keep];torque=torque[keep]
        if not len(candidates):continue
        cosine=(emb[candidates]*emb[i]).sum(-1).min(-1)
        pixel=np.abs(rgb[candidates].astype(np.float32)-rgb[i]).mean((2,3,4)).max(-1)
        for j,dq,df,dt,cos,mae in zip(candidates,joint,force,torque,cosine,pixel):
            if cos < limits['camera_cosine_min'] or mae > limits['camera_mae_max_255']:continue
            distance=np.linalg.norm(values['target'][i,:,:3]-values['target'][j,:,:3],axis=-1)*1000
            eligible_queries.add(int(i))
            if mode=='train_cross_episode':eligible_queries.add(int(j))
            rows.append(dict(a=int(i),b=int(j),episode_a=int(episodes[i]),episode_b=int(episodes[j]),
                frame_a=int(frames[i]),frame_b=int(frames[j]),source_a=int(values['source'][i]),source_b=int(values['source'][j]),
                joint_max_rad=float(dq),force_l2_n=float(df),torque_l2_nm=float(dt),
                camera_cosine_min=float(cos),camera_mae_max_255=float(mae),
                next_target_diff_mm=float(distance[0]),mean_chunk_target_diff_mm=float(distance.mean()),
                last_target_diff_mm=float(distance[-1])))
    differences=np.array([r['mean_chunk_target_diff_mm'] for r in rows])
    summary=dict(query_anchors=len(queries),queries_with_neighbors=len(eligible_queries),pairs=len(rows),
        unique_source_episodes=len(set([r['source_a'] for r in rows]+[r['source_b'] for r in rows])),
        median_chunk_disagreement_mm=float(np.median(differences)) if len(rows) else None,
        p90_chunk_disagreement_mm=float(np.quantile(differences,.9)) if len(rows) else None,
        pairs_chunk_disagreement_ge_10mm=int((differences>=10).sum()))
    return rows,summary


def original_images(root,episode,frame):
    table=pq.read_table(root/f'data/chunk-000/file-{episode:03d}.parquet',columns=list(CAMERAS))
    return [np.asarray(Image.open(io.BytesIO(table[key][frame].as_py()['bytes'])).convert('RGB')) for key in CAMERAS]


def witness_gallery(index,records,examples,output):
    """Check original RGB for top numeric witnesses, one candidate per source."""
    root=Path(index['dataset_root']);checked=[]
    for row in sorted(examples,key=lambda r:-r['next_target_diff_mm'])[:30]:
        first=original_images(root,row['episode'],row['frame_a'])
        second=original_images(root,row['episode'],row['frame_b'])
        mae=[float(np.abs(a.astype(float)-b).astype(float).mean()) for a,b in zip(first,second)]
        candidate=dict(row,camera_fullres_mae_255=mae,all_cameras_mae_le_3=bool(max(mae)<=3))
        checked.append(candidate)
        if candidate['all_cameras_mae_le_3'] and sum(r.get('gallery') is not None for r in checked)<6:
            record=records[row['episode']];qa=record['action'][row['frame_a']+1];qb=record['action'][row['frame_b']+1]
            canvas=Image.new('RGB',(864,610),'white');draw=ImageDraw.Draw(canvas)
            for k,images in enumerate((first,second)):
                for camera,image in enumerate(images):canvas.paste(Image.fromarray(image),(camera*288,45+k*280))
                frame=row['frame_a'] if k==0 else row['frame_b'];target=qa if k==0 else qb
                draw.text((5,28+k*280),f'frame {frame}; next command xyz (m): '+', '.join(f'{v:.4f}' for v in target[:3]),fill='black')
            draw.text((5,5),f'source {row["source"]}, {row["split"]}; 1s apart; next target difference {row["next_target_diff_mm"]:.1f} mm',fill='black')
            draw.text((5,588),f'joint max={row["joint_max_rad"]:.5f} rad; force diff={row["force_l2_n"]:.3f}N; torque diff={row["torque_l2_nm"]:.3f}Nm',fill='black')
            path=output/f'witness_source_{row["source"]:03d}.jpg';canvas.save(path,quality=94)
            candidate['gallery']=path.name
    return checked


def plot(values,position,rotation,difference,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val=values['split']=='validation';train=~val
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for mask,label in [(train,'Train sample'),(val,'Fixed validation')]:
        means=[position[mask&(values['progress']>=lo)&(values['progress']<hi)].mean() for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]]
        axes[0,0].plot(range(4),means,'o-',label=label)
    axes[0,0].set(xticks=range(4),xticklabels=['0-25%','25-50%','50-75%','75-100%'],ylabel='Mean position error (mm)',xlabel='Relative time within retained clip (not task phase)')
    axes[0,0].legend()
    for mask,label in [(train,'Train sample'),(val,'Fixed validation')]:
        p=np.sort(position[mask].mean(-1));axes[0,1].plot(p,np.arange(1,len(p)+1)/len(p),label=label)
    axes[0,1].set(xlabel='Mean error per 20-step chunk (mm)',ylabel='Cumulative fraction');axes[0,1].legend()
    for mask,label in [(train,'Train sample'),(val,'Fixed validation')]:
        axes[1,0].plot(np.arange(1,21)*.05,position[mask].mean(0),label=label)
    axes[1,0].set(xlabel='Prediction horizon (seconds)',ylabel='Mean position error (mm)');axes[1,0].legend()
    mae=np.abs(difference[val]).mean((0,1));axes[1,1].bar(['X','Y','Z'],mae)
    axes[1,1].set(ylabel='Validation absolute component error (mm)')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'error_breakdown.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('/mnt/d/robot_models/aic_sfp0_act_v2_fresh/checkpoints/best.pt'))
    parser.add_argument('--output-dir',type=Path,default=Path('/mnt/d/robot_models/aic_sfp0_act_v2_fresh/audits/observation_ambiguity_v1'))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--train-samples',type=int,default=1200)
    parser.add_argument('--reuse-cache',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model,payload,sha=freeze_model(args.checkpoint,args.device)
    index=read_index(payload['config']['index_path'])
    if index['index_content_sha256']!=payload['data_index_sha256']:
        raise ValueError('Checkpoint and data index differ')
    started=time.perf_counter()
    records,examples=numeric_scan(index,args.output_dir)
    if args.reuse_cache:
        with np.load(args.output_dir/'observations.npz',allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
        if values['checkpoint_sha256'].item()!=sha or values['index_sha256'].item()!=index['index_content_sha256']:
            raise ValueError('Cached observations refer to different weights or data')
    else:
        values=collect(index,model,model_config(payload,args.device),
                       args.output_dir,args.batch_size,args.train_samples,records,sha)
    # Release GPU before the purely CPU neighbor/image analysis.
    del model
    if args.device=='cuda':torch.cuda.empty_cache()
    position,rotation,difference=errors(values)
    report=dict(checkpoint=str(args.checkpoint),checkpoint_step=payload['step'],checkpoint_sha256=sha,
        data_index_sha256=index['index_content_sha256'],phase_labels_available=False,
        physical_tcp_observation_available=False,test_split_used=False,
        limitations=['Relative clip time is not a physical contact/insertion label.',
                    'Raw force magnitude groups are not calibrated contact labels.',
                    'Neighbor similarity is approximate; small task-relevant visual differences can remain.',
                    'Within-episode pairs do not establish multiple strategies: elapsed time/previous command may be hidden.',
                    'Training anchors are a balanced sample, not an exhaustive neighborhood search.',
                    'Repeated/overlapping pairs are correlated; counts are not independent trials.',
                    'No automatic conclusion that a CVAE would improve closed-loop success.'])
    for split in ['validation','train']:
        mask=values['split']==split
        report[split]=summarize(position,rotation,difference,mask)
        report[split]['relative_time_quartiles']=[dict(interval=f'{lo:.2f}-{hi:.2f}',
            **summarize(position,rotation,difference,mask&(values['progress']>=lo)&(values['progress']<hi)))
            for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]]
        force=np.linalg.norm(values['wrench'][:,:3],axis=-1)
        edges=np.quantile(force[values['split']=='train'],[0,.25,.5,.75,1]);edges[-1]=float('inf')
        report[split]['force_magnitude_groups']=[dict(force_norm_interval_n=[float(lo),float(hi) if np.isfinite(hi) else None],
            **summarize(position,rotation,difference,mask&(force>=lo)&(force<hi))) for lo,hi in zip(edges[:-1],edges[1:])]
    expected=payload['metrics']['position_l2_m']*1000
    report['validation_reproduction']=dict(recorded_position_mm=expected,
        audited_position_mm=report['validation']['mean_position_mm'],
        absolute_difference_mm=abs(expected-report['validation']['mean_position_mm']))
    if report['validation']['samples']!=1200 or report['validation_reproduction']['absolute_difference_mm']>.2:
        raise AssertionError('Audit did not reproduce the fixed validation result within 0.2mm')
    report['neighbors']={}
    for level,limits in [('strict',dict(joint_max_rad=.005,force_l2_n=1.,torque_l2_nm=.1,camera_cosine_min=.995,camera_mae_max_255=3.)),
                         ('moderate',dict(joint_max_rad=.01,force_l2_n=2.,torque_l2_nm=.2,camera_cosine_min=.99,camera_mae_max_255=5.))]:
        report['neighbors'][level]=dict(thresholds=limits)
        for mode in ['train_cross_episode','validation_to_train','validation_within_episode']:
            rows,summary=neighbor_pairs(values,mode,limits)
            report['neighbors'][level][mode]=summary
            if rows:
                with (args.output_dir/f'neighbors_{level}_{mode}.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                report['neighbors'][level][mode]['largest_disagreements']=sorted(rows,key=lambda r:-r['mean_chunk_target_diff_mm'])[:5]
            print(json.dumps(dict(stage='neighbors',level=level,mode=mode,**summary)),flush=True)
    report['numeric_witness_rgb_review']=witness_gallery(index,records,examples,args.output_dir)
    plot(values,position,rotation,difference,args.output_dir)
    with (args.output_dir/'per_anchor.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['split','source','episode','frame','relative_clip_time','position_mean_mm','position_first_mm','position_last_mm','rotation_mean_deg','x_mae_mm','y_mae_mm','z_mae_mm'])
        for i in range(len(position)):
            writer.writerow([values['split'][i],values['source'][i],values['episode'][i],values['frame'][i],values['progress'][i],position[i].mean(),position[i,0],position[i,-1],rotation[i].mean(),*np.abs(difference[i]).mean(0)])
    report['elapsed_seconds']=time.perf_counter()-started
    json_write(args.output_dir/'report.json',report)
    print(json.dumps(dict(report=str(args.output_dir/'report.json'),validation=report['validation'],train=report['train']),indent=2),flush=True)


def model_config(payload,device):
    from ftact_v2 import SFPConfig
    config=SFPConfig(**payload['config']);config.device=device
    if device=='cpu':config.amp='none'
    return config


if __name__=='__main__':main()
