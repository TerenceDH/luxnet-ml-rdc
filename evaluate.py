"""Evaluate the checkpoint on a selected manifest split; no training occurs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from luxnet.model import load_model
from luxnet.metrics import gaussian_kernel_2d, compute_batch_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/sample'))
    p.add_argument('--splits',type=Path,default=Path('data/sample_splits.json'))
    p.add_argument('--split',choices=['train','validation','test'],default='test')
    p.add_argument('--checkpoint',type=Path,default=Path('weights/mresunet_epoch888.pt'))
    p.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.batch_size<1: p.error('batch-size must be positive')
    ids=json.loads(a.splits.read_text(encoding='utf-8'))[a.split]
    root=a.data/('test' if a.split=='test' else 'train')
    if not ids: p.error('Empty split')
    model,mean,std=load_model(a.checkpoint,a.device)
    a.output.mkdir(parents=True,exist_ok=False)
    kernel=gaussian_kernel_2d(11,1.5,torch.device(a.device))
    rows=[]
    for start in range(0,len(ids),a.batch_size):
        batch=ids[start:start+a.batch_size]
        x=np.stack([np.load(root/'A_float'/f) for f in batch]).astype('float32')
        c=np.stack([np.load(root/'cond_vec'/f).ravel() for f in batch]).astype('float32')
        gt=np.stack([np.load(root/'B_float'/f).squeeze() for f in batch]).astype('float32')
        with torch.inference_mode():
            pred=model(torch.from_numpy(x.clip(0,1)).to(a.device),torch.from_numpy((c-mean)/std).to(a.device))[:,0].cpu().numpy()
        count,mae,mse,ssim=compute_batch_metrics(pred,gt,torch.device(a.device),kernel)
        for j,f in enumerate(batch):
            rows.append(dict(file=f,num_core_pixels=int(count[j]),mae=float(mae[j]),mse=float(mse[j]),ssim=float(ssim[j])))
        if start==0:
            np.savez_compressed(a.output/'first_prediction.npz',prediction=pred[0],ground_truth=gt[0],input=x[0])
        print(f'{min(start+len(batch),len(ids))}/{len(ids)}',flush=True)
    with (a.output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'split':a.split,'count':len(rows),'mae_lux':float(np.mean([r['mae'] for r in rows])),'ssim':float(np.mean([r['ssim'] for r in rows])),'ssim_scale_lux':603.86}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
