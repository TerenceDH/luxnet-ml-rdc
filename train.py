"""Train the archived architecture with explicit, disjoint split manifests."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from luxnet import training as t

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/full'))
    p.add_argument('--splits', type=Path, default=Path('data/splits.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/train'))
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    split = json.loads(a.splits.read_text(encoding='utf-8'))
    for key in ['train', 'validation', 'test']:
        if not split[key] or len(split[key]) != len(set(split[key])):
            p.error('Split lists must be nonempty and contain unique IDs')
    if any(set(split[x]) & set(split[y]) for x,y in [('train','validation'),('train','test'),('validation','test')]):
        p.error('Split overlap detected')
    if len(split['train']) < a.batch_size:
        p.error('Training set must contain at least one full batch')
    t.DATA_ROOT = str(a.data.resolve() / 'train')
    t.A_DIR = str(Path(t.DATA_ROOT)/'A_float')
    t.B_DIR = str(Path(t.DATA_ROOT)/'B_float')
    t.COND_DIR = str(Path(t.DATA_ROOT)/'cond_vec')
    for key in ['train','validation']:
        for f in split[key]:
            for d in [t.A_DIR,t.B_DIR,t.COND_DIR]:
                if not (Path(d)/f).is_file():
                    p.error(f'Missing {Path(d)/f}; extract the dataset first')
    a.output.mkdir(parents=True, exist_ok=False)
    t.RUN_ROOT = t.DATA_SAVE = str(a.output.resolve())
    for name, folder in [('CHECKPOINT_DIR','checkpoints'),('SAMPLE_DIR','samples'),('LOG_DIR','logs')]:
        path = a.output.resolve()/folder; path.mkdir(); setattr(t,name,str(path))
    t.HPARAM_JSON = str(a.output.resolve()/'hparams.json')
    t.NUM_EPOCHS, t.BATCH_SIZE, t.device = a.epochs, a.batch_size, torch.device(a.device)
    t.set_seed(t.SEED)
    mean,std=t.compute_cond_stats(split['train'])
    def loaders():
        args=(t.A_DIR,t.B_DIR,t.COND_DIR)
        return (
            DataLoader(t.LuxDualInputDataset(*args,split['train'],mean,std),batch_size=a.batch_size,shuffle=True,drop_last=True),
            DataLoader(t.LuxDualInputDataset(*args,split['validation'],mean,std),batch_size=a.batch_size,shuffle=False),
            mean,std)
    t.build_loader=loaders
    original_collect=t.collect_hparams
    def collect(cond_dim):
        h=original_collect(cond_dim)
        h['dataset'].update(total_files=len(split['train'])+len(split['validation']),train_files=len(split['train']),val_files=len(split['validation']))
        h['dataset']['split_manifest']=str(a.splits)
        return h
    t.collect_hparams=collect
    (a.output/'splits.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    t.train()

if __name__ == '__main__':
    main()
