"""Run archived Case 01 or Case 07 from geometry through inference and PSO."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from luxnet.scene import parse_combine_txt, infer_lamp_arrays_at_points
from luxnet.optimization import pso_optimize_lighting_fast

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['case01','case07'],default='case01')
    p.add_argument('--personalized',action='store_true',help='800/300 lux at upper/lower right four sensor points; 500 elsewhere')
    p.add_argument('--checkpoint',type=Path,default=Path('weights/mresunet_epoch888.pt'))
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    root=Path(__file__).resolve().parent/'examples'/a.case
    out=a.output or Path('outputs')/(a.case+('_personalized' if a.personalized else ''))
    scene=parse_combine_txt(root/'combine.txt')
    pts=np.load(root/'pts_xyz_valid.npy')
    keys=['room_vertices_mm','room_dims_mm','lamp_positions_mm','wall_rhos','surface_rhos','window_rects_mm','win_h_mm','sill_h_mm','window_rho']
    lamp,maps,mask,x,y,scale=infer_lamp_arrays_at_points(**{k:scene[k] for k in keys},pts_xyz_mm=pts,g_ckpt_path=a.checkpoint,train_global_mm2px=float(np.load(root/'s_global_mm2px.npy')[0]))
    if not mask.all(): raise ValueError('Archived sensor order changed; refusing misaligned daylight')
    daylight=np.load(root/'daylight_lux.npy')
    target=np.full(len(pts),500,dtype=np.float32)
    if a.personalized:
        upper=np.argsort((pts[:,0]-pts[:,0].max())**2+(pts[:,1]-pts[:,1].max())**2)[:4]
        lower=[i for i in np.argsort((pts[:,0]-pts[:,0].max())**2+(pts[:,1]-pts[:,1].min())**2) if i not in upper][:4]
        target[upper]=800;target[lower]=300
    result=pso_optimize_lighting_fast(daylight,lamp,target)
    baseline=daylight+0.5*lamp.sum(axis=0)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'fields.npz',lamp_maps=maps,lamp_arrays=lamp,daylight=daylight,target=target,optimized_total=result['best_total'],ratios=result['best_ratios'])
    with (out/'sensor_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['x_mm','y_mm','z_mm','daylight_lux','target_lux','half_power_total_lux','optimized_total_lux'])
        w.writerows(np.column_stack([pts,daylight,target,baseline,result['best_total']]))
    summary={'case':a.case,'personalized':a.personalized,'time':'January 1 09:00, archived shaded daylight','objective':'MSE * (1 + mean dimming ratio)','ratios':result['best_ratios'].tolist(),'objective_value':result['best_objective'],'half_power_mae_lux':float(np.mean(abs(baseline-target))),'optimized_mae_lux':float(np.mean(abs(result['best_total']-target))),'power_w':float(33*result['best_ratios'].sum()),'reference_lamp_max_abs_diff_lux':float(np.max(abs(lamp-np.load(root/'reference_lamp_arrays.npy')))),'note':'Demonstration from archived inputs; not a rerun of all annual manuscript results.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    vmax=max(800,float(baseline.max()),float(result['best_total'].max()))
    for axis,values,title in zip(ax,[daylight,baseline,result['best_total']],['Daylight','Half-power baseline','Optimized lighting']):
        im=axis.scatter(pts[:,0]/1000,pts[:,1]/1000,c=values,cmap='viridis',vmin=0,vmax=vmax,s=200)
        axis.set(title=title,xlabel='x (m)',ylabel='y (m)',aspect='equal')
    fig.colorbar(im,ax=ax,label='Illuminance (lux)',shrink=.8)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
