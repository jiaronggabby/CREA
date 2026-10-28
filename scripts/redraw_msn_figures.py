"""Render the five CREA MSN figures at their final physical sizes.

Author: Jia Rong. Quantitative figures reuse retained numerical results.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/figures'
DATA = ROOT/'results/final_v1_current/analysis'
BLUE, TEAL, CORAL, INK = '#4C78A8', '#5AA6A6', '#D98373', '#253341'
DIFF = LinearSegmentedColormap.from_list('difference',[BLUE,'#F7F7F5',CORAL])


def style():
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',
        'mathtext.bf':'Arial:bold','mathtext.cal':'Arial:italic',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'savefig.facecolor':'white'})


def save(fig, name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/name,bbox_inches=None,dpi=600)
    plt.close(fig)


def phase():
    evidence = json.loads((DATA/'traffic_five_seed_evidence.json').read_text(encoding='utf-8'))['per_cell']
    fig = plt.figure(figsize=(3.5,2.3))
    ax = fig.add_axes([.17,.24,.60,.54])
    names = ['pems03','pems04','pems07','pems08','metr_la']
    labels = ['PEMS03','PEMS04','PEMS07','PEMS08','METR-LA']
    markers = ['o','s','^','D','P']
    values = [100*(1-r['mae']['CREA']/r['mae']['L']) for r in evidence]
    limit = max(abs(np.array(values)))
    norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    handles = []
    for dataset,label,marker in zip(names,labels,markers):
        rows = [r for r in evidence if r['dataset']==dataset]
        x = [100*(1-r['mae']['R']/r['mae']['A']) for r in rows]
        y = [100*(1-r['mae']['Oracle']/r['mae']['S']) for r in rows]
        c = [100*(1-r['mae']['CREA']/r['mae']['L']) for r in rows]
        ax.scatter(x,y,c=c,cmap=DIFF,norm=norm,marker=marker,s=23,
                   edgecolor=INK,linewidth=.3,zorder=3)
        handles.append(plt.Line2D([],[],linestyle='',marker=marker,markerfacecolor='white',
                        markeredgecolor=INK,markersize=4,label=label))
    ax.axvline(0,color=INK,linewidth=.6,linestyle='--')
    ax.grid(color='#E0E5EB',linewidth=.4)
    ax.set_xlabel('Relational gain over A (%)',labelpad=2)
    ax.set_ylabel('Oracle gain over S (%)',labelpad=2)
    ax.tick_params(length=2,pad=2)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,
               frameon=False,columnspacing=.7,handletextpad=.25,labelspacing=.3)
    cax = fig.add_axes([.81,.24,.025,.54])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DIFF),cax=cax)
    cb.set_ticks([-1,0,1] if limit>=1 else [-.5,0,.5])
    cb.ax.tick_params(length=2,pad=2)
    cb.set_label('CREA vs. L (%)',labelpad=4)
    save(fig,'fig_relational_phase_diagram.pdf')


def trust():
    data = np.load(DATA/'msn_plot_data.npz')
    fig = plt.figure(figsize=(3.5,1.65))
    axes = [fig.add_axes([.145+i*.28,.35,.245,.57]) for i in range(3)]
    for i,(h,ax) in enumerate(zip((3,6,12),axes)):
        counts = data[f'density_{h}'].T
        ax.pcolormesh(data[f'xedges_{h}'],data[f'yedges_{h}'],np.ma.masked_equal(counts,0),
                      cmap='Blues',norm=LogNorm(1,max(counts.max(),2)),alpha=.34,rasterized=True)
        x,mean,sd = (data[f'{key}_{h}'] for key in ('centers','mean','sd'))
        ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=CORAL,alpha=.24,linewidth=0)
        ax.plot(x,mean,color=CORAL,linewidth=1.1)
        ax.set_xlim(data[f'xedges_{h}'][[0,-1]])
        ax.set_ylim(0,1)
        ax.set_yticks([0,.5,1],['0','0.5','1'])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(length=2,pad=2)
        if i:
            ax.tick_params(labelleft=False)
        ax.text(.5,-.28,f'{chr(97+i)}   h = {h}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    axes[0].set_ylabel('Direct R weight',labelpad=2)
    fig.text(.55,.015,'Past A−R error advantage',ha='center',fontsize=9)
    save(fig,'fig_trust_utility_alignment.pdf')


def atlas():
    data = np.load(DATA/'msn_effective_reliance_atlas.npz')
    fig = plt.figure(figsize=(3.5,1.75))
    axes = [fig.add_axes([.15+i*.277,.35,.23,.39]) for i in range(3)]
    for i,(ax,values) in enumerate(zip(axes,data['panels'])):
        limit = max(float(np.percentile(np.abs(values),98)),1e-6)
        step = 10**np.floor(np.log10(limit))
        limit = float(np.ceil(limit/step)*step)
        norm = Normalize(0,1) if i==1 else TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        im = ax.imshow(values.T,origin='lower',aspect='auto',cmap='GnBu' if i==1 else DIFF,
                       norm=norm,interpolation='nearest',extent=(0,100,0,int(data['nodes'])-1),rasterized=True)
        ax.set_xticks([0,50,100])
        ax.get_xticklabels()[0].set_horizontalalignment('left')
        ax.get_xticklabels()[-1].set_horizontalalignment('right')
        ax.set_yticks([0,441,882])
        ax.tick_params(length=2,pad=2,labelleft=(i==0))
        ax.text(.5,-.37,chr(97+i),transform=ax.transAxes,ha='center',va='top',fontweight='bold',fontsize=9)
        cax = fig.add_axes([ax.get_position().x0,.85,ax.get_position().width,.027])
        cb = fig.colorbar(im,cax=cax,orientation='horizontal',extend='neither' if i==1 else 'both')
        cb.set_ticks([0,1] if i==1 else [-limit,limit])
        cb.ax.tick_params(length=2,pad=1)
        cb.set_label(('A−R error',r'$\alpha^{\mathrm{eff}}$','L−CREA error')[i],labelpad=3)
        cb.ax.xaxis.set_label_position('top')
    axes[0].set_ylabel('Sensor index',labelpad=2)
    fig.text(.55,.015,'Test-stream progress (%)',ha='center',fontsize=9)
    save(fig,'fig_effective_relational_reliance.pdf')


def shift():
    rows = json.loads((DATA/'prior_reliability_shift/final_v1_prior_reliability_shift.json').read_text(encoding='utf-8'))['rows']
    fig = plt.figure(figsize=(3.5,1.85))
    axes = [fig.add_axes([.17,.30,.31,.47]),fig.add_axes([.65,.30,.31,.47])]
    for dataset,label,color,marker in [('pems04','PEMS04',BLUE,'o'),('pems07','PEMS07',TEAL,'D'),('metr_la','METR-LA',CORAL,'s')]:
        groups = [[r for r in rows if r['dataset']==dataset and r['requested_endpoint_change_fraction']==level] for level in (0,.25,.5,.75,1)]
        assert all(len(g)==5 for g in groups)
        x = [100*np.mean([r['actual_endpoint_change_fraction'] for r in g]) for g in groups]
        for ax,key in zip(axes,['prior_expert_weight_drop','prior_harm_rate_second_half']):
            v = [100*np.array([r[key] for r in g]) for g in groups]
            ax.errorbar(x,[a.mean() for a in v],yerr=[a.std(ddof=1) for a in v],label=label,
                        color=color,marker=marker,markersize=3,linewidth=.8,capsize=1.5,elinewidth=.55)
    axes[0].axhline(0,color='#8792A0',linewidth=.6)
    axes[0].set_ylabel('R-weight drop (pp)',labelpad=2)
    axes[1].set_ylabel('Origins with larger\nR MAE (%)',labelpad=2)
    for i,ax in enumerate(axes):
        ax.set_xticks([0,50,100])
        ax.tick_params(length=2,pad=2)
        ax.text(.5,-.25,chr(97+i),transform=ax.transAxes,ha='center',va='top',fontweight='bold',fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
               bbox_to_anchor=(.53,1.02),columnspacing=.6,handlelength=1,handletextpad=.3)
    fig.text(.55,.015,'Realized endpoint changes (%)',ha='center',fontsize=9)
    save(fig,'fig_prior_shift.pdf')


def method():
    fig = plt.figure(figsize=(7,2.65))
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,7); ax.set_ylim(0,2.65); ax.axis('off')
    def box(x,y,w,h,text='',color='#F3F5F6',edge='#AAB6C0',size=9,bold=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.045',facecolor=color,edgecolor=edge,linewidth=.7))
        if text: ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',linespacing=1.25)
    def arrow(x0,y0,x1,y1,color=INK,style='-'):
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=.85,color=color,linestyle=style,shrinkA=0,shrinkB=0))
    panels = [(0.04,.78,'Issue-time\ninputs','#F0F3F3'),(.94,1.30,'Frozen structural\nexperts','#EAF3FC'),(2.36,1.65,'Allocation\npolicies','#FFF1E5'),(4.13,1.75,'Policy trust\naggregation','#F0F3F3'),(6.0,.95,'H-step\nforecast','#F0F3F3')]
    for x,w,title,color in panels:
        box(x,.76,w,1.79,color=color)
        ax.text(x+w/2,2.43,title,ha='center',va='top',fontsize=10,fontweight='bold',linespacing=1.1)
    box(.12,1.57,.62,.37,r'$X_u$'+'\nHistory',size=9)
    box(.12,.95,.62,.37,r'$G$'+'\nRelation',size=9)
    box(1.04,1.51,1.10,.49,'A  Unrestricted\n'+r'$p^{(A)}=f_A(X_u)$',color='white')
    box(1.04,.91,1.10,.49,'R  Constrained\n'+r'$p^{(R)}=f_R(X_u;G)$',color='white')
    arrow(.78,1.76,1.0,1.76); arrow(.80,1.61,.90,1.61); ax.plot([.90,.90],[1.61,1.2],color=INK,lw=.85); arrow(.90,1.2,1.0,1.2)
    arrow(.78,1.05,1.0,1.05)
    for y,text in [(1.98,'A, R  Direct forecasts'),(1.60,'S  Static Mix'),(1.22,'L  Loss Gate')]:
        box(2.46,y-.14,1.45,.28,text,color='white')
    ax.text(3.18,.86,'History + forecast differences',ha='center',fontsize=8)
    arrow(2.16,1.79,2.42,1.79); arrow(2.16,1.19,2.42,1.19)
    ax.text(5.005,1.90,r'$W_u=\{w^{(k)}_{u,h,n}\}$',ha='center',fontsize=10)
    ax.text(5.005,1.57,r'$\hat y_{u,h,n}=\sum_k w^{(k)}_{u,h,n}p^{(k)}_{u,h,n}$',ha='center',fontsize=9)
    ax.text(5.005,1.13,'Effective relational reliance',ha='center',fontsize=9)
    ax.text(5.005,.88,r'$\alpha^{\mathrm{eff}}=w^{(R)}+\alpha_h^* w^{(S)}+\widetilde q\,w^{(L)}$',ha='center',fontsize=8.5)
    arrow(4.04,1.70,4.14,1.70); arrow(5.91,1.70,6.00,1.70)
    ax.text(6.475,1.72,r'$\hat Y_u$',ha='center',fontsize=14)
    ax.text(6.475,1.35,'h = 3, 6, 12',ha='center',fontsize=9)
    box(.04,.06,6.91,.47,color='#F2EBFA',edge='#B49BCF')
    for x,text in [(.12,'Forecast issued\nat t'),(1.48,'Target observed\nat t + h'),(3.12,'Eligible losses\n'+r'$t+h\leq u$'),(4.87,'Revise trust\nbefore forecast u')]:
        ax.text(x+.55,.30,text,ha='center',va='center',fontsize=9)
    for x0,x1 in [(1.20,1.40),(2.66,2.97),(4.24,4.70)]: arrow(x0,.30,x1,.30)
    arrow(5.73,.56,5.73,.76,color='#B54457',style='--')
    ax.text(4.63,.63,'Matured outcomes update future trust',ha='center',fontsize=8,color='#A83A50')
    save(fig,'fig_method_msn.pdf')


if __name__=='__main__':
    style()
    method(); phase(); trust(); atlas(); shift()
