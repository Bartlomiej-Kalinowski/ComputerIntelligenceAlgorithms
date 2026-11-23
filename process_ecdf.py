import json
import math
from pathlib import Path

import numpy as np
import openpyxl
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'outputs' / 'ecdf_ae_20261118'
OUT.mkdir(parents=True, exist_ok=True)
SRC = BASE / 'upload' / 'Wyniki_AlgorytmEwolucyjny.xlsx'
N = 10
EVALS = 10000
budgets = np.rint(N * 10 ** (np.arange(41) * (math.log10(EVALS/N) / 40))).astype(int)
thresholds = 10 ** (-8 + np.arange(51) * .2)

def f1(s):
    return -5 / (1 + s) + math.sin(1 / math.tan(math.exp(-5 / (1 + s))))

grid = np.linspace(0, 90, 900001)
idx = np.argmin([f1(float(s)) for s in grid])
opt = minimize_scalar(f1, bounds=(max(0, grid[idx] - .001), grid[idx] + .001), method='bounded', options={'xatol':1e-14})
fstar = {'F1': float(min(opt.fun, f1(0))), 'F2': 0.0}

book = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
data = {}
raw_sample = {}
for fn in ('F1', 'F2'):
    for rep in ('gray', 'real'):
        ws = book[f'{fn}_{rep}']
        sampled = np.empty((41, 100), dtype=float)
        last = 0
        for index, row in enumerate(ws.iter_rows(min_row=2, max_row=EVALS+1, values_only=True), 1):
            while last < 41 and index == int(budgets[last]):
                sampled[last, :] = row[1:101]
                last += 1
            if last == 41:
                break
        if last != 41:
            raise RuntimeError(f'Missing budget rows: {fn}_{rep}: {last}')
        if np.any(np.diff(sampled, axis=0) > 1e-9):
            raise ValueError(f'History not monotonic: {fn}_{rep}')
        diff = np.abs(sampled - fstar[fn])
        data[(fn, rep)] = np.array([(diff < threshold).sum(axis=1) for threshold in thresholds], dtype=int).T
        raw_sample[f'{fn}_{rep}'] = sampled.tolist()

result = {'budget':budgets.tolist(), 'thresholds':thresholds.tolist(), 'optima':fstar,
          'sampled':raw_sample, 'ecdf':{}, 'final':{}}
for rep in ('gray', 'real'):
    pooled = data[('F1',rep)] + data[('F2',rep)]
    assert pooled.shape == (41,51) and np.all(np.diff(pooled,axis=0) >= 0)
    result['ecdf'][rep] = {'combined':pooled.tolist(),'F1':data[('F1',rep)].tolist(),'F2':data[('F2',rep)].tolist()}
    result['final'][rep] = {'F1':raw_sample[f'F1_{rep}'][-1], 'F2':raw_sample[f'F2_{rep}'][-1]}

(BASE/'ecdf_data.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')

plt.rcParams.update({'font.size': 10,'axes.titlesize': 13,'figure.facecolor':'white'})
selected = [-4,-3,-2,-1,0,0.6,1,1.4,2]
indices = [int(round((v+8)/.2)) for v in selected]
for rep in ('gray','real'):
    fig,ax=plt.subplots(figsize=(10.4,6))
    colors=plt.cm.viridis(np.linspace(.12,.95,len(indices)))
    for col,exponent,color in zip(indices,selected,colors):
        ax.step(budgets, np.array(result['ecdf'][rep]['combined'])[:,col]/200,
                where='post',label=rf'$10^{{{exponent:g}}}$',color=color,linewidth=1.9)
    ax.set_xscale('log');ax.set_ylim(-.015,1.035);ax.set_xlim(10,10000)
    ax.set_xlabel('Liczba wywołań funkcji celu F (budżet)')
    ax.set_ylabel('Udział udanych eksperymentów (z 200)')
    ax.set_title(f'ECDF — {"reprezentacja binarna Gray" if rep=="gray" else "reprezentacja rzeczywistoliczbowa"}\nF1 + F2, 100 uruchomień każdej funkcji')
    ax.grid(alpha=.25,which='both');ax.legend(title='Próg jakości ΔF <',ncol=3,loc='lower right')
    fig.tight_layout();fig.savefig(OUT/f'ECDF_{rep}.png',dpi=190);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
for ax,fn in zip(axs,('F1','F2')):
    for rep,color in [('gray','#1d4ed8'),('real','#e76f51')]:
        sampled=np.array(raw_sample[f'{fn}_{rep}'])
        errors=np.maximum(np.abs(sampled-fstar[fn]),1e-12)
        ax.plot(budgets, np.median(errors,axis=1),marker='.',markersize=3,
                color=color,label='Gray' if rep=='gray' else 'Real')
    ax.set(xscale='log',yscale='log',xlabel='Liczba wywołań F',title=f'{fn} — mediana |F_best − F*|')
    ax.grid(alpha=.25,which='both');ax.legend()
axs[0].set_ylabel('Błąd względem optimum')
fig.tight_layout();fig.savefig(OUT/'Postep_optymalizacji.png',dpi=190);plt.close(fig)
print(json.dumps({'optima':fstar,'budgets':budgets.tolist(),'final_summary':{rep:{fn:{'median_error':float(np.median(np.abs(np.array(result['final'][rep][fn])-fstar[fn])))} for fn in ('F1','F2')} for rep in ('gray','real')}},ensure_ascii=False))
