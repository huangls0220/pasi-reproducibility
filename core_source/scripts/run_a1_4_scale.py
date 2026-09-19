"""A1.4 Scale trend analysis with paired tests and trend model."""
import pandas as pd, numpy as np, math, json
from scipy import stats

df = pd.read_csv('results/route_a/a1_3/scale_audit/scale_audit_all_runs.csv')
outdir = 'results/route_a/a1_4/scale_trend'
import os; os.makedirs(outdir, exist_ok=True)

scales = [(25,20),(50,40),(100,80),(200,160)]
seeds = sorted(df['seed'].unique())

seed_data = []
for N,M in scales:
    sub = df[(df['N']==N)&(df['M']==M)]
    moi = sub[sub['method']=='MOI'].set_index('seed')['cumulative_payment'].astype(float)
    pasi = sub[sub['method']=='PASI'].set_index('seed')['cumulative_payment'].astype(float)
    for s in seeds:
        if s in moi.index and s in pasi.index:
            rel = (moi[s]-pasi[s])/moi[s]*100
            seed_data.append({'N':N,'M':M,'scale':f'{N}/{M}','seed':s,'rel':rel})

sdf = pd.DataFrame(seed_data)
sdf.to_csv(f'{outdir}/scale_seed_level.csv', index=False)

# Endpoint paired test 25/20 vs 200/160
r25 = sdf[sdf['scale']=='25/20'].set_index('seed')['rel']
r200 = sdf[sdf['scale']=='200/160'].set_index('seed')['rel']
common = sorted(set(r25.index)&set(r200.index))
deltas = [r200[s]-r25[s] for s in common]
mean_d = np.mean(deltas); se_d = np.std(deltas,ddof=1)/math.sqrt(len(deltas))
t_stat, p_val = stats.ttest_rel([r200[s] for s in common],[r25[s] for s in common])
w_stat, w_p = stats.wilcoxon([r200[s] for s in common],[r25[s] for s in common])
cohen_dz = mean_d/np.std(deltas,ddof=1)

print('Endpoint paired test (200/160 vs 25/20):')
print(f'  Mean delta: {mean_d:.4f} pp (SE={se_d:.4f})')
print(f'  95% CI: [{mean_d-1.96*se_d:.4f}, {mean_d+1.96*se_d:.4f}]')
print(f'  Paired t: t={t_stat:.3f}, p={p_val:.4f}')
print(f'  Wilcoxon: p={w_p:.4f}')
print(f'  Cohen dz: {cohen_dz:.3f}')

pd.DataFrame([{'mean_delta':mean_d,'se':se_d,'ci_lo':mean_d-1.96*se_d,'ci_hi':mean_d+1.96*se_d,
    't_stat':t_stat,'p_value':p_val,'wilcoxon_p':w_p,'cohen_dz':cohen_dz}]).to_csv(
    f'{outdir}/scale_endpoint_paired_test.csv', index=False)

# Trend model: R_s,k = alpha + beta * log2(N_k) + eps
N_values = np.array([25,50,100,200])
X_list = []; y_list = []
for s in seeds:
    for N in [25,50,100,200]:
        row = sdf[(sdf['seed']==s)&(sdf['N']==N)]
        if len(row)>0:
            X_list.append(np.log2(N)); y_list.append(row['rel'].iloc[0])

X = np.array(X_list); y = np.array(y_list)
Xm = np.column_stack([np.ones(len(X)), X])
beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
y_pred = Xm @ beta
resid = y - y_pred
se_b = np.sqrt(np.diag(np.linalg.inv(Xm.T @ Xm) * np.var(resid)))
t_b = beta[1]/se_b[1]
n = len(X); p_b = 2*stats.t.sf(abs(t_b), n-2)

print(f'\nTrend model: R = {beta[0]:.4f} + {beta[1]:.4f}*log2(N)')
print(f'  beta per N-doubling: {beta[1]:.4f} pp (SE={se_b[1]:.4f}, t={t_b:.3f}, p={p_b:.4f})')

pd.DataFrame([{'intercept':beta[0],'beta_per_log2':beta[1],'se':se_b[1],
    't_stat':t_b,'p_value':p_b,'n_obs':n}]).to_csv(f'{outdir}/scale_trend_model.csv', index=False)

# Monotonicity
mono_count = 0
for s in seeds:
    vals = [sdf[(sdf['seed']==s)&(sdf['N']==N)]['rel'].iloc[0] for N in [25,50,100,200]]
    if vals == sorted(vals):
        mono_count += 1
print(f'\nMonotonic seeds: {mono_count}/{len(seeds)}')

pd.DataFrame([{'monotonic_seeds':mono_count,'total_seeds':len(seeds)}]).to_csv(
    f'{outdir}/scale_monotonicity.csv', index=False)

# Classification
if abs(mean_d) < 0.5 and p_val > 0.05:
    cl = 'SCALE_INVARIANT'
elif p_b < 0.05 and abs(beta[1]) > 0.1:
    cl = 'TRUE_FINITE_MARKET_EFFECT'
elif p_val > 0.05:
    cl = 'RANDOM_SMALL_SAMPLE_EFFECT'
else:
    cl = 'INCONCLUSIVE'

print(f'\nClassification: {cl}')
cli = {'classification': cl, 'delta_mean': float(mean_d), 'delta_p': float(p_val),
    'beta_per_log2': float(beta[1]), 'beta_p': float(p_b), 'monotonic_seeds': mono_count}
json.dump(cli, open(f'{outdir}/scale_classification.json','w'), indent=2)
print(f'Done: {outdir}')
