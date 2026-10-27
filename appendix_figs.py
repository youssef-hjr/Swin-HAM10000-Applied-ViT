import numpy as np, pandas as pd, matplotlib, sys, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score
from PIL import Image
V = sys.argv[1]; OUT = f'{V}/appendix'
NAVY, TEAL, ORANGE, GREEN, ICE = '#14213D', '#1F6F8B', '#E76F51', '#2A9D8F', '#8FB8C9'
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
CL = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']; M = [0, 2, 3]
meta = pd.read_csv(f'{V}/data/meta.csv')
meta['tone'] = pd.cut(meta.ita, [-90, 41, 55, 90], labels=['Darker', 'Light', 'Very light'])
Z = {t: np.load(f'{V}/run/outputs/pred_{t}.npz') for t in ['22k', '1k']}
te = Z['22k']['te']; yt = meta.label.values[te]
models = {'ImageNet-22k, linear probe': Z['22k']['p_lp'], 'ImageNet-22k, fine-tuned': Z['22k']['p_test'],
          'ImageNet-1k, linear probe': Z['1k']['p_lp'], 'ImageNet-1k, fine-tuned': Z['1k']['p_test']}
# A1 confusion matrices for all four models
fig, ax = plt.subplots(2, 2, figsize=(10, 9))
for a, (n, p) in zip(ax.ravel(), models.items()):
    cm = confusion_matrix(yt, p.argmax(1), labels=range(7), normalize='true')
    a.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(7):
        for j in range(7): a.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=7.5, color='white' if cm[i, j] > .5 else NAVY)
    a.set_xticks(range(7)); a.set_yticks(range(7)); a.set_xticklabels(CL, fontsize=8); a.set_yticklabels(CL, fontsize=8)
    a.set_title(n, fontsize=10.5); a.set_xlabel('predicted', fontsize=9); a.set_ylabel('true', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/figA_cm_all.png'); plt.close()
# A2 reliability diagrams (22k LP vs FT, 1k FT)
def rel(p, t, bins=10):
    conf = p.max(1); acc = (p.argmax(1) == t); e = np.linspace(0, 1, bins + 1); xs, ys, ns = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.sum() >= 5: xs.append(conf[s].mean()); ys.append(acc[s].mean()); ns.append(s.sum())
    return xs, ys
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([0, 1], [0, 1], ls='--', c='grey', lw=1, label='perfect calibration')
for n, col in [('ImageNet-22k, linear probe', ICE), ('ImageNet-22k, fine-tuned', ORANGE), ('ImageNet-1k, fine-tuned', TEAL)]:
    x, y = rel(models[n], yt); ax[0].plot(x, y, 'o-', c=col, label=n)
ax[0].set_xlabel('mean confidence'); ax[0].set_ylabel('observed accuracy'); ax[0].set_title('Reliability diagram (test set)'); ax[0].legend(fontsize=8)
for n, col in [('ImageNet-22k, linear probe', ICE), ('ImageNet-22k, fine-tuned', ORANGE)]:
    ax[1].hist(models[n].max(1), bins=20, range=(0, 1), alpha=.7, color=col, label=n)
ax[1].set_xlabel('confidence of predicted class'); ax[1].set_ylabel('images'); ax[1].set_title('Confidence distribution'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/figA_calibration.png'); plt.close()
# A3 malignancy threshold trade-off
p = models['ImageNet-22k, fine-tuned']; s = p[:, M].sum(1); mal = np.isin(yt, M)
th = np.linspace(0.05, 0.95, 91)
sens = [(s[mal] >= x).mean() for x in th]; spec = [(s[~mal] < x).mean() for x in th]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(th, sens, c=ORANGE, label='sensitivity (malignant caught)'); ax[0].plot(th, spec, c=TEAL, label='specificity (benign cleared)')
for x in [0.5, 0.3]:
    ax[0].axvline(x, ls=':', c='grey'); ax[0].text(x + .01, .05, f't={x}: sens {(s[mal] >= x).mean():.2f}, spec {(s[~mal] < x).mean():.2f}', fontsize=7.5, rotation=90, va='bottom')
ax[0].set_xlabel('threshold on P(malignant)'); ax[0].set_title('Sensitivity and specificity vs threshold'); ax[0].legend(fontsize=8, loc='center left')
for c, col in [(1, ICE), (4, NAVY), (0, ORANGE), (2, GREEN), (3, TEAL)]:
    ax[1].hist(s[yt == c], bins=20, range=(0, 1), histtype='step', lw=1.8, color=col, density=True, label=CL[c])
ax[1].set_xlabel('P(malignant) = P(MEL)+P(BCC)+P(AKIEC)'); ax[1].set_ylabel('density'); ax[1].set_title('Malignancy score by true class'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/figA_threshold.png'); plt.close()
out = {'sens50': float((s[mal] >= .5).mean()), 'spec50': float((s[~mal] < .5).mean()), 'sens30': float((s[mal] >= .3).mean()), 'spec30': float((s[~mal] < .3).mean())}
# A4 ROC by skin tone
tones = meta.tone.values[te]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for g, col in [('Very light', ICE), ('Light', TEAL), ('Darker', ORANGE)]:
    k = tones == g; f, tp, _ = roc_curve(mal[k], s[k])
    ax.plot(f, tp, c=col, lw=2, label=f'{g}: AUC {roc_auc_score(mal[k], s[k]):.3f} ({mal[k].sum()} malignant)')
ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1); ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate'); ax.set_title('ROC by skin-tone proxy (ImageNet-22k FT)'); ax.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(f'{OUT}/figA_roc_tone.png'); plt.close()
# A5 example images per tone group
X = np.load(f'{V}/data/X224.npy', mmap_mode='r'); r = np.random.default_rng(1)
fig, ax = plt.subplots(3, 6, figsize=(10, 5.4))
for i, g in enumerate(['Very light', 'Light', 'Darker']):
    idx = r.choice(np.where(meta.tone.values == g)[0], 6, replace=False)
    for j, k in enumerate(idx):
        ax[i, j].imshow(X[k]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, j].set_title(f'{CL[meta.label[k]]}, ITA {meta.ita[k]:.0f}°', fontsize=8)
        for sp in ax[i, j].spines.values(): sp.set_visible(False)
    ax[i, 0].set_ylabel(g, fontsize=11, color=[TEAL, NAVY, ORANGE][i])
plt.tight_layout(); plt.savefig(f'{OUT}/figA_tone_examples.png', dpi=150); plt.close()
# per-class table
rows = []
for n in ['ImageNet-22k, fine-tuned', 'ImageNet-1k, fine-tuned']:
    pr, rc, f1, sup = precision_recall_fscore_support(yt, models[n].argmax(1), labels=range(7), zero_division=0)
    rows.append({'model': n, 'precision': pr.tolist(), 'recall': rc.tolist(), 'f1': f1.tolist(), 'support': sup.tolist()})
out['perclass'] = rows
json.dump(out, open(f'{OUT}/appendix.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'perclass'}))
for r_ in rows: print(r_['model'], [round(x, 2) for x in r_['precision']], [round(x, 2) for x in r_['f1']], r_['support'])
