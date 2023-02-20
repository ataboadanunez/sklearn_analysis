#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code to run Linear Discriminant Analysis on Monte Carlo data from Auger SD stations
# used to produce results of my PhD Thesis DOI: 10.5445/IR/1000104548 
# author: alvaro taboada

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
# sklearn stuff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# dataframe with data previously stored in a pickle file
data = pd.read_pickle('./data.p')

# apply filters on energy, zenith angle and reconstructed signal size
selenergy = 19.5
mintheta = 0  #theta_arr[z]
maxtheta = 56 #theta_arr[z+1]

seldata = data[(data.lge_mc == selenergy) & (data.theta_mc >= mintheta) & (data.theta_mc <= maxtheta) & (data.ssd_size_rec.values > 1)]

# prepare data for LDA 
for i, colname in enumerate(seldata.columns):
	if colname == 'primary':
		primary_col_id = i
	elif colname == 'wcd_s38':
		wcd_col_id = i
	elif colname == 'ssd_s38':
		ssd_col_id = i
	elif colname == 'lge_sd_corr':
		lge_col_id = i

X = seldata.iloc[:, [wcd_col_id, ssd_col_id]].apply(np.log10).values
y_raw = seldata.iloc[:, primary_col_id].values
enc = LabelEncoder()
label_encoder = enc.fit(y_raw)
y = label_encoder.transform(y_raw) + 1

nclass = int(max(np.unique(y)))
N = X.shape[1] # number of features

# LDA via scikit-learn
lda = LDA(n_components=2, store_covariance=True)
y_pred = lda.fit_transform(X, y)
# 
y_prot = y_pred[y==2]
y_iron = y_pred[y==1]

# plot results
fig, splot = plt.subplots()
plt.xlabel(r'$\lg(S^\mathrm{wcd}_{38} / \si{VEM})$')
plt.ylabel(r'$\lg(S^\mathrm{ssd}_{38} / \si{MIP})$')
# plot scatter points
plt.scatter(X[:, 0][y==2], X[:, 1][y==2], marker='.', color='red', label='Proton')
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], marker='.', color='blue', label='Iron')
# plot proton and iron areas 
nx, ny = 200, 200
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
										 np.linspace(y_min, y_max, ny))
Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
							 norm=colors.Normalize(0., 1.), zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
plt.xlim(min(X[:,0]), max(X[:,0]))
plt.ylim(min(X[:,1]), max(X[:,1]))
cornertext([r'$\lg(E_\mathrm{mc} / \si{eV} = %.1f)$' %selenergy,
						r'$%i \leq \theta_\mathrm{mc} / ^\circ \leq %i$' %(mintheta, maxtheta)],
						frameon=True, loc='upper left')
legend_elements = [Line2D([0], [0], lw=0, marker='o', markersize=4, color='r', mec='r', label='Proton'),
									 Line2D([0], [0], lw=0, marker='o', markersize=4, color='b', mec='b', label='Iron')]
legend2 = plt.legend(handles=legend_elements, loc='best', frameon=True)
plt.savefig('./plots/LDA_results.png')

############## MERIT FACTOR ########################

# compute merit-factor (separation power between two distributions)
mp = np.mean(y_prot)
mi = np.mean(y_iron)
stdp = np.std(y_prot)
stdi = np.std(y_iron)
mf = np.abs(mp - mi) / np.sqrt(stdp**2 + stdi**2)

# apply bootstrap technique to estimate uncertainty on the merit factor
nprot = len(y_prot)
niron = len(y_iron)
mf_boot = []
for i in range(500):
	boot_prot = resample(y_prot, replace=True, n_samples=nprot)
	boot_iron = resample(y_iron, replace=True, n_samples=niron)
	mbp = np.mean(boot_prot)
	mbi = np.mean(boot_iron)
	stdbp = np.std(boot_prot)
	stdbi = np.std(boot_iron)
	mf_boot.append(np.abs(mbp - mbi) / np.sqrt(stdbp**2 + stdbi**2))

mf_boot = np.array(mf_boot)
n_mf_boot, bins_mf_boot = np.histogram(mf_boot, bins=30)


fig = plt.figure()
plt.xlabel(r'$\mathrm{MF}$')
plt.ylabel(r'Counts')
plot_hist(bins_mf_boot, n_mf_boot, lw=2, facecolor='k', alpha=0.5)
#plt.hist(mf_boot, bins=40, lw=2, histtype='step', facecolor='k', alpha=0.5)
plt.axvline(np.mean(mf_boot), lw=2, color='k', label=r'$\langle \mathrm{MF}_\mathrm{boot} \rangle = %.2f$' %np.mean(mf_boot))
plt.axvline(np.mean(mf), lw=2, color='k', ls='--', label=r'$\mathrm{MF} = %.2f$' %mf)
plt.legend(loc='upper right')
plt.savefig('./plots/MF.png')