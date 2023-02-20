#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code to run Principal Component Analysis on Monte Carlo data from Auger SD stations
# used to produce results of my PhD Thesis DOI: 10.5445/IR/1000104548 
# author: alvaro taboada

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
# pyik and scipy for fitting / analysis
from pyik.numpyext import *
from scipy.linalg import block_diag
# sklearn stuff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils import resample

# useful functions and transformations
def Epsilon(u, v, lgSwcd, lgSssd):
	u1 = u[0]
	u2 = u[1]
	u3 = u[2]
	v1 = v[0]
	v2 = v[1]
	v3 = v[2]

	D = (u1*v2 - u2*v1)
	eps = (1./D) * ((u3*(lgSwcd*v2 - lgSssd*v1)) + v3*(lgSssd*u1 - lgSwcd*u2))

	return eps

def EnergyPCA(X, pca):
	u = pca.components_[0]
	v = pca.components_[1]
	mean = pca.mean_
	lgSwcd = X[:,0] - mean[0]
	lgSssd = X[:,1] - mean[1]
	lgEmean = mean[2]

	E = 10**(Epsilon(u, v, lgSwcd, lgSssd)) * 10**lgEmean

	return E

def lnAPCA(X, pca):
	u = pca.components_[0]
	v = pca.components_[1]
	u1, u2, u3, u4 = u
	v1, v2, v3, v4 = v
	mean = pca.mean_

	lgSwcd = X[:,0] - mean[0]
	lgSssd = X[:,1] - mean[1]

	D = (u1*v2 - u2*v1)
	x = (lgSwcd*v2 - lgSssd*v1)
	y = (lgSssd*u1 - lgSwcd*u2)

	lnA = (1. / D) * (u4*x + v4*y) + mean[3]

	return lnA

def lnAEnergyCorrection(lgE, pars):
	lnAcorr = pars[0] * (lgE - pars[1])
	return lnAcorr

def main():

	# dataframe with data previously filtered and stored in a pickle file
	data = pd.read_pickle('./data_pca.p')
	# select data from proton and iron primaries
	dataprot = data[data.primary=='Proton']
	datairon = data[data.primary=='Iron']
	seldata = pd.concat([dataprot, datairon])


	for i, colname in enumerate(seldata.columns):
		if colname == 'primary':
			primary_col_id = i
		elif colname == 'wcd_s38':
			wcd_col_id = i
		elif colname == 'ssd_s38':
			ssd_col_id = i
		elif colname == 'lge_mc':
			energy_col_id = i
		elif colname == 'lnA':
			lnA_col_id = i

		# split data table into data X  and class labels y
		X = seldata.iloc[:, [wcd_col_id, ssd_col_id, energy_col_id, lnA_col_id]].values
		y = seldata.iloc[:, primary_col_id].values
		X[:, 0] = np.log10(X[:,0])
		X[:, 1] = np.log10(X[:,1])
		n = X.shape[1]
		# standardize the data
		X_std = StandardScaler().fit_transform(X) 
		# eigendecomposition - computing eigenvectors and eigenvalues
		# of the covariance matrix
		cov_mat = np.cov(X.T)
		print("Covariance Matrix \n %s" %cov_mat)

		# eigenvectors and eigenvalues of the covariance matrix
		eig_vals, eig_vecs = np.linalg.eig(cov_mat)

		print("Eigenvectors \n%s" %eig_vecs)
		print("\nEigenvalues \n%s" %eig_vals)

		# selecting Principal Components
		# testing that all eigenvectors have unit length 1
		for ev in eig_vecs:
			np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
		print("\nEverything ok!")

		# sort eigenvectors by decreasing eigenvalues
		eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
		eig_pairs.sort()
		eig_pairs.reverse()

		print("\nEigenvalues in descending order: ")
		for i in eig_pairs:
			print(i[0])

		# now the question is, how many principal components are we going to choose for our new feature subspace?
		# a useful measure is the so-called "explained variance", which can be calculated from the eigenvalues.
		# The explained variance tells us how much information (variance) can be attributed to each of the principal components

		tot = sum(eig_vals)
		var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
		print("\nExplained variance:\n%s" %var_exp)

		# compute the projection matrix from the 2 eigenvectors
		matrix_w = np.hstack((eig_pairs[0][1].reshape(n,1),
													eig_pairs[1][1].reshape(n,1)))

		print("Matrix W:\n", matrix_w)

		# Projection onto the new Feature Space
		# Y = X \times W where Y is a len(X) times 2 matrix of our transformed samples
		Y = X.dot(matrix_w)

		# using sklearn
		pca = PCA(n_components=n)
		Y_skelearn = pca.fit_transform(X) # just for testing with 'manual' calculation above

		# obtain Energy and lnA estimates using Principal Components
		epca = EnergyPCA(X, pca)
		lnApca = lnAPCA(X, pca)

		# prepare estimates for modelling / analysis
		lges = seldata.lge_mc.values
		thetas = seldata.theta_mc.values
		lgepca = np.log10(epca)
		etrue = 10**lges
		e_sd = 10**seldata.lge_sd.values
		lgebins = np.array(list(np.unique(lges)) + [20.1])
		avg_bias = (epca - etrue) / etrue
		bias_sd = (e_sd - etrue) / etrue
		bias_prot = avg_bias[y == 'Proton']
		bias_iron = avg_bias[y == 'Iron']
		bias_prot_sd = bias_sd[y == 'Proton']
		bias_iron_sd = bias_sd[y == 'Iron']
		lges_prot = lges[y == 'Proton']
		lges_iron = lges[y == 'Iron']
		thetas_prot = thetas[y=='Proton']
		thetas_iron = thetas[y=='Iron']


		lnAraw_prot = lnApca[y=='Proton']
		lnAraw_iron = lnApca[y=='Iron']
		lgEraw_prot = lgepca[y=='Proton']
		lgEraw_iron = lgepca[y=='Iron']

		lge_bins = np.array([18.7,  18.8,  18.9,  19. ,  19.1,  19.2,  19.3,
				19.4,  19.5,  19.6,  19.7,  19.8,  19.9,  20. ,  20.1])
		lgEcen_prot, lgEhw_prot, ydata_prot, uncs_prot, uncsuncs_prot, ns_prot = profilevarvar(lges_prot, lnAraw_prot,
				bins=lge_bins, usemed=True, sigma_cut=5)

		lgEcen_iron, lgEhw_iron, ydata_iron, uncs_iron, uncsuncs_iron, ns_iron = profilevarvar(lges_iron, lnAraw_iron,
				bins=lge_bins, usemed=True, sigma_cut=5)

		# try simple Chi2 fit (from pyik)
		xdata = lgEcen_prot
		yerrs_prot = uncs_prot / np.sqrt(ns_prot)
		yerrs_iron = uncs_iron / np.sqrt(ns_iron)
		estimates = np.array([1., 19.1])
		#estimates = np.array([4., 18.5])
		pars_prot, cov_prot, chi2_prot, ndof_prot = ChiSquareFunction(lnAEnergyCorrection, xdata, ydata_prot, yerrs_prot).Minimize(estimates)
		pars_iron, cov_iron, chi2_iron, ndof_iron = ChiSquareFunction(lnAEnergyCorrection, xdata, ydata_iron, yerrs_iron).Minimize(estimates)
		
		print("\nlnA(pca) correction with energy")
		print("Fit results (proton): ", pars_prot)
		print("Chi2/ndof = %.2f / %i" %(chi2_prot, ndof_prot))

		print("\nFit results (iron): ", pars_iron) 
		print("Chi2/ndof = %.2f / %i" %(chi2_iron, ndof_iron))


		pars = np.concatenate([pars_prot, pars_iron])
		COV  = block_diag(cov_prot, cov_iron)

		# plot data with fit
		xplot = np.linspace(18.4, 20.1, 50)
		yplot_prot = lnAEnergyCorrection(xplot, pars_prot)
		yplot_iron = lnAEnergyCorrection(xplot, pars_iron)
		# compute errors on the model
		yplot_err_prot = []
		yplot_err_iron = []
		for i, xi in enumerate(xplot):
			yplot_err_prot.append(uncertainty(flnAEnergyCorrection, pars_prot, cov_prot))
			yplot_err_iron.append(uncertainty(flnAEnergyCorrection, pars_iron, cov_iron))

		yplot_err_prot = np.array(yplot_err_prot) 
		yplot_err_iron = np.array(yplot_err_iron) 

		# plot raw lnA from PCA	
		fig = plt.figure()
		sfig = fig.add_axes([0.15, 0.11, 0.845, 0.78])
		plt.xlabel(r'$\lg (E_\mathrm{mc} / \si{eV})$')
		plt.ylabel(r'$\langle \ln A_\mathrm{pca} \rangle$')
		plt.errorbar(xdata-lgEhw_prot[0], ydata_prot, yerrs_prot, ls='None', marker='s', color='r', mec='r', mfc='white', ms=6,
				elinewidth=2, capsize=0, mew=1.5, label='Proton')
		plt.errorbar(xdata-lgEhw_iron[0], ydata_iron, yerrs_iron, ls='None', marker='s', color='b', mec='b', mfc='white', ms=6,
				elinewidth=2, capsize=0, mew=1.5, label='Iron')

		plt.plot(xplot, yplot_prot, lw=1, color='r')
		plt.plot(xplot, yplot_iron, lw=1, color='b')
		plt.fill_between(xplot, yplot_prot - yplot_err_prot, yplot_prot + yplot_err_prot, color='r', alpha=0.2)
		plt.fill_between(xplot, yplot_iron - yplot_err_iron, yplot_iron + yplot_err_iron, color='b', alpha=0.2)

		plt.legend(loc='upper left')
		plt.xlim(18.4, 20.1)
		plt.ylim(-2, 6)
		plt.savefig('./plots/PCA_results.png')

if __name__ == "__main__":
	main()