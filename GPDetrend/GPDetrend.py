from mpl_toolkits.axes_grid.inset_locator import inset_axes
import exotoolbox
import batman
import seaborn as sns
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pymultinest
from scipy import interpolate
import numpy as np
import utils
import os

parser = argparse.ArgumentParser()
# This reads the lightcurve file. First column is time, second column is flux:
parser.add_argument('-lcfile', default=None)
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument('-eparamfile', default=None)
# This defines which of the external parameters you want to use, separated by commas.
# Default is all:
parser.add_argument('-eparamtouse', default='all')
# This reads the external parameters to fit (assumed to go in the columns):
parser.add_argument('-compfile', default=None)
# This defines which comparison stars, if any, you want to use, separated by commas.
# Default is all:
parser.add_argument('-comptouse', default='all')
# This reads an output folder:
parser.add_argument('-ofolder', default='')
# Number of live points:
parser.add_argument('-nlive', default=1000)
args = parser.parse_args()

# Extract lightcurve and external parameters. When importing external parameters, 
# standarize them and save them on the matrix X:
lcfilename = args.lcfile
tall,fall,f_index = np.genfromtxt(lcfilename,unpack=True,usecols=(0,1,2))
idx = np.where(f_index == 0)[0]
t,f = tall[idx],fall[idx]
out_folder = args.ofolder

eparamfilename = args.eparamfile
eparams = args.eparamtouse
data = np.genfromtxt(eparamfilename,unpack=True)
for i in range(len(data)):
    x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
    if i == 0:
        X = x
    else:
        X = np.vstack((X,x))
if eparams != 'all':
    idx_params = np.array(eparams.split(',')).astype('int')
    X = X[idx_params,:]

compfilename = args.compfile
if compfilename is not None:
    comps = args.comptouse
    data = np.genfromtxt(compfilename,unpack=True)
    for i in range(len(data)):
        x = (data[i] - np.mean(data[i]))/np.sqrt(np.var(data[i]))
        if i == 0:
            Xc = x
        else:
            Xc = np.vstack((Xc,x))
    if comps != 'all':
        idx_params = np.array(comps.split(',')).astype('int')
        Xc = Xc[idx_params,:]

# Other inputs:
n_live_points = int(args.nlive)

# Cook the george kernel:
import george
kernel = np.var(f)*george.kernels.ExpSquaredKernel(np.ones(X[:,idx].shape[0]),ndim=X[:,idx].shape[0],axes=range(X[:,idx].shape[0]))
# Cook jitter term
jitter = george.modeling.ConstantModel(np.log((200.*1e-6)**2.))

# Wrap GP object to compute likelihood
gp = george.GP(kernel, mean=0.0,fit_mean=False,white_noise=jitter,fit_white_noise=True)
gp.compute(X[:,idx].T)

# Now define MultiNest priors and log-likelihood:
def prior(cube, ndim, nparams):
    # Prior on "median flux" is uniform:
    cube[0] = utils.transform_uniform(cube[0],-2.,2.)
    # Pior on the log-jitter term (note this is the log VARIANCE, not sigma); from 0.01 to 100 ppm:
    cube[1] = utils.transform_uniform(cube[1],np.log((0.01e-3)**2),np.log((100e-3)**2))
    pcounter = 2
    # Prior on coefficients of comparison stars:
    if compfilename is not None:
        for i in range(Xc.shape[0]):
            cube[pcounter] = utils.transform_uniform(cube[pcounter],-10,10)
            pcounter += 1
    # Prior on kernel maximum variance; from 0.01 to 100 mmag: 
    cube[pcounter] = utils.transform_loguniform(cube[pcounter],(0.01*1e-3)**2,(100*1e-3)**2)
    pcounter = pcounter + 1
    # Now priors on the alphas = 1/lambdas; gamma(1,1) = exponential, same as Gibson+:
    for i in range(X.shape[0]):
        cube[pcounter] = utils.transform_exponential(cube[pcounter])
        pcounter += 1    
def loglike(cube, ndim, nparams):
    # Evaluate the log-likelihood. For this, first extract all inputs:
    mflux,ljitter = cube[0],cube[1]
    pcounter = 2
    model = mflux
    if compfilename is not None:
        for i in range(Xc.shape[0]):
            model = model + cube[pcounter]*Xc[i,idx]
            pcounter += 1
    max_var = cube[pcounter]
    pcounter = pcounter + 1
    alphas = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        alphas[i] = cube[pcounter]
        pcounter = pcounter + 1
    gp_vector = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas))
    
    # Evaluate model:     
    residuals = f - model
    gp.set_parameter_vector(gp_vector)
    return gp.log_likelihood(residuals)

n_params = 3 + X.shape[0]
if compfilename is not None:
    n_params +=  Xc.shape[0]

out_file = out_folder+'out_multinest_trend_george_'

import pickle
# If not ran already, run MultiNest, save posterior samples and evidences to pickle file:
if not os.path.exists(out_folder+'posteriors_trend_george.pkl'):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    mc_samples = output.get_equal_weighted_posterior()[:,:-1]
    a_lnZ = output.get_stats()['global evidence']
    out = {}
    out['posterior_samples'] = mc_samples
    out['lnZ'] = a_lnZ
    pickle.dump(out,open(out_folder+'posteriors_trend_george.pkl','wb'))
else:
    mc_samples = pickle.load(open(out_folder+'posteriors_trend_george.pkl','rb'))['posterior_samples']
    

# Extract posterior parameter vector:
cube = np.median(mc_samples,axis=0)
cube_var = np.var(mc_samples,axis=0)

mflux,ljitter = cube[0],cube[1]
pcounter = 2
model = mflux
if compfilename is not None:
    for i in range(Xc.shape[0]):
        model = model + cube[pcounter]*Xc[i,idx]
        pcounter += 1
max_var = cube[pcounter]
pcounter = pcounter + 1
alphas = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    alphas[i] = cube[pcounter]
    pcounter = pcounter + 1
gp_vector = np.append(np.append(ljitter,np.log(max_var)),np.log(1./alphas))

# Evaluate model:     
residuals = f - model
gp.set_parameter_vector(gp_vector)

# Get prediction from GP:
pred_mean, pred_var = gp.predict(residuals, X.T, return_var=True)
pred_std = np.sqrt(pred_var)

model = mflux
pcounter = 2
if compfilename is not None:
    for i in range(Xc.shape[0]):
        model = model + cube[pcounter]*Xc[i,:]
        pcounter += 1

fout,fout_err = exotoolbox.utils.mag_to_flux(fall-model,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
plt.errorbar(tall - int(tall[0]),fout,yerr=fout_err,fmt='.')
pred_mean_f,fout_err = exotoolbox.utils.mag_to_flux(pred_mean,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
plt.plot(tall - int(tall[0]),pred_mean_f)
plt.show()
fall = fall - model - pred_mean
#plt.errorbar(tall,fall,yerr=np.ones(len(tall))*np.sqrt(np.exp(ljitter)),fmt='.')
#plt.show()
plt.errorbar(tall - int(tall[0]),fall,yerr=np.ones(len(tall))*np.sqrt(np.exp(ljitter)),fmt='.')
plt.show()
fout,fout_err = exotoolbox.utils.mag_to_flux(fall,np.ones(len(tall))*np.sqrt(np.exp(ljitter)))
fileout = open('detrended_lc.dat','w')
for i in range(len(tall)):
    fileout.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(tall[i],fout[i],fout_err[i]))
fileout.close()
plt.errorbar(tall - int(tall[0]),fout,yerr=fout_err,fmt='.')
plt.xlabel('Time (BJD - '+str(int(tall[0]))+')')
plt.ylabel('Relative flux')
plt.show()
"""
# Plot:
sns.set_context("talk")
sns.set_style("ticks")
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['axes.linewidth'] = 1.2 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.markeredgewidth'] = 1 
fig = plt.figure(figsize=(10,4.5))
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

tzero = int(t[0])
# Plot MAP solution:
ax = plt.subplot(gs[0])
color = "cornflowerblue"
plt.plot(t-tzero, f,".k",markersize=1,label='K2 data')
plt.plot(t-tzero, pred_mean + mflux + model, linewidth=1, color='red',label='GP',alpha=0.5)
#plt.np.min(t-tzero)-0.1,np.max(t-tzero)+0.1
#plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
#plt.plot([10.,1],[np.max(pred_mean+mflux),0.995],'black',linewidth=1,alpha=0.5)
#plt.plot([11.,22.],[np.max(pred_mean+mflux),0.995],'black',linewidth=1,alpha=0.5)
plt.ylabel('Relative flux')
plt.legend(loc='lower right')

# Get prediction from GP to get residuals:
ax = plt.subplot(gs[1])
pred_mean, pred_var = gp.predict((f-mflux-model), X.T, return_var=True)
#plt.errorbar(t, (f-theta[0]-pred_mean)*1e6, yerr=ferr*1e6, fmt=".k",label='K2 data',markersize=1,alpha=0.1)
plt.plot(t-tzero,(f-mflux - model - pred_mean)*1e6,'.k',markersize=1)
print 'rms:',np.sqrt(np.var((f-mflux-pred_mean)*1e6))
print 'med error:',np.median(ferr*1e6)
plt.xlabel('Time (BJD-'+str(tzero)+')')
plt.ylabel('Residuals')
#plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
plt.tight_layout()
plt.savefig(out_folder+'GP_fit_george.pdf')
"""
