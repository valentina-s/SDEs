# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:36:54 2015

@author: val
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import pandas as pd
import random as random
import unittest
from scipy.optimize import minimize
from scipy import stats
import warnings
import time
import multiprocessing as mp

def generateSample():
    z = random.normalvariate(0,1)
    return(z)
    
def likelihoodRatio(x,param):
    if stats.expon.pdf(x) == 0:
        warnings.warn('The generated sample point has zero probability.')
        # do I need a warning for the pdf from which we sample
        # it should never be zero
    l = stats.expon.pdf(x)/stats.norm.pdf(x)
    return(l)

# param = 4
# random.seed(0)# using random.seed instead of no.random.seed since I generate samples with random.normalvariate

# sample = importanceSampling_noparameters(generateSample,likelihoodRatio,param,10)
# print ('First sample is '+ str(sample))




random.seed(0)
mu = 0
def generateSample(mu):
    z = random.normalvariate(mu,1)
    return(z)
    
param  = 5
    
sample = importanceSampling(generateSample,[mu],likelihoodRatio,[param],10)
# sample = importanceSampling_noparameters(generateSample,likelihoodRatio,param,10)
# print('Second sample is '+str(sample))




temp = 3

def generateSample():
    z = random.normalvariate(0,1)
    return(z)



def f(n):
    z = generateSample()
    l = likelihoodRatio(z,temp)
    return([z,l])
        
p = mp.Pool(8)
# print('This is the parallel output')
# print(p.map(f,range(10)))
sample = p.map(f,range(10))

N = 100000

start = time.time()
sample = importanceSampling(generateSample,[],likelihoodRatio,[3],N)
end = time.time()
elapsed_time = end - start
print('Non-parallel time is '+ str(elapsed_time))

start = time.time()
sample = importanceSampling_parallel2(N)
end = time.time()
elapsed_time = end - start
print('Parallel time is '+ str(elapsed_time))


# run importance sampling for diffusions:

# 1) define a sampling function
# I should already have it
# 2) define a define a likelihood function
# generate brownian motion 



def try_ImportanceSampling_withDiffusions():
    def drift(x, param):
            # constant drift
            return(np.ones(x.shape)*param)
    def diffusion(x):
            # assume parameters are known
            # returns an np.array
            D = np.eye(x.shape[0])
            return(D)
            
    dt = 0.1
    T = 1
    theta = 4
    tol = 0.01
    x0 = np.zeros((3,))
    x = sampleDiffusion(x0,drift,[theta],diffusion,[],dt, T)
    N = 10
    def likelihoodRatio(data,drift,param,diffusion,dt):
        data_diff = data.diff().iloc[1::] # makes start from time 2  
        # add the inversion of the diffusion matrices
        ratio = dt * drift(data.iloc[1::],param)*drift(data.iloc[1::],param)/2\
        - drift(data.iloc[1::],param)*np.array(data_diff)
        return(sum(sum(ratio)))
        
    sample = importanceSampling(sampleDiffusion,[x0,drift,[0],diffusion,[],0.1,10],likelihoodRatio,[drift,param,diffusion,dt],N)

    return(sample)
    
    
def define_l(data, drift, diffusion, dt):
    def l(param,drift,diffusion,dt):
        data_diff = data.diff().iloc[1::] # makes start from time 2  
        # add the inversion of the diffusion matrices
        ratio = dt * drift(data.iloc[1::],param)*drift(data.iloc[1::],param)/2        - drift(data.iloc[1::],param)*np.array(data_diff)
        return(sum(sum(ratio)))
    return l

def define_likelihoodRatio(data, drift, diffusion, dt):
    def likelihoodRatio(param,drift,diffusion,dt):
        data_diff = data.diff().iloc[1::] # makes start from time 2  
        # add the inversion of the diffusion matrices
        ratio = dt * drift(data.iloc[1::],param)*drift(data.iloc[1::],param)/2        - drift(data.iloc[1::],param)*np.array(data_diff)
        return(sum(sum(ratio)))
    return likelihoodRatio
    
    
    
def define_l(data, drift, diffusion, dt):
    def l(param):
        data_diff = data.diff().iloc[1::] # makes start from time 2  
        # add the inversion of the diffusion matrices
        ratio = dt * drift(data.iloc[1::],param)*drift(data.iloc[1::],param)/2\
        - drift(data.iloc[1::],param)*np.array(data_diff)
        return(np.sum(ratio))
    return l
    
    
theta_hat = []
for dt in [1,0.1,0.01]:
    data = sampleDiffusion(np.zeros((100,)),drift,[5],diffusion,[],dt,10)
    L = define_l(data,drift,diffusion,dt)
    theta_hat.append(minimize(L,0,method = 'Nelder-Mead').x[0])


# Adding more observations does not help!!!! 




# Can I observe it from the estimates.
# Do an OU example
def OU_drift(x,theta):
    mu = 5
    return(theta*(mu - np.array(x)))
    
data = sampleDiffusion(np.zeros((1,)),OU_drift,[1],diffusion,[],0.01, 10)    
N = data.shape[1]
theta_hat = []
index = data.index
for dt in [1,0.1,0.01,0.001,0.0001]:
    L = define_l(data.loc[index[range(0,N)]],OU_drift,diffusion,dt)
    theta_hat.append(minimize(L,0,method = 'Nelder-Mead').x[0])
    
    


