import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import pandas as pd
import random as random
import unittest
from scipy.optimize import minimize
from scipy import stats
import multiprocessing as mp

random.seed(0)


# Diffusion Sampling
T = 10 # final time
dt = 0.1 # time step 
# make dt difference in t's
# pass x0
# pass drift as a function
# pass diffusion as a function




def drift(x, param):
        # should be a list of paramters
        # specify it is just a constant
        # returns a data frame 
        # return(np.array(x*param))
        return(np.ones(x.shape)*param)
        
        

    
def diffusion(x):
    # assume parameters are known
    # returns an np.array
    D = np.eye(x.shape[0])
    return(D)


def sampleDiffusion_oneparameter(x0,drift,param,diffusion,dt,T):
    """ x = sampleDiffusion(x0,drift,param,diffusion,dt,T)
        sampleDiffusion samples a path of a diffusion process with given drift and diffusion functions
        
    """
# change to allow a starting point different from zero
    
    # N = T/dt; # add case T is not divisible by dt 
    # create an index for 
    index = np.arange(0,T+dt,dt)

    N = index.shape[0]
    x = pd.DataFrame(np.zeros((N,x0.shape[0])),np.array(index))
    x.loc[0] = x0

    i = 1
    for t in index[:-1]:
        # generate random vector
        z = random.normalvariate(np.zeros((x.shape[1],)), np.ones(x.shape[1]))   
        x.iloc[i] = x.loc[t] + dt*drift(x.loc[t],param) + \
        np.sqrt(dt) * np.dot(diffusion(x.loc[t]).T , z)     
        i = i+1
    return(x)
    
def sampleDiffusion(x0,drift,drift_args,diffusion,diffusion_args,dt,T):
    """ x = sampleDiffusion(x0,drift_args,diffusion,diffusion_args,dt,T)
        sampleDiffusion samples a path of a diffusion process with given drift and diffusion functions.
        The function uses a Euler integration scheme with a fixed time step dt and final time T.
        At this stage it assumes the initial time is zero, but this can be modified.
        x0  - is np.array of any dimension (I believe I want it to be a vector).        
        The output is a pandas data frame.
    """
# change to allow a starting point different from zero
    
    # N = T/dt; # add case T is not divisible by dt 
    # create an index for 
    index = np.arange(0,T+dt,dt)
    N = index.shape[0]
    x = pd.DataFrame(np.zeros((N,x0.shape[0])),np.array(index))
    x.loc[0] = x0

    i = 1
    for t in index[:-1]:
        # generate random vector
        z = np.random.multivariate_normal(np.zeros((x.shape[1],)), np.eye(x.shape[1]))
        # evaluate next step
        x.iloc[i] = x.loc[t] + dt*drift(x.loc[t],*drift_args) + \
        np.sqrt(dt) * np.dot(diffusion(x.loc[t],*diffusion_args).T , z)     
        i = i+1
    return(x)  
    
def sampleDiffusion_timedependent(x0,drift,drift_args,diffusion,diffusion_args,dt,T):
    """ x = sampleDiffusion(x0,drift_args,diffusion,diffusion_args,dt,T)
        sampleDiffusion samples a path of a diffusion process with given drift and diffusion functions.
        The function uses a Euler integration scheme with a fixed time step dt and final time T.
        At this stage it assumes the initial time is zero, but this can be modified.
        x0  - is np.array of any dimension (I believe I want it to be a vector).        
        The output is a pandas data frame.
    """
# change to allow a starting point different from zero
    
    # N = T/dt; # add case T is not divisible by dt 
    # create an index for 
    index = np.arange(0,T+dt,dt)
    N = index.shape[0]
    x = pd.DataFrame(np.zeros((N,x0.shape[0])),np.array(index))
    x.loc[0] = x0

    i = 1
    for t in index[:-1]:
        # generate random vector
        z = np.random.multivariate_normal(np.zeros((x.shape[1],)), np.eye(x.shape[1]))
        # evaluate next step
        x.iloc[i] = x.loc[t] + dt*drift(x.loc[t],t,*drift_args) + \
        np.sqrt(dt) * np.dot(diffusion(x.loc[t],*diffusion_args).T , z)     
        i = i+1
    return(x) 
    
    
def sampleDiffusionBridge(x0,x1,drift,drift_args,diffusion,diffusion_args,dt,T):
    # we can do it directly if we have the transition density
    # otherwise need IS formulation
    # test first for Gaussian Bridge

    index = np.arange(0,T+dt,dt)
    N = index.shape[0]
    x = pd.DataFrame(np.zeros((N,x0.shape[0])),np.array(index))

    x.loc[0] = x0
    x.loc[T] = x1 # T+dt
    i = 1
    for t in index[:-1]:
        z = np.random.multivariate_normal(np.zeros((x.shape[1],)), np.eye(x.shape[1]))
        x.iloc[i] = x.loc[t] + dt*drift(x.loc[t],*drift_args) + \
        dt*(x.loc[T] - x.loc[t])/(T - t) + \
        np.sqrt(dt) * np.dot(diffusion(x.loc[t],*diffusion_args).T , z) 
        i = i+1
    return(x)
    
    
    
    
    
    
    # Exercise:
    # sample from a diffusion bridge with a drift

# def sampleDiffusionBridge_IS():
     # generate samples from Brownian bridge
     # attach weights based on likelihood ratios
 
 
 # Create a framework for Importance Sampling
 
 # 1) Generate samples - any sort of object
 # 2) Evaluate likelihood of this object

# Store samples in a list
# pass a function to generate a sample
# pass a function which calculates the weight
# 
# class ImportanceSample:
#    def generateObservation():


# create importance sampling function
def generateSample():
    z = random.normalvariate(0,1)
    return(z)
    
def likelihoodRatio(x,param):
    l = stats.expon.pdf(x)/stats.norm.pdf(x)
    return(l)
    
def likelihoodRatio1(x,l1,args1,l2,args2):   
    r = l1(x,*args1)/l2(x,*args2)
    return(r)
    
    

# rescale     
def likelihoodRatio_Gaussian(x,param):
    """ This function tests importance sampling\
        when sampling from centered gaussian \
        to generate a gaussian with a fixed mean
    """
    
    mu = param[0]
    sigma = param[1]
    l1 = np.exp(-(mu - x)**2/(2*sigma*sigma))
    l2 = np.exp(-x**2/(2*sigma*sigma))
    #return(np.log(l1)- np.log(l2))
    return(l1/l2)
    

def importanceSampling_Gaussian(generateSample, logLikelihood, param,N):  
    """ A function implementing importance sampling for a general object.
        [sample, total_weights] = importanceSampling_Gaussian(generateSample, logLikelihood, param,N)
    """
    
    sample = []    
    for i in range(0,N):
        z = generateSample()
        l = logLikelihood(z,param)
        sample.append([z,l])
        
    total_weights = sum([sublist[1] for sublist in sample])
    # normalize the weights
    sample =  [[sublist[0], sublist[1]/total_weights] for sublist in sample]
    return(sample)
 
 
# This function is already contained in importanceSampling
 
#def importanceSampling_noparameters(generateSample, likelihoodRatio, param,N): 
#    sample = [] 
#    print(np.random.exponential(1))
#    for i in range(0,N):
#        z = generateSample()
#        l = likelihoodRatio(z,param)
#        sample.append([z,l])
#        
#    total_weights = sum([sublist[1] for sublist in sample])
#    # normalize the weights
#    sample =  [[sublist[0], sublist[1]/total_weights] for sublist in sample]
#    return(sample)
    
    
    
    
# class importanceSampling:
    
def systematicResampling(sample):
    # assumes the weights are normalized
    weights = [sublist[1] for sublist in sample]
    N =  len(weights)
    r = random.random()  
    index = np.arange(N)
    cdf = np.cumsum(weights)
    T = np.arange(0,1,1/N) + r/N
    
    resampled_sample = sample    
    i = 0
    j = 0
    while i<N and j<N:
        while cdf[j]<T[i]:
            j=j+1
        index[i] = j
        resampled_sample[i][0] = sample[j][0]
        resampled_sample[i][1] = 1/N
        
        i = i+1
        
    return (resampled_sample)
       
        
        
        
    
    
def importanceSampling(generateSample, args_generateSample,likelihoodRatio, args_likelihoodRatio,N): 
    sample = []    
    for i in range(0,N):
        z = generateSample(*args_generateSample)
        l = likelihoodRatio(z,*args_likelihoodRatio)
        sample.append([z,l])
        
    total_weights = sum([sublist[1] for sublist in sample])
    # normalize the weights
    sample =  [[sublist[0], sublist[1]/total_weights] for sublist in sample]
    return(sample)
    
def importanceSampling_parallel(generateSample, args_generateSample,likelihoodRatio, args_likelihoodRatio,N): 
    # not working

#    sample = []    
#    for i in range(0,N):
#        z = generateSample(*args_generateSample)
#        l = likelihoodRatio(z,*args_likelihoodRatio)
#        sample.append([z,l])
#        
#    total_weights = sum([sublist[1] for sublist in sample])
#    # normalize the weights
#    sample =  [[sublist[0], sublist[1]/total_weights] for sublist in sample]
    
    
    
    pool = mp.Pool(processes=1)
    def func(n):
        z = generateSample(*args_generateSample)
        l = likelihoodRatio(z,*args_likelihoodRatio)
        return(z)
        
    print(pool.map(func, range(N)))
    
    # pool.close()
    # print(pool.join())
    
    return(pool.map(func,range(N)))
    
def importanceSampling_parallel1(generateSample, args_generateSample,likelihoodRatio, args_likelihoodRatio,N): 
    # not working
#    sample = []    
#    for i in range(0,N):
#        z = generateSample(*args_generateSample)
#        l = likelihoodRatio(z,*args_likelihoodRatio)
#        sample.append([z,l])
#        
#    total_weights = sum([sublist[1] for sublist in sample])
#    # normalize the weights
#    sample =  [[sublist[0], sublist[1]/total_weights] for sublist in sample]
    
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
    temp = 1
    
    pool = mp.Pool(processes=1)
    def func(n):
        z = generateSample()
        l = likelihoodRatio(z,temp)
        return(z)
        
    print(pool.map(func, range(N)))
    
    pool.close()
    pool.join()
    # print(pool.join())
    
    return(1)
   
   
   


def f(n):
    temp = 1
    z = generateSample()
    l = likelihoodRatio(z,temp)
    return([z,l])
    
def importanceSampling_parallel2(N):
    
   
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
    
    temp = 3
    p = mp.Pool(8)
    # print('This is the parallel output')
    # print(p.map(f,range(N)))
    sample = p.map(f,range(N))
    
    # need to normalize  weights in the end
    
    return(sample)
    
def importanceSampling_parallel3(generateSample,args_generateSample,likelihoodRatio,args_likelihoodRatio, N):  
    # not working    
    temp = 3
    p = mp.Pool(4)
    print('This is the parallel output')
    sample = p.map(generateSample,np.array((10,)))
    print(sample)
    return(sample)
    
def importanceSampling_parallel4(generateSample,args_generateSample,likelihoodRatio,args_likelihoodRatio, N):  
    # not working
    temp = 3
    from IPython.parallel import Client
    p = Client()[:]
    p.use_dill()
    print('This is the parallel output')
    p.map_sync(generateSample,np.arange(6))
    #sample = p.map_sync(generateSample,np.array((10,)))
    #print(sample)
    return(1)
    
    
        
#     
# def likelihoodRatio(param, data, drift, diffusion, dt):
#    # dt
#    return(dt * drift(data,param) * drift(data))   + diffusion(data)*data.loc[] 
    
def define_l(data, drift, diffusion, dt):
    def l(param):
        data_diff = data.diff().iloc[1::] # makes start from time 2  
        # add the inversion of the diffusion matrices
        ratio = dt * drift(data.iloc[1::],param)*drift(data.iloc[1::],param)/2\
        - drift(data.iloc[1::],param)*np.array(data_diff)
        return(np.sum(ratio))
    return l
    
# Diffusion Bridge Sampling

def visualizeDiffusion(timeSeries):
    """This function takes a timeSeries in 1D,2D or 3D and displays the path
       I should allow to generate a whole sample of paths
    """
    dims = timeSeries.shape
    
    
    
    
    



# -------------------------------- Testing ---------------------------------------
class TestDiffusionMethods(unittest.TestCase):

    def test_ConstantDrift(self):
    # this function tests the performance with a constant drift and identity diffusion
    # estimating mean of iid Gaussians
    # fix tolerance level?
        def drift(x, param):
            # constant drift
            return(np.ones(x.shape)*param)
        def diffusion(x):
            # assume parameters are known
            # returns an np.array
            D = np.eye(x.shape[0])
            return(D)
            
        dt = 0.01
        T = 10
        theta = 4
        tol = 0.01
        x0 = np.zeros((3,))
        x = sampleDiffusion(x0,drift,[theta],diffusion,[],dt, T)
        l = define_l(x,drift,diffusion,dt)
        theta_hat = minimize(l,0,method = 'Nelder-Mead').x
        error = np.abs(theta_hat - theta_hat)
        # incorporate variance into tolerance      
        self.assertTrue(error < tol)
  
    def test_ConstantDrift_Gaussian(self):
        # discrete
        def drift(x, param):
            # constant drift
            return(np.ones(x.shape)*param)
        def diffusion(x):
            # assume parameters are known
            # returns an np.array
            D = np.eye(x.shape[0])
            return(D)
        dt = 0.01
        T = 10
        theta = 4
        tol = 1
        x0 = np.zeros((3,))
        x = sampleDiffusion(x0,drift,[theta],diffusion,[],dt, T)  
        l = define_l(x,drift,diffusion,dt)
        theta_hat1 = minimize(l,4,method = 'Nelder-Mead').x[0]
        z = np.random.normal(dt*theta, np.sqrt(dt),size = x.shape)
        theta_hat2 = np.mean(z)/dt # vectorize and add covariance      
        error = np.abs(theta_hat1 - theta_hat2)
        self.assertTrue(error<tol)
          
      # test for difference in the first and last position
      # test with a nonconstant drift
      # test with a non identity diffusion matrix
         
      # test sampleDiffusion with a Brownian motion
          
      # include a check for linear parameters and use least squares instead of general optimization
          
      
  
    
      
    def test_BrownianBridge(self):
        def drift(x,param):
            return (np.ones(x.shape)*param)
        def diffusion(x):
            D = np.eye(x.shape[0])
            return(D)
           
        dt = 0.01
        T = 1 # generalize
        theta = 4
        tol = 1
        # run many independently
        n = 10
        x0 = np.zeros((n,))
        x1 = np.zeros((n,)) # pin beginning and end to zero
       
        # generate x through the diffusion equation
        x_diffusion = sampleDiffusionBridge(x0,x1,drift, [0], diffusion,[],dt,T)
        # generate x directly through the distribution
        # 1) generate Brownian motion
        # w = random.normalvariate(np.zeros())
        w = sampleDiffusion(np.zeros((n,)), drift,[0], diffusion,[], dt,T)
    # 2)        
        index = np.array(w.index)
        x_distribution = w - np.reshape(index, (len(index),1)) * np.reshape(w.loc[T],(1,len(w.loc[T])))
       
        # What is a good way to compare two sets of paths
        # Or compare the paths to the distribution
       
        error = 0 # calculate a real measure
        self.assertTrue(error<tol)
      
       
    # Create functions which measure the goodness of samples in some way
       
    def test_ImportanceSampling_Gaussian(self):
        tol = 1  
        param = [4,1]
        
        def likelihoodRatio(x,param):
            # param corresponds to the parameters of the normal dist from which we want to sample
            l = stats.norm.pdf(x,param)/stats.norm.pdf(x)
            return(l)
        
        # using likelihoodRatio_Gaussian instead of likelihoodRatio saves 20s in testing
        sample = importanceSampling(generateSample,[],likelihoodRatio_Gaussian,[param],100000)
        mean_estimate = np.sum([sublist[0]*sublist[1] for sublist in sample])
        error = np.abs(mean_estimate - param[0])
        self.assertTrue(error < tol)   
      
    def test_ImportanceSampling_Exponential(self):
        tol = 1
        param = 1
        
        def likelihoodRatio(x,param):
            # param corresponds to the parameters of the exponential dist
            # l = stats.expon.pdf(x,param)/stats.norm.pdf(x)
            if x<0:
                l = 0
            else:
                l = param *np.exp(-x*param)/np.exp(-x**2/(2)) # faster than using stats...pdf functions
            return(l) 
        
        sample = importanceSampling(generateSample,[],likelihoodRatio,[param],100000)
        mean_estimate = np.sum([sublist[0]*sublist[1] for sublist in sample])
        error = np.abs(mean_estimate - 1)
        self.assertTrue(error < tol)   
 
      
if __name__ == '__main__':
    unittest.main()
    