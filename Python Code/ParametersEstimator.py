#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 11 14:15:21 2018
@authors: Yassine EL MAAZOUZ and Amine Bennouna
A package containing useful functions and procedures for the study of Hawkes' autoexcited process with function g(x) = alpha*exp(-beta*t)

"""

import numpy as np
import Ogata_Base_Code as Oga
import scipy.optimize as opt
from numba import jit


@jit(nopython = True)
def UCalculator(lambda0,alpha,beta):
    UValues = [0]
    v=0
    u=0
    
    for i in range(len(jumpTimes)-1):
        delta = jumpTimes[i+1]-jumpTimes[i]
        v = np.exp(-beta*delta)*(v+1)
        u = u*np.exp(-beta*delta) + delta * v
        UValues.append(u)
    
    return np.array(UValues)

@jit(nopython = True)
def GradLogLikelihood(x):
    
    
    lambda0,alpha,beta = x[0],x[1],x[2]
    
    param0 = T, lambda0, alpha, beta

    Intensities =  Oga.intensityCalculator(param0,jumpTimes)
    InvIntensities = 1/Intensities
    UValues = UCalculator(lambda0,alpha,beta)

    Grad_lambda0 = np.sum(InvIntensities) - T
    
    Grad_alpha = 1/alpha * np.sum(InvIntensities*(Intensities-lambda0)) - 1/beta *np.sum(1-np.exp(-beta*(T-jumpTimes)))
    
    Grad_beta = -alpha*np.sum(InvIntensities*UValues) - alpha/beta * np.sum(-1/beta * (1-np.exp(-beta*(T-jumpTimes))) + (T-jumpTimes)*np.exp(-beta*(T-jumpTimes)))


    return np.array([Grad_lambda0, Grad_alpha, Grad_beta])
    

@jit(nopython = True)
def logLikelihood0(x):
    
    """  
 		Input : params  = (T,lambda0,alpha,beta)
 				 
 						   T      : Time Horizon of simulation.
 						   lambda0: Base intensity.
 						   alpha  : the stretch parameter of function g.
 						   beta   : the decaying parameter of function g
 
 						   "g(x) = alpha*exp(-beta*x)"
 				
 				jumpTimes: The jump Times of a Hawkes process 
 
 
 		Output:  The logLikelihood of the given Hawkes with respect to a standard Poisson process
   
    """
        
    
    lambda0, alpha, beta = x[0],x[1],x[2]
    params0 = T, lambda0, alpha, beta
    
    
    intensityValues =  Oga.intensityCalculator(params0,jumpTimes)
    
    firstTerm  = np.sum(np.log(intensityValues))
    
    secondTerm = np.sum(np.exp(-beta*(T-jumpTimes)) - 1)
    
    return firstTerm +  T*(1-lambda0)  + (alpha/beta)*secondTerm
    




params = (1000, 1.2, 0.6, 0.8)
T,lambda0,alpha,beta = params
x = [lambda0,alpha,beta]




def f(x):
    return -logLikelihood0(x)
def grad(x):
    return -GradLogLikelihood(x)


result = np.zeros(3)


M = 100
percent_ratio=M/100

for i in range(M):
    
    jumpTimes = Oga.simulateHawkesProcess(params)
    T = jumpTimes[-1]    
    result = result + opt.fmin_tnc(f, x, fprime = grad, approx_grad = False)[0]/M

result = np.concatenate([np.array([params[0]]),result])

print("Estimated Parameters:  ",result)
print("\n\n")
print("Real Parameters      :  ",params)

