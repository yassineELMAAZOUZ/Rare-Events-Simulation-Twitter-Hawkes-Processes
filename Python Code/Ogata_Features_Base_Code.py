#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 11 14:15:21 2018
@authors: Yassine EL MAAZOUZ and Amine Bennouna
A package containing useful functions and procedures for the study of Hawkes' autoexcited process with function g(x) = alpha*exp(-beta*t)

"""

########################################
##### Modules to import:
########################################
import numpy as np
import numpy.random as npr
from numba import jit

#########################################
##### Functions:
########################################

@jit
def generateFeatures(featuresProba):
    
    """
        Input : featuresProba, a list of probabilities for the features.
        
        Output: A feature (or coefficient_index) is randomly chosen with the given probability and returned as an output.
        
    """
    n = len(featuresProba)
    return npr.choice(a=np.arange(n),p=featuresProba)


@jit
def simulateHawkesProcessWithFeature(params , featuresProba , featureCoeffs ):
    """  
		Input :  params = (T,lambda0,alpha,beta)
				 
							T      : Time Horizon of simulation.
							lambda0: Base intensity.
							alpha  : the stretch parameter of function g.
							beta   : the decaying parameter of function g

							"g(x) = alpha*exp(-beta*x)"
	
		Output:  jumpTimes: an array containing the jump times of Hawkes process
    """
    T,lambda0,alpha,beta = params
    currentInstant = 0            # the current generated instant.
    numberOfInstants = 0          # the current number of jumps.
    lambdaUpperBound = lambda0    # the current upper bound for lambda_t for thining algorithm
    jumpTimes = []                # a list of accepted instants (jumpTimes)
    generatedCoeffs = []          # a list of generated feature coefficients
    
    while(currentInstant<T):

        jumpGapWidth = npr.exponential(1/lambdaUpperBound)
        currentInstant += jumpGapWidth
        D = npr.uniform(0,1)
        intensityOfNewPoint = lambda0 + np.exp(-beta*jumpGapWidth)*(lambdaUpperBound-lambda0)

        if(lambdaUpperBound*D<=intensityOfNewPoint):
            
            gamma = featureCoeffs[generateFeatures(featuresProba)]
            
            lambdaUpperBound = intensityOfNewPoint + alpha*gamma    
            numberOfInstants+=1
            jumpTimes.append(currentInstant)
            
            generatedCoeffs.append(gamma)

        else:
            
            lambdaUpperBound = intensityOfNewPoint
  
    if(jumpTimes[-1]>T):
        jumpTimes.pop()
        generatedCoeffs.pop()

    	
    return (np.array(jumpTimes),np.array(generatedCoeffs))





@jit(nopython = True)
def intensityCalculatorWithFeatures(params,jumpTimes,generatedCoeffs):
    """  
		Input : params  = (T,lambda0,alpha,beta)
				 
						   T      : Time Horizon of simulation.
						   lambda0: Base intensity.
						   alpha  : the stretch parameter of function g.
						   beta   : the decaying parameter of function g.

						   "g(x) = alpha*exp(-beta*x)"
				
				jumpTimes: The jump Times of a Hawkes process 


		Output:  intensityValues: an array containing the intensities with which each points was generated
	   
	"""
    
    T,lambda0,alpha,beta = params
    
    Mu = [0]
    N_T = len(jumpTimes)
    for i in range(N_T-1):   
        
        Mu.append(np.exp(-beta*(jumpTimes[i+1]-jumpTimes[i])) * (Mu[-1] + alpha ))

    return (lambda0 + np.array(Mu))



@jit(nopython = True)
def logLikelihood(params,jumpTimes,generatedCoeffs):
    
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
    
    T,lambda0,alpha,beta = params
    
    intensityValues = intensityCalculatorWithFeatures(params,jumpTimes,generatedCoeffs)
    
    firstTerm  = np.sum(np.log(intensityValues))
    
    secondTerm = np.sum(  (np.exp(-beta*(T-jumpTimes)) - 1 ) )
    
    return firstTerm +  T*(1-lambda0)  + (alpha/beta)*secondTerm





@jit
def featureProbaLikelyHood(featureProba1,featureProba2,generatedCoeffs,featureCoeffs):
    
    NT = len(generatedCoeffs)
    n  = len(featureCoeffs)
    
    Matrice    = (featureProba1/featureProba2).reshape(n,1) * np.ones((1,NT))
    IndexMatrix = featureCoeffs.reshape(n,1) == generatedCoeffs.reshape(1,NT)
    
    return np.prod(np.prod(Matrice**IndexMatrix))
    
    




@jit
def likeliHood(params1,featureProba1,params2,featureProba2,featureCoeffs,jumpTimes,generatedCoeffs):
    """  
		Input : params  = (T,lambda0,alpha,beta)
				 
						   T      : Time Horizon of simulation.
						   lambda0: Base intensity.
						   alpha  : the stretch parameter of function g.
						   beta   : the decaying parameter of function g

						   "g(x) = alpha*exp(-beta*x)"
				
				jumpTimes: The jump Times of a Hawkes process 


		Output:  The logLikelihood of the first Hawkes with respect to the second one. (dP_1/dP_2)
	   
	 """
     
    logdP1_P0 = logLikelihood(params1,jumpTimes,generatedCoeffs)
    logdP2_P0 = logLikelihood(params2,jumpTimes,generatedCoeffs)
    
    dP1_P2 = np.exp(logdP1_P0-logdP2_P0)
    
    dpi1_pi2 = featureProbaLikelyHood(featureProba1,featureProba2,generatedCoeffs,featureCoeffs)
    
    return     dP1_P2*dpi1_pi2





@jit(nopython = True)      
def asymptoticExpextedValue(params,featureProbas,featureCoeffs):
    """  
		Input : params  = (T,lambda0,alpha,beta)
				 
						   T      : Time Horizon of simulation.
						   lambda0: Base intensity.
						   alpha  : the stretch parameter of function g.
						   beta   : the decaying parameter of function g

						   "g(x) = alpha*exp(-beta*x)"
				


		Output:  asymptotic expected value for the given parameters.
	   
	 """
    T,lambda0,alpha,beta = params
    
    if(alpha>=beta):
        raise Exception("alpha>beta:  +inf, undefined value")
    
    return (lambda0*T) /(1- ((alpha/beta)*np.sum(featureProbas*featureCoeffs)) )
 
    
@jit(nopython = True)
def asymptoticStd(params,featureProbas,featureCoeffs):
    """  
		Input : params  = (T,lambda0,alpha,beta)
				 
						   T      : Time Horizon of simulation.
						   lambda0: Base intensity.
						   alpha  : the stretch parameter of function g.
						   beta   : the decaying parameter of function g

						   "g(x) = alpha*exp(-beta*x)"
				


		Output:  asymptotic expected value for the given parameters.
	   
	 """
    T,lambda0,alpha,beta = params

    if(alpha>=beta):

    	raise Exception("alpha>beta:  +inf, undefined value")

    return  np.sqrt((lambda0*T) / ((1 - (alpha/beta)*np.sum(featureProbas*featureCoeffs))**3))




    



@jit(nopython = True)
def getParameters(T,beta,expectedValue,std):
    """
        Input :  expectedValue,var, T, beta
    
        Output:  parameters
    
    """
    var = std**2    
    return (T, expectedValue/T * np.sqrt(expectedValue/var), 1-np.sqrt(expectedValue/var) * beta , beta)


@jit(nopython = True)
def getLambda0(T,alpha,beta,featureProba,featureCoeffs,expectedValue):

    return (expectedValue/T)*(1-((alpha/beta)*np.sum(featureProba*featureCoeffs)))
















