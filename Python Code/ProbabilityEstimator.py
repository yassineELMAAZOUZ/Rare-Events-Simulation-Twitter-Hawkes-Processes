#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 11 14:15:21 2018
@authors: Yassine EL MAAZOUZ and Amine Bennouna
A package containing useful functions and procedures for the study of Hawkes' autoexcited process with function g(x) = alpha*exp(-beta*t)

"""

import numpy as np
import scipy.stats as stats
import Ogata_Base_Code as Oga
from numba import jit
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.close()


#########################################
######### Setting parameters:
#########################################

params1 = (1000, 1, 0.3, 1)


sigma1 = Oga.asymptoticStd(params1)
mu1 = Oga.asymptoticExpextedValue(params1)


#Choix du seuil a comme quantile de la gaussienne  qui approche la distribution.
seuil = stats.norm.ppf(1-3.1e-10,mu1,sigma1) 
                                                    

@jit
def probabilityEstimator(params,M,n,seuil):
    
    T,lambda0,alpha ,beta = params    
    
    lam0 = Oga.getLambda0(T,alpha,beta,seuil)
    
    params2 =(T, lam0,alpha ,beta)

    
    
    percent_ratio = M/100  #to track to progress of the estimation

    chgtdeProba_Estimations = []
    listForImportanceSampling=[]


        
    for j in range(n):
        
        print("Values number: ",j+1)
        
        estimatedValue = 0.
        for i in range(M):
            
            jumpTimes2     = Oga.simulateHawkesProcess(params2)
            numberOfjumps = len(jumpTimes2)
            likeliHood = Oga.likeliHood(params1,params2,jumpTimes2)
            estimatedValue      +=  (numberOfjumps>=seuil) * likeliHood /M
            listForImportanceSampling.append((numberOfjumps>=seuil)*likeliHood)
            k             =   int(i+1)/percent_ratio
            if(k%10==0):
                print("\t",k,"Percent done....")
        
        chgtdeProba_Estimations.append(estimatedValue)
        
    a = 5./100
    q = stats.norm.ppf(1-a/2.,0,1)
        
    sigma2 = np.std(listForImportanceSampling)
    mu2    = np.mean(listForImportanceSampling)
    relativeError2 = 2*q*sigma2/(np.sqrt(M*n)*mu2)

    
    print("MonteCarlo Importance Sampling: ")
    print("\tEstimated Values: ",mu2)
    print("\tIntervalle de confiance à 95%")
    print("\t\t[",mu2 -q*sigma2/np.sqrt(M*n)  ,";",mu2 +q*sigma2/np.sqrt(M*n)  ,"]")
    print("\tRelative Error: ",relativeError2)
    
    plt.boxplot([chgtdeProba_Estimations],labels = ["MonteCarlo par Importance Sampling"])


probabilityEstimator(params1,1000,100,seuil)
















