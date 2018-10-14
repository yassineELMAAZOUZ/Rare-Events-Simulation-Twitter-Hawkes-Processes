#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 11 14:15:21 2018
@authors: Yassine EL MAAZOUZ and Amine Bennouna
A package containing useful functions and procedures for the study of Hawkes' autoexcited process with function g(x) = alpha*exp(-beta*t)

"""

################# Avec auto-excitation ###############
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import Ogata_Base_Code as Oga
plt.close()
plt.figure(2)


##### Paramètres fixés.
params1 = (100, 1, 0.3, 1)
T,lambda0_1,alpha_1,beta_1 = params1

M     = 1000
Nlist = np.zeros(M)
for i in range(M):
    
    jumpTimes     = Oga.simulateHawkesProcess(params1)  
    numberOfjumps = len(jumpTimes)
    Nlist[i]      =  numberOfjumps
    
    

mu1    = Oga.asymptoticExpextedValue(params1)
sigma1 = Oga.asymptoticStd(params1)

start1 = mu1 - 5*sigma1
end1   = mu1 + 5*sigma1

xLine = np.arange(start1,end1,1)
Gauss = stats.norm.pdf(np.arange(start1,end1,1),mu1,sigma1)

plt.hist(Nlist,bins = np.arange(start1,end1,T*0.1),histtype = "step" ,color ="blue",normed = 1,label="Parameters1: "+str(params1))

plt.plot(xLine,Gauss,color = "blue",label="Parameters1: "+str(params1))
plt.legend()
plt.xlabel("Nombre total des sauts")
plt.ylabel("Proportion de réalisation")



##### Paramètres sur lesquels nous allons jouer.
sigma1 = Oga.asymptoticStd(params1)
mu1 = Oga.asymptoticExpextedValue(params1)

seuil    = stats.norm.ppf(1-1e-1,mu1,sigma1)
lam0 = Oga.getLambda0(T,alpha_1,beta_1,seuil)


#################################################################
##### Choice of the right parameters
#################################################################


############ changement de lambda_0.
params2 =(T, lam0,alpha_1 ,beta_1)



Nlist = np.zeros(M)
for i in range(M):
    
    jumpTimes     = Oga.simulateHawkesProcess(params2)
    numberOfjumps = len(jumpTimes)
    Nlist[i]      =  numberOfjumps

mu2    = Oga.asymptoticExpextedValue(params2)
sigma2 = Oga.asymptoticStd(params2)

start2 = mu2 - 5*sigma2
end2   = mu2 + 5*sigma2

xLine = np.arange(start2,end2,1)
Gauss = stats.norm.pdf(np.arange(start2,end2,1),mu2,sigma2)



plt.hist(Nlist,bins = np.arange(start2,end2,T*0.1),histtype = "step" ,color = "green" ,normed = 1,label="Parameters2: "+str(params2))

plt.plot(xLine,Gauss,color = "orange",label="Parameters2: "+str(params2))
plt.legend()

plt.xlabel("Nombre de sauts")
plt.ylabel("Nombre de realisations")
plt.title("Histogrames pour différents paramètres")

