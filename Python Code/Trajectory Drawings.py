#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 12:42:29 2018

@author: Yassine EL MAAZOUZ
    
"""
################# Avec auto-excitation ###############
import numpy as np
import matplotlib.pyplot as plt
import Ogata_Base_Code as Oga


##### Paramètres fixés.
params1 = (100,1,0.5,1)
T,lambda0_1,alpha_1,beta_1 = params1

##### Paramètres sur lesquels nous allons jouer.
params2 = (T,1.1,2.9,3)
T,lambda0_2,alpha_2,beta_2 = params2


M = 10
for i in range(M):
    
    plt.subplot(1,2,1)
    jumpTimes1 = np.concatenate([np.zeros(1),Oga.simulateHawkesProcess(params1)])

    plt.step(jumpTimes1,np.arange(len(jumpTimes1)))
    plt.title("Parameter theta_1")
    plt.xlabel("timeLine")
    plt.ylabel("N_t")
    
    
    plt.subplot(1,2,2)
    jumpTimes2 = np.concatenate([np.zeros(1),Oga.simulateHawkesProcess(params2)])
    plt.title("Parameter theta_2")
    plt.step(jumpTimes2,np.arange(len(jumpTimes2))   )
    plt.xlabel("timeLine")
    plt.ylabel("N_t")

plt.show()





