# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:26:05 2020

@author: caioh
"""

import numpy as np
# from scipy.optimize import root
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#############Dados##################
q = .1 # m³/h
V = .1 # m³

k0     = 9703*3600 # L/h
deltaH = 5960      # kcal/kgmol
Ea     = 11843     # kcal/kgmol

Cp = 500   # kcal/m³/K
Ah = 15    # kcal/h/K
R  = 1.987 # kcal/kgmol/K

Tc = 298.5  # K
Tf = 298.15 # K
Cf = 10     # kgmol/m³

############ Sistema de EDO#########
def ODEsystem(x,t):
    C = x[0]
    T = x[1]
    
    dCdt = (q*Cf- q*C - C*V*k0*np.exp(-Ea/(R*T)))/V 
    dTdt = (q*Cp*(Tf-T) + (deltaH)*V*k0*np.exp(-Ea/(R*T))*C - Ah*(T-Tc))/(V*Cp)

    return dCdt, dTdt    

####condições iniciais#############
C0 = 5.68  
T0 = 337.7
x0 = (C0, T0)


tf = 20 
t  = np.linspace(0, tf, 1000)

x = odeint(ODEsystem, x0, t)


#####Plotando#######################
plt.subplot(2, 1, 1)
plt.plot(t, x[:,0])   
ax = plt.gca()
ax.set(xlim=(0, tf))
plt.ylabel('C (kgmol/m³)')
plt.xlabel('tempo ')

plt.subplot(2, 1, 2)
plt.plot(t, x[:,1]) 
ax = plt.gca()
ax.set(xlim=(0, tf))     
plt.ylabel('T (K)')
plt.xlabel('tempo ')





