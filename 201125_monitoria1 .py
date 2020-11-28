# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:58:05 2020

@author: caioh
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import odeint
import matplotlib.pyplot as plt

k1 = 6 # L/mol.h
k2 = 3 # h-¹

P  = 73.80 # atm
V0 = 1.0   # L
T  = 300   # K
R  = 0.082 # atm.L/mol.K


na0 = 1 # mol
nb0 = 2 # mol
nc0 = 0 # mol

csia = -1/3
M    = nb0/na0

##### conversão e volume final no estado estecionário############
def F(fa):
    return ((+k1*(1-fa)*(M-fa))/(V0*(1+csia*fa)))-k2*fa

if __name__ == "__main__":

    faguess  = .7
    sol1 = root(F, faguess)
    # print(f"fa = {sol1.x[0]:5.5e}")
    # print(f"res = {sol1.fun[0]:5.5e}")
    
# V = V0*(1+csia*sol1.x[0])
# na = na0*(1-sol1.x[0])
# nb = na0*(M-sol1.x[0])
# nc = na0*sol1.x[0]

# print(f"V = {V:5.5e}")
# print(f'na, nb, nc, total = {na, nb, nc, na+nb+nc} mol ')



############ transiente ############
def diff(fa, t):
    
    dfadt = ((+k1*(1-fa)*(M-fa))/(V0*(1+csia*fa)))-k2*fa
    return dfadt


t = np.linspace(0, 1)
fa0 = 0

fa = odeint(diff, fa0 , t  )


V = V0*(1+csia*fa)
# print(V)
na = na0*(1-fa)
nb = na0*(M-fa)
nc = na0*fa

print(na+nb+nc)


plt.subplot(2, 2, 1)
plt.plot(t,  fa, label='')      
plt.ylabel('conversão (%)')
plt.xlabel('tempo (h)')
# plt.legend(loc='upper right', fontsize=9)

plt.subplot(2, 2, 2)
plt.plot(t, na, label='na')
plt.plot(t, nb, label='nb')
plt.plot(t, nc, label='nc')
plt.ylabel('número de mols (mol)')
plt.xlabel('tempo (h)')
plt.legend(loc='upper right', fontsize=7)

plt.subplot(2, 2, 3)
plt.plot(t, V, label='')
plt.ylabel('Volume (L)')
plt.xlabel('tempo (h)')
# plt.legend(loc='lower right', fontsize=9)






