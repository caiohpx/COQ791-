# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:11:42 2021

@author: caioh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.integrate import odeint
import math as mt


# Resolva o sistema transformado para mostrar que o círculo unitário em plano (x1,x2) é um ciclo
# limite estável. Represente a solução graficamente. ###

######## analise dos valores caracteristicos#############
import scipy.linalg as la

# J = np.array([[-1, -1],
#               [1, -1]])

# results = la.eigvals(J)

# lamb1, lamb2 = results[0],  results[1] 

# # print(lamb1,lamb2)

# fig2 = plt.figure()  
# plt.plot(np.zeros(1000),np.linspace(-1.15,1.15,1000), color='darkgray',linewidth=.8)
# plt.plot(np.linspace(-2.7,2.7,1000),np.zeros(1000), color='darkgray',linewidth=.8)

# ax = plt.gca()
# ay = plt.gca()
# ay.set(ylim=(-1.15,1.15)) 
# ax.set(xlim=(-2.7,2.7))
# ax.text(.75, .2, 'Instabilidade', fontsize=12)               ##### escreve as viadagens
# ax.text(-2., .2, 'Estabilidade', fontsize=12)



# ax.text(lamb1.real-.1, lamb1.imag-.2, '${\lambda_{1}}$', fontsize=12)
# ax.text(lamb2.real-.1, lamb2.real+.1, '${\lambda_{2}}$', fontsize=12)

# plt.axvspan(-2.7, 0, -1, 1, alpha=0.3, color='lightgrey')   ##### coloca o fundo cinza nos 2º e 3º quadrantes



# plt.plot(lamb1.real,lamb1.imag, 'o', color='red')
# plt.plot(lamb2.real,lamb2.imag, 'o', color='blue')

# plt.ylabel('Im(${\lambda_{1,2}}$)')
# plt.xlabel('Re(${\lambda_{1,2}}$)')


# # plt.savefig('fig13.pdf', dpi=300)    ##<---------- salva a fig





######################################################################





def ODEsystem(rho,t):
    
    drhodt  = rho*((rho**2) - 1)
 
    return drhodt



cirho = np.array([.9,1,1.1])

tf = 30
M = 1000
t  = np.linspace(0, tf, M)

theta = t # integrando

fig2 = plt.figure()


r = odeint(ODEsystem, cirho[2], t) # resolvendo (muda aqui e muda na figura lá embaixo). não faria um condicional pra isso,
rho = r[:,0] 


x1, x2 = np.zeros(len(rho)), np.zeros(len(rho))



for i in range(len(rho)):
    
    x1[i] = rho[i]*mt.cos(theta[i])
    x2[i] = rho[i]*mt.sin(theta[i])


fig2 = plt.figure()


plt.subplot(121)

plt.plot(x1, x2, color='black')

ax = plt.gca()
ay = plt.gca()

# ax.set(xlim=(-.5,1))  ## somente pra 1ª fig
# ay.set(ylim=(-.2,1))

ax.set_xlabel('$x_{1}$')
ay.set_ylabel('$x_{2}$')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 

plt.subplot(122)

plt.plot(t, x1, label= '$x_{1}$')
plt.plot(t, x2, label= '$x_{2}$')

ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(0,.9)) ## somente pra 1ª fig 
ay.set(ylim=(0,10))

ax.set_xlabel('$t$')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 

plt.legend(loc='upper left', fontsize=9) 
    
plt.savefig('fig15.pdf', dpi=300)    ##<---------- salva a fig

