# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:13:09 2021

@author: caioh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math as mt


#Resolva o sistema transformado para mostrar que o círculo unitário em plano (x1,x2) é um ciclo
#limite estável. Represente a solução graficamente. ###

def ODEsystem(x,t):
    
    rho   = x[0]
    theta = x[1]
    
    drhodt  = 1 - rho**2
    dthetadt  = 1 
    
    return drhodt, dthetadt

#### array de CI ##########



tf = 10
M = 1000
t  = np.linspace(0, tf, M)

cix, ciy = 4, 4   #condições iniciais
r = odeint(ODEsystem, (cix,ciy), t) # resolvendo

# plt.subplot(131)
# plt.plot(r[:,1], r[:,0], color='black')

# ax = plt.gca()
# ay = plt.gca()
# ax.set(xlim=(0,10)) 
# ay.set(ylim=(0,10))

# ax.set_xlabel(r'$\theta$(t)')
# ay.set_ylabel(r'$\rho$(t)')

plt.subplot(122)
plt.plot(t,r[:,0],'--', color='blue', label=r'$\rho$(t)')
plt.plot(t,r[:,1], color='blue', label=r'$\theta$(t)')

plt.legend(loc='uper left', fontsize=9) 

ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(-.2,10)) 
ay.set(ylim=(-.2,10))

plt.xlabel('t')

################# ainda primeira figura (figura da esquerda) ###########
N=25               # numero de pontos do vetor CI
ci = np.linspace(0,10,N) #condições iniciais
plt.subplot(121)

for i in range(len(ci)):
    
    g = odeint(ODEsystem, (ci[i],ci[i]), t) # resolvendo

    plt.plot(g[:,1], g[:,0],'-', color='blue', linewidth=.3)

plt.plot(r[:,1], r[:,0], color='black', linewidth=1)

plt.plot(r[0,1],r[0,0], '*', color='black')

ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(-.2,10)) 
ay.set(ylim=(-.2,10))

ax.set_xlabel(r'$\theta$(t)')
ay.set_ylabel(r'$\rho$(t)')


# plt.savefig('fig112.pdf', dpi=300)    ##<---------- salva a fig



################## retornando as variaveis originais#############
# theta, rho = np.zeros(len(r[:,1])), np.zeros(len(r[:,1]))

theta, rho = r[:,1], r[:,0]

x1 = np.zeros(len(theta))
x2 = np.zeros(len(theta))

for i in range(len(theta)):
    
    x1[i] = rho[i]*mt.cos(theta[i])
    
    x2[i] = rho[i]*mt.sin(theta[i])
    

# plt.subplot(131)


fig2 = plt.figure()

plt.subplot(122)
plt.plot(t,x1,'--', color='darkorange', label= '$x_{1}$')
plt.plot(t,x2,color='darkorange', label= '$x_{2}$')

ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(0,10)) 
ay.set(ylim=(-1.1,1.1))


plt.xlabel('t')

plt.legend(loc='lower right', fontsize=9) 



################# ainda segunda figura (figura da esquerda) ###########



x1 = np.zeros(len(theta))
x2 = np.zeros(len(theta))


plt.subplot(121)
for i in range(len(ci)):
    
    g = odeint(ODEsystem, (ci[i],ci[i]), t) # resolvendo
    
    theta, rho = g[:,1], g[:,0]
    
    for i in range(len(theta)):
    
        x1[i] = rho[i]*mt.cos(theta[i])
        
        x2[i] = rho[i]*mt.sin(theta[i])
        
    
    plt.plot(x1, x2,'-', color='darkorange', linewidth=.4)

plt.plot(x1,x2, color='black', linewidth=.8)
plt.plot(x1[0],x2[0], '*', color='black')



ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(-1.1,1.1)) 
ay.set(ylim=(-1.1,1.1))


    
plt.ylabel('$x_{2}$')
plt.xlabel('$x_{1}$')
# plt.savefig('fig12.pdf', dpi=300)    ##<---------- salva a fig

########### analise dos valores caracteristicos#############
# import scipy.linalg as la

# J = np.array([[-2, 0],
#               [0, 0]])

# results = la.eigvals(J)

# lamb1, lamb2 = results[0],  results[1] 

# fig2 = plt.figure()  
# plt.plot(np.zeros(1000),np.linspace(-1.15,1.15,1000), color='darkgray',linewidth=.8)
# plt.plot(np.linspace(-2.7,2.7,1000),np.zeros(1000), color='darkgray',linewidth=.8)

# ax = plt.gca()
# ay = plt.gca()
# ay.set(ylim=(-1.15,1.15)) 
# ax.set(xlim=(-2.7,2.7))
# ax.text(.75, .2, 'Instabilidade', fontsize=12)               ##### escreve as viadagens
# ax.text(-2., .2, 'Estabilidade', fontsize=12)

# plt.axvspan(-2.7, 0, -1, 1, alpha=0.3, color='lightgrey')   ##### coloca o fundo cinza nos 2º e 3º quadrantes



# plt.plot(lamb1.real,lamb1.imag, 'o', color='red')
# plt.plot(lamb2.real,lamb2.imag, 'o', color='blue')

# plt.ylabel('Im(${\lambda_{1,2}}$)')
# plt.xlabel('Re(${\lambda_{1,2}}$)')
