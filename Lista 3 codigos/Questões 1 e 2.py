# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:01:44 2021

@author: caioh
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#### Resolva o sistema de duas equações diferenciais abaixo             ####
####             dx1/dt = x2 ; dx2/dt = -x1 -mu*x2                      ####
#### para μ = -1, μ = 0, μ = 1. Quando o ponto de equilíbrio é estável, ####
#### assintoticamente estável e instável                                ####

mu=np.array([-1,0,1])

lamb1,lamb2 = np.zeros(3, dtype=complex), np.zeros(3, dtype=complex)

for i in range(len(mu)):
    J = np.array([[ 0,      1],                   #Jacobiana do sistema
                  [-1,  -mu[i]]])
    
    results = la.eigvals(J)                       #retorna os valores característicos
    
    lamb1[i], lamb2[i] = results[0],  results[1]  #guarda


#####PLOTANDO###################################################################
fig1 = plt.figure(1)  
plt.plot(np.zeros(1000),np.linspace(-1.15,1.15,1000), color='black',linewidth=.8)
plt.plot(np.linspace(-1,1,1000),np.zeros(1000), color='black',linewidth=.8)



plt.plot(lamb1[0].real,lamb1[0].imag, 'o', color='red', label='${\mu}=-1$')
plt.plot(lamb2[0].real,lamb2[0].imag, 'o', color='red')

plt.plot(lamb1[1].real,lamb1[1].imag, 'o', color='green', label='${\mu}=0$')
plt.plot(lamb2[1].real,lamb2[1].imag, 'o', color='green')

plt.plot(lamb1[2].real,lamb1[2].imag, 'o', color='blue', label='${\mu}=+1$')
plt.plot(lamb2[2].real,lamb2[2].imag, 'o', color='blue')


plt.axvspan(-1, 0, -1, 1, alpha=0.3, color='lightgrey')



ax = plt.gca()
ay = plt.gca()
ay.set(ylim=(-1.15,1.15)) 
ax.set(xlim=(-1,1))
ax.text(.27, .2, 'Instabilidade', fontsize=12)
ax.text(-.70, .2, 'Estabilidade', fontsize=12)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='best',
           ncol=3, mode="expand", borderaxespad=0.)

plt.ylabel('Im(${\lambda_{1,2}}$)')
plt.xlabel('Re(${\lambda_{1,2}}$)')

plt.savefig('fig1.pdf',dpi=300)  
plt.show()
#######################################################################################

#### 2-Discuta a estabilidade dos pontos de equilíbrio do sistema dinâmica #######
#### da Questão 1 para todos os valores de μ.                              ####### 

def ODEsystem(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu[0]
    dx1dt = x2
    dx2dt = -x1 - mi*x2
    
    return dx1dt, dx2dt

def ODEsystem2(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu[1]
    dx1dt = x2
    dx2dt = -x1 - mi*x2
    
    return dx1dt, dx2dt

def ODEsystem3(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu[2]
    dx1dt = x2
    dx2dt = -x1 - mi*x2
    
    return dx1dt, dx2dt

tf = 30
M = 100
t  = np.linspace(0, tf, M)

N = 100
x1guess = np.linspace(-100,100,N)   #coloquei guess, mas não é chute é CI
x2guess = np.linspace(-100,100,len(x1guess))

######PLOTANDO###############
######figura 2###############
p = odeint(ODEsystem, (x1guess[18],x2guess[18]), t)              #resolve pra uma primeira CI
fig2 = plt.subplot(1,2,1) 
plt.plot(p[:,0], p[:,1], color='black', linewidth=.9)            #plota a solução desse chute


plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=.9) #plota um vetor de CI
plt.plot(x1guess[18], x2guess[18], '*', color='black') 

for i in range(N):
    #####
    x = odeint(ODEsystem, (x1guess[i],x2guess[i]), t)    #resolve para o vetor CI
   
    plt.plot(x[:,0], x[:,1], color='red', linewidth=.2)
    
ax1 = plt.gca()
ay1 = plt.gca()
ay1.set(ylim=(-150,150)) 
ax1.set(xlim=(-150,150)) 
plt.ylabel('$x_{2}$')
plt.xlabel('$x_{1}$')
    



fig2 = plt.subplot(1,2,2)   
plt.plot(t, p[:,0], color ='black', label= '$x_{1}$' ) 
plt.plot(t, p[:,1], color ='gray', label= '$x_{2}$')

plt.legend(loc='lower left', fontsize=9) 
      
ax2 = plt.gca()
ay2 = plt.gca()
ax2.set(xlim=(0,30)) 

plt.xlabel('$t$')

plt.savefig('fig2.pdf', dpi=300)  


######figura 3###############
fig3 = plt.figure()
fig3 = plt.subplot(1,2,1) 
p = odeint(ODEsystem2, (x1guess[18],x2guess[18]), t)             #resolve pra uma primeira CI
plt.plot(p[:,0], p[:,1], color='black', linewidth=2)            #plota a solução desse chute


plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=2) #plota um vetor de CI
plt.plot(x1guess[18], x2guess[18], '*', color='black') 

for i in range(N):
    #####
    x = odeint(ODEsystem2, (x1guess[i],x2guess[i]), t)    #resolve para o vetor CI
   
    plt.plot(x[:,0], x[:,1], color='green', linewidth=.2)
    
ax1 = plt.gca()
ay1 = plt.gca()
ay1.set(ylim=(-150,150)) 
ax1.set(xlim=(-150,150)) 
plt.ylabel('$x_{2}$')
plt.xlabel('$x_{1}$')
    



fig3 = plt.subplot(1,2,2)   
plt.plot(t, p[:,0], color ='black', label= '$x_{1}$' ) 
plt.plot(t, p[:,1], color ='gray', label= '$x_{2}$')

plt.legend(loc='upper right', fontsize=9) 
      
ax2 = plt.gca()
ay2 = plt.gca()
ax2.set(xlim=(0,30)) 
ay2.set(ylim=(-150,150)) 
plt.xlabel('$t$')
plt.savefig('fig3.pdf', dpi=300)  

######figura 4###############
fig4 = plt.figure()
p = odeint(ODEsystem3, (x1guess[18],x2guess[18]), t)              #resolve pra uma primeira CI
fig4 = plt.subplot(1,2,1) 
plt.plot(p[:,0], p[:,1], color='black', linewidth=.9)            #plota a solução desse chute


plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=.9) #plota um vetor de CI
plt.plot(x1guess[18], x2guess[18], '*', color='black') 

for i in range(N):
    #####
    x = odeint(ODEsystem3, (x1guess[i],x2guess[i]), t)    #resolve para o vetor CI
   
    plt.plot(x[:,0], x[:,1], color='blue', linewidth=.2)
    
ax1 = plt.gca()
ay1 = plt.gca()
ay1.set(ylim=(-150,150)) 
ax1.set(xlim=(-150,150)) 
plt.ylabel('$x_{2}$')
plt.xlabel('$x_{1}$')
    



fig2 = plt.subplot(1,2,2)   
plt.plot(t, p[:,0], color ='black', label= '$x_{1}$' ) 
plt.plot(t, p[:,1], color ='gray', label= '$x_{2}$')

plt.legend(loc='upper right', fontsize=9) 
      
ax2 = plt.gca()
ay2 = plt.gca()
ax2.set(xlim=(0,30)) 
ay2.set(ylim=(-150,150)) 
plt.xlabel('$t$')

plt.savefig('fig4.pdf', dpi=300)  





    


    
    
    ##### 
    # x = odeint(ODEsystem2, (x1guess[i],x2guess[i]), t)
    # 
    
    # fig3 = plt.figure(3) 
    # plt.plot(x[:,0], x[:,1], color='green', linewidth=.2)
    # plt.plot(p[:,0], p[:,1], color='black', linewidth=.2) 
    
    # plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=.2)
    
    # ax = plt.gca()
    # ay = plt.gca()
    
    # ay.set(ylim=(-150,150)) 
    # ax.set(xlim=(-150,150)) 
    # plt.ylabel('$x_{2}$')
    # plt.xlabel('$x_{1}$')
    
    # #####
    # x = odeint(ODEsystem3, (x1guess[i],x2guess[i]), t)
    # p = odeint(ODEsystem3, (x1guess[18],x2guess[18]), t)
    
    # fig4 = plt.figure(4)  
    # plt.plot(x[:,0], x[:,1], color='blue', linewidth=.2)
    # plt.plot(p[:,0], p[:,1], color='black', linewidth=.2) 
    # plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=.2)
    # ax = plt.gca()
    # ay = plt.gca()
    # ay.set(ylim=(-150,150)) 
    # ax.set(xlim=(-150,150)) 
    # plt.ylabel('$x_{2}$')
    # plt.xlabel('$x_{1}$')
    

plt.show()