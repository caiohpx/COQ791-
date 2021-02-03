# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:09:30 2021

@author: caioh
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import odeint

K=10000
mu=np.linspace(-3,3,K)         # um vetor grandão
mu_p=np.array([-3,-1,0,1,3])   # poucos pontinhos, importantes para discussão


ReLamb1, ImLamb1, ReLamb2, ImLamb2 = np.zeros(len(mu)), np.zeros(len(mu)), np.zeros(len(mu)), np.zeros(len(mu)) #para os vetores
lamb1,lamb2 = np.zeros(len(mu), dtype=complex), np.zeros(len(mu), dtype=complex) #para os pontos

for i in range(len(mu)):
    
    J = np.array([[0,  1],
                  [-1,mu[i]]])
    
    results = la.eigvals(J)                       #retorna os valores característicos
    
    ReLamb1[i], ImLamb1[i], ReLamb2[i], ImLamb2[i]  = results[0].real,  results[0].imag, results[1].real,  results[1].imag  #guarda


for i in range(len(mu_p)):                        #certamente tem uma maneira mais inteligente, manipulando os arrays. fiquei com preguiça de procurar, então fiz um condicional só para os pontos
    
    J = np.array([[0,  1],
                  [-1,mu_p[i]]])
    
    results = la.eigvals(J)                       #retorna os valores característicos
    
    lamb1[i], lamb2[i] = results[0],  results[1]  #guarda

####PLOTANDO###################################################################
fig1 = plt.figure(1)  
plt.plot(np.zeros(1000),np.linspace(-1.15,1.15,1000), color='darkgray',linewidth=.8)
plt.plot(np.linspace(-2.7,2.7,1000),np.zeros(1000), color='darkgray',linewidth=.8)

#####para os vetores###
plt.plot(ReLamb1,ImLamb1, color='blue',linewidth=.8)
plt.plot(ReLamb2,ImLamb2, color='red',linewidth=.8)
#######################

####para os pontos#####
plt.plot(lamb1[0].real,lamb1[0].imag, 'o', color='cyan', label='${\mu}=-3$')
plt.plot(lamb2[0].real,lamb2[0].imag, 'o', color='cyan')

plt.plot(lamb1[1].real,lamb1[1].imag, 'o', color='darkblue', label='${\mu}=-1$')
plt.plot(lamb2[1].real,lamb2[1].imag, 'o', color='darkblue')

plt.plot(lamb1[2].real,lamb1[2].imag, 'o', color='green', label='${\mu}=0$')
plt.plot(lamb2[2].real,lamb2[2].imag, 'o', color='green')

plt.plot(lamb1[3].real,lamb1[3].imag, 'o', color='darkorange', label='${\mu}=1$')
plt.plot(lamb2[3].real,lamb2[3].imag, 'o', color='darkorange')

plt.plot(lamb1[4].real,lamb1[4].imag, 'o', color='purple', label='${\mu}=3$')
plt.plot(lamb2[4].real,lamb2[4].imag, 'o', color='purple')
#######################



plt.axvspan(-2.7, 0, -1, 1, alpha=0.3, color='lightgrey')   ##### coloca o fundo cinza nos 2º e 3º quadrantes

ax = plt.gca()
ay = plt.gca()
ay.set(ylim=(-1.15,1.15)) 
ax.set(xlim=(-2.7,2.7))
ax.text(.75, .2, 'Instabilidade', fontsize=12)               ##### escreve as viadagens
ax.text(-2., .2, 'Estabilidade', fontsize=12)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='best',
            ncol=len(mu_p), mode="expand", borderaxespad=0.) ####coloca a legendinha bonita

plt.ylabel('Im(${\lambda_{1,2}}$)')
plt.xlabel('Re(${\lambda_{1,2}}$)')

# plt.savefig('fig5.pdf', dpi=300)  
plt.show()
#######################################################################################

def ODEsystem1(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu_p[0]
    dx1dt = x2
    dx2dt = -x1 + mi*(1-(x1**2))*x2
    
    return dx1dt, dx2dt

def ODEsystem2(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu_p[1]
    dx1dt = x2
    dx2dt = -x1 + mi*(1-(x1**2))*x2
    
    return dx1dt, dx2dt

def ODEsystem3(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu_p[2]
    dx1dt = x2
    dx2dt = -x1 + mi*(1-(x1**2))*x2
    
    return dx1dt, dx2dt
    
def ODEsystem4(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu_p[3]
    dx1dt = x2
    dx2dt = -x1 + mi*(1-(x1**2))*x2
    
    return dx1dt, dx2dt

def ODEsystem5(x,t):
    x1 = x[0]
    x2 = x[1]
    mi = mu_p[4]
    dx1dt = x2
    dx2dt = -x1 + mi*(1-(x1**2))*x2
    
    return dx1dt, dx2dt

tf = 30
M = 1000
t  = np.linspace(0, tf, M)

N = 100
x1guess = np.linspace(-2.5,2.5,N)                          #coloquei guess, mas não é chute é CI
x2guess = np.linspace(-3,3,len(x1guess))


######PLOTANDO############### (não fiz todas as figuras pra não forçar) mude na linha indicada com a seta <--------------
####figura 2###############
fig2 = plt.figure()
fig2 = plt.subplot(1,2,1)
u=56                                                       # anda com o vetor chutes, a sua escolha, ele é o ponto do vetor CI que vai aparecer no gráfico como: *


for i in range(N):
    #####
    x = odeint(ODEsystem1, (x1guess[i],x2guess[i]), t)   #<---------- ecolha a ODE . ODEsystemi. resolve para o vetor CI  
   
    plt.plot(x[:,0], x[:,1], color='cyan', linewidth=.2) #<---------- ecolha a cor 
    
plt.plot(x1guess, x2guess, '-', color='lightgray', linewidth=.9) #plota um vetor de CI
plt.plot(x1guess[u], x2guess[u], '*', color='black') 

p = odeint(ODEsystem1, (x1guess[u],x2guess[u]), t)         ##<---------- ecolha a ODE . ODEsystemi. resolve para o vetor CI  resolve pra uma primeira CI
plt.plot(p[:,0], p[:,1], color='black', linewidth=.9)      #plota a solução dessa CI espessifica

ax1 = plt.gca()
ay1 = plt.gca()
ax1.set(xlim=(-3,3)) 
ay1.set(ylim=(-5,5)) 
 
plt.ylabel('$x_{2}$')
plt.xlabel('$x_{1}$')



fig2 = plt.subplot(1,2,2)  #plot da direita
plt.plot(t, p[:,0], color ='black', label= '$x_{1}$' ) 
plt.plot(t, p[:,1], color ='gray', label= '$x_{2}$')

plt.legend(loc='lower right', fontsize=9) 
      
ax2 = plt.gca()
ay2 = plt.gca()
ay2.set(ylim=(-5,5)) 
ax2.set(xlim=(0,30)) 

plt.xlabel('$t$')
# plt.savefig('fig6.pdf', dpi=300)    ##<---------- salva a fig
plt.show()



