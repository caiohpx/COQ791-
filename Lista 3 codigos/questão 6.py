# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:02:14 2021

@author: caioh
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import odeint

J = np.array([[ 0,  1],                   #Jacobiana do sistema (0,0)
              [1,  -1]])

results = la.eigvals(J)                       #retorna os valores característicos

lamb1, lamb2 = results[0],  results[1]  #guarda

J2 = np.array([[ 0,  1],                   #Jacobiana do sistema (+-1,0)
              [-2,  -1]])

results = la.eigvals(J2)                       #retorna os valores característicos

lamb3, lamb4 = results[0],  results[1]  #guarda


# #####PLOTANDO###################################################################
fig1 = plt.figure(1)  
plt.plot(np.zeros(1000),np.linspace(-1.55,1.55,1000), color='black',linewidth=.8)
plt.plot(np.linspace(-1.7,1.7,1000),np.zeros(1000), color='black',linewidth=.8)



plt.plot(lamb1.real,lamb1.imag, 'o', color='red', label='$({x_{1}^{*},x_{2}^{*})}=(0,0)$')
plt.plot(lamb2.real,lamb2.imag, 'o', color='red')


plt.plot(lamb3.real,lamb3.imag, 'o', color='blue', label='$({x_{1}^{*},x_{2}^{*}})=(1,0)$ ou $({x_{1}^{*},x_{2}^{*}})$ = $(-1,0)$')
plt.plot(lamb4.real,lamb4.imag, 'o', color='blue')


# ax.text(lamb1.real-.1, lamb1.imag-.2, '${\lambda_{1}}$', fontsize=12)
# ax.text(lamb2.real-.1, lamb2.real+.1, '${\lambda_{2}}$', fontsize=12)

plt.axvspan(-1.7, 0, -1, 1, alpha=0.3, color='lightgrey')



ax = plt.gca()
ay = plt.gca()
ay.set(ylim=(-1.55,1.55)) 
ax.set(xlim=(-1.7,1.7))
ax.text(.5, .5, 'Instabilidade', fontsize=12)
ax.text(-1.25, .5, 'Estabilidade', fontsize=12)

ax.text(lamb1.real-.06, lamb1.imag-.2, '${\lambda_{1}}=0,618$', fontsize=10)
ax.text(lamb2.real+.06, lamb2.imag-.2, '${\lambda_{2}}=-0,618$', fontsize=10)

ax.text(lamb3.real-.5, lamb3.imag-.3, '${\lambda_{1}}=-0,5+1,32i$', fontsize=10)
ax.text(lamb4.real-.5, lamb4.imag+.2, '${\lambda_{2}}=-0,5-1,32i$', fontsize=10)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='best',
            ncol=3, mode="expand", borderaxespad=0.)

plt.ylabel('Im(${\lambda_{1,2}}$)')
plt.xlabel('Re(${\lambda_{1,2}}$)')

# plt.savefig('fig16.pdf',dpi=300)  

# #######################################################################################
PE1_x1, PE1_x2 =-1, 0 
PE2_x1, PE2_x2 = 0, 0
PE3_x1, PE3_x2 = 1, 0

def Duffing(x,t):
    x1 = x[0]
    x2 = x[1]
    
    f1 = x2
    f2 = -x2+x1-x1**3
    
    return f1,f2

N  = 1000
t = np.linspace(0,-10,N)

x1i, x2i = +.001,+.001 # separatriz verm
x1j, x2j = -.001,-.001 # separatriz azul

r = odeint(Duffing,(x1i, x2i ), t)  # separatriz verm
p = odeint(Duffing,(x1j, x2j ), t)  # separatriz azul

x1,x2 = r[:,0], r[:,1] # separatriz verm
x1p,x2p = p[:,0], p[:,1] # separatriz azul



fig2 = plt.figure()
plt.plot(x1,x2, '-', color= 'red')
plt.plot(x1p,x2p, '-', color= 'blue')



####brincando com a trajetória ###########

x1k, x2k = -4, -15 # chutes
t = np.linspace(0,10,N)

q = odeint(Duffing,(x1k, x2k ), t)  # trajetória em verde
x1q,x2q = q[:,0], q[:,1] # separatriz azul

plt.plot(x1k, x2k, '*', color = 'black')
plt.plot(x1q,x2q, '--', color= 'blue', linewidth=.8)



x1l, x2l = 2, 20 # chutes


s = odeint(Duffing,(x1l, x2l ), t)  # trajetória em verde
x1s,x2s = s[:,0], s[:,1] # separatriz azul

plt.plot(x1l, x2l, '*', color = 'black')
plt.plot(x1s,x2s, '--', color= 'red', linewidth=.8)






plt.plot(PE1_x1, PE1_x2, 'o', color = 'darkorange')
plt.plot(PE2_x1, PE2_x2, 'o', color = 'black')
plt.plot(PE3_x1, PE3_x2, 'o', color = 'green')


ax = plt.gca()
ay = plt.gca()

ax.set(xlim=(-6,6))  
ay.set(ylim=(-33,33))

ax.set_xlabel('$x_{1}$')
ay.set_ylabel('$x_{2}$')

# plt.savefig('fig17.pdf',dpi=300)  