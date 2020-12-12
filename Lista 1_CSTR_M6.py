# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:29:07 2020

@author: caioh
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib.pyplot as plt




#############Dados##################
q = .1 # m³/h
V = .1 # m³

k0      = 9703*3600 # 1/h
_deltaH = 5960      # kcal/kgmol
Ea      = 11843     # kcal/kgmol

Cp = 500   # kcal/m³/K
Ah = 15    # kcal/h/K
R  = 1.987 # kcal/kgmol/K

Tc = 298.5  # K
Tf = 298.15 # K
Cf = 10     # kgmol/m³





####  A) Faça o gráfico da curva do calor gerado e calor retirado   #########
####  de modo a observar a existência de três estados estacionários #########
####  do reator. Use a faixa de temperatura 300K£T£400K.            #########

Ti = 300
Tfinal = 400
T = np.linspace(Ti, Tfinal, 1000)

def Calor(T):
    Qr = q*Cp*(T-Tf)+Ah*(T-Tc)
    Qg = _deltaH*V*((q*Cf*k0*(np.exp(-Ea/(R*T))))/(q+(V*k0*np.exp(-Ea/(R*T)))))
    return np.array([Qr, Qg])

Qr, Qg = Calor(T)

# plt.figure(1)             # plotei o gráfico só para saber onde os EEs estavam, pra me ajudar nos chutes
# plt.plot(T, Qr, '-', label='Qr (Calor retirado)', color = 'blue')   
# plt.plot(T, Qg, '-', label='Qg (Calor gerado)', color='red')  
# plt.legend(loc='upper left', fontsize=9)
# ax = plt.gca()
# ay = plt.gca()
# ax.set(xlim=(Ti, Tfinal)) 
# ay.set(ylim=(0, 7000)) 
# plt.ylabel('Q (kcal/h)')
# plt.xlabel('T (K)')

#### B) Resolva as equações do modelo estacionário do reator (duas     #####
#### equações algébricas não lineares acopladas) com variados chutes   #####
#### iniciais para o par (CEE,TEE), a fim de obter os três             #####
#### estados estacionários do reator (para o conjunto de dados acima). #####

############ Sistema Algébrico######
def Algsystem(x):
    C,T = x
    
    F1 = (q*Cf- q*C - C*V*k0*np.exp(-Ea/(R*T)))/V 
    F2 = (q*Cp*(Tf-T) + (_deltaH)*V*k0*np.exp(-Ea/(R*T))*C - Ah*(T-Tc))/(V*Cp) 

    return np.array([F1, F2])   

#### Chutes #####
Cee0_1, Tee0_1 = 8, 310
Cee0_2, Tee0_2 = 5.6, 336
Cee0_3, Tee0_3 = 2.2, 368

Cee1, Tee1 =  fsolve(Algsystem, (Cee0_1, Tee0_1))
Cee2, Tee2 =  fsolve(Algsystem, (Cee0_2, Tee0_2))
Cee3, Tee3 =  fsolve(Algsystem, (Cee0_3, Tee0_3))

qee1, qee2, qee3 = Calor(Tee1), Calor(Tee2), Calor(Tee3)

# print(Cee1,Tee1) # 8.503577778059714 311.95180991194474
# print(Cee2,Tee2) # 5.686587398609067 337.78144477583066
# print(Cee3,Tee3) # 2.2946418346280915 368.88297640864084

################### PLOTANDO FIGURA 1 #####################
plt.figure(1)            # agora eu plotei novamente com os pontos do EE
plt.plot(T, Qr, '-', label='Qr (Calor retirado)', color = 'blue')   
plt.plot(T, Qg, '-', label='Qg (Calor gerado)', color='red')  
plt.plot(Tee1, qee1[0], '*', label='EE1', color='cyan')  
plt.plot(Tee2, qee2[0], '*', label='EE2', color='black') 
plt.plot(Tee3, qee3[0], '*', label='EE2', color='darkorange')   
plt.legend(loc='upper left', fontsize=9)
ax = plt.gca()
ay = plt.gca()
ax.set(xlim=(Ti, Tfinal)) 
ay.set(ylim=(0, 7000)) 
plt.ylabel('Q (kcal/h)')
plt.xlabel('T (K)')


#### C) Resolva as equações transientes do modelo (duas equações            #####
#### diferenciais ordinárias acopladas) e dê, como condição inicial,        #####
#### vários pontos nas vizinhanças dos três estados estacionários           #####
#### (valores de C0 e T0 espalhados pelo espaço C×T). Armazene os dados das #####
#### simulações para construir o plano de fases do sistema, isto é,         #####
#### um gráfico C×T (ou viceversa)tendo o tempo como parâmetro.             #####
#### Dica: utilize uma mesma cor de linha para as trajetórias dinâmicas     #####
#### chegando ao mesmo ponto.                                               #####                                          

############ Sistema de EDO#########
def ODEsystem(x,t):
    C = x[0]
    T = x[1]
    
    dCdt = (q*Cf- q*C - C*V*k0*np.exp(-Ea/(R*T)))/V 
    dTdt = (q*Cp*(Tf-T) + (_deltaH)*V*k0*np.exp(-Ea/(R*T))*C - Ah*(T-Tc))/(V*Cp)

    return dCdt, dTdt    

####condições iniciais#############

# print(Cee1,Tee1) # 8.503577778059714 311.95180991194474   coloquei aqui só pra me lembrar
# print(Cee2,Tee2) # 5.686587398609067 337.78144477583066
# print(Cee3,Tee3) # 2.2946418346280915 368.88297640864084

N = 12

C0 = np.linspace(0, 15, N)
T0 = np.linspace(150, 450, len(C0))


tf = 30
M = 1000
t  = np.linspace(0, tf, M)


################## PLOTANDO FIGURA 2 ##################### 
plt.figure(2)
for j in range(N): 
    for i in range(N):
    
        x = odeint(ODEsystem, (C0[i],T0[j]), t)
        
        if (x[M-1,1] > 360):
    
          plt.plot(x[:,0], x[:,1], color='red', linewidth=.2) 
    
        else:
          plt.plot(x[:,0], x[:,1], color='cornflowerblue', linewidth=.2) 
   
plt.plot(Cee1, Tee1, '*', label='EE1', color='cyan')  
plt.plot(Cee2, Tee2, '*', label='EE2', color='black')  
plt.plot(Cee3, Tee3, '*', label='EE3',color='darkorange')  
ax = plt.gca()
ay = plt.gca()
plt.legend(loc='upper right', fontsize=9)
ay.set(ylim=(100, 650)) 
ax.set(xlim=(-1.5, 16))     
plt.ylabel('T (K)')
plt.xlabel('C (kgmol/m³) ')  




#### D) Programe a estratégia apresentada nas notas de aula (Seção 2.5.5) ####
#### para construir o diagrama de soluções estacionárias do reator,       ####
#### tendo, inicialmente, a temperatura da camisa como parâmetro          ####
#### independente e, a seguir, a temperatura da alimentação. Construa os  ####
#### diagramas para variados valores de q/V (deixando V constante e       ####
#### variando q), armazenando os  resultados. Faça um diagrama            ####                                                    ####
#### tridimensional representando, numa coordenada, TEE, e nas demais     #### 
#### coordenadas q/V e Tc (ou Tf).                                        #### 

def func1(Te,q_V):
        
         
    
        Ce = (q_V*Cf)/(q_V + k0*np.exp(-Ea/(R*Te)))
        
        Tc_var = Te + (V*q_V*Cp*(Te-Tf)  -_deltaH*V*k0*np.exp(-Ea/(R*Te))*Ce)/Ah
        Tf_var = Te + ((Ah*(Te-Tc) - _deltaH*V*k0*np.exp(-Ea/(R*Te))*Ce))/(V*q_V*Cp)

        return Ce, Tc_var, Tf_var


Te     = np.linspace(Ti, Tfinal, M)
Ce     = np.zeros(len(Te))
Tc_var = np.zeros(len(Te))
Tf_var = np.zeros(len(Te))



for k in range(len(Te)):
     Ce[k], Tc_var[k], Tf_var[k] = func1(Te[k], q/V)

################### PLOTANDO FIGURA 3 #####################
plt.figure(3)
plt.subplot(2,1,1)
ax = plt.gca()
ay = plt.gca()
ay.set(ylim=(300,400)) 
ax.set(xlim=(270,400)) 
plt.plot(np.full(len(Te), 305.5), Te, '--')
plt.plot(np.full(len(Te), 289.5), Te, '--')
plt.plot(Tc_var,Te, color="black", label='Tc')
ax.text(390, 380, '(a)', fontsize=15)
plt.ylabel('Tc (k)')


plt.subplot(2,1,2)
ax = plt.gca()
ay = plt.gca()
ay.set(ylim=(300,400)) 
ax.set(xlim=(270,400)) 
plt.plot(np.full(len(Te), 300.5), Te, '--')
plt.plot(np.full(len(Te), 295.5), Te, '--')
plt.plot(Tf_var,Te, color="black")
ax.text(390, 380, '(b)', fontsize=15)
plt.ylabel('Tf (k)')
plt.xlabel('Te (K)')

################### Agora Variando q #####################
q_f     = np.linspace(.01,.1,M)
q_V     = q_f/V

Te, q_V = np.meshgrid(Te, q_V)
Ce2, Tc2, Tf2 = func1(Te, q_V)


# # ################### PLOTANDO FIGURA 4 #####################
plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_wireframe(q_V , Te, Tf2)
ax.set_xlabel('q/V (h-¹)')
ax.set_ylabel('Te (K)')
ax.set_zlabel('Tf (K)')


# ################### PLOTANDO FIGURA 5 #####################
plt.figure(5)
ax = plt.axes(projection='3d')
ax.plot_wireframe(q_V, Te, Tc2)
ax.set_xlabel('q/V (h-¹)')
ax.set_ylabel('Te (K)')
ax.set_zlabel('Tc (K)')





              




