import numpy as npfrom scipy.optimize import rootfrom correlacoes_empiricas3 import f_empiricasimport matplotlib.pyplot as pltimport math as mtfrom matplotlib import ticker####Dados do óleo#########################D      = 0.8901    # g/cm³ a 20-24°CMM_S   = 32.065    # g/molMM_N   = 14.007    # g/molMM_H2  =  2.001    # g/molMM_H2S = 34.067    # g/molMM_NH3 = 17.010    # g/molrho0  = .89  #g/cm³ (densidade do óleo)API   = 30    #°APIVgas  = 10      # a.u   (razao H2/carga)F     = .00235  # cm³/s (vazão de carga)#1 PARÂMETROS PARA O BALANÇO################1.1 Para o balanço na fase gasosa#######R = 8.31447  # J/(mol.K)T = 613.15      #KP =  7        #MPa##1.2 Do catalisador ####################dp    = .035 # cm (diâmetro da partícula) #dissertação ana 2012#dpe   = .02 # cm (diametro da particula equivalente do catalisador mais inerte)E     = .53  #    (porosidade do catalisador)e     = .52  #    (fracao de vazios no leito)qsi   = .55  #    (razao do leito catalitico diluido por inertes)rho_B  = .75  # g/cm³ (densidade do bulk catalitico)rhos  = .75   # g/cm³ (densidade do catalisador)n_HDT  = .7   # (fator efetividade do catalisador para um processo de HDT)eps_B = .4   # a.u (porosidade do leito)##1.3 Cálculo dos parâmetros empíricos#################para = np.zeros(14)para  = f_empiricas(P,T, rho0, API, Vgas, F, dp, eps_B)H_H2  = para[0] # MPa.cm³/mol p/ H2H_H2S = para[1]  # MPa.cm³/mol p/ H2SH_NH3 = para[2]  # MPa.cm³/mol p/ NH3u_G, u_L = para[3], para[4]k_L_H2al, k_L_H2Sal, k_L_NH3al   = para[5], para[6], para[7]k_S_H2, k_S_H2S, k_S_NH3  = para[8], para[9]*10, para[10]a_S = para[11]k_S_S, k_S_N = para[12], para[13]#############################################2 PARÂMETROS PARA AS TAXAS DE REAÇÃO#########2.1 Ordens de reação (dados obtidos do artigo do Leandro)m_HDS = 1.8m_HDN = 1.11n_HDS = .52n_HDN = .17##2.2 constantes calculadas### dados obtidos no artigoEa_HDS = 116.91e3   #J/molEa_HDN = 156.61e3   #J/molk0_HDS = 7.89e14    #k0_HDN = 1.43e14    #k_HDS = k0_HDS*mt.exp(-Ea_HDS/(R*T)) #1541.01     # [cm³/(g.s).(cm³/mol)^0.45] p/ HDSk_HDN = k0_HDN*mt.exp(-Ea_HDN/(R*T)) #1.6500e-6   # [cm³/(g.s).(cm³/mol)]   p/ HDNK_H2S = 71900      # cm³/mol; contsnate de adsorção p/ H2S    #####CONDIÇÕES DE CONTORNO##################P_G_H2S_z0 = 0   # p/H2SP_G_NH3_z0 = 0   # p/NH3P_G_H2_z0  = P   #MPa p/H2C_L_H2S_z0 = 0                 # p/H2SC_L_NH3_z0 = 0                 # p/NH3C_L_H2_z0 = P_G_H2_z0/H_H2     # mol/cm³ p/H2;C_L_S_z0  = ((4053e-6)/MM_S)*D #9.34122e-5    # mol/cm³ p/SC_L_N_z0  = ((1214e-6)/MM_N)*D #6.40552e-5   # mol/cm³ p/N# print('C_L_S_z0 = ', C_L_S_z0)# print('C_L_N_z0 = ', C_L_N_z0)C_S_H2S_z0 = 0  #p/ H2SC_S_NH3_z0 = 0  #p/ NH3C_S_H2_z0  = 0  #p/ H2C_S_S_z0   = 0  #p/ SC_S_N_z0   = 0  #p/ N#######Coeficientes do sistema ################################ Obs.: Apesar de parecer estranho, fazer as operações algebricas antes otimiza o fsolve######u = np.zeros(50, dtype=float)################################################# DEFININDO UMA MALHA UNIFORME##########N = 50h = 26.9   # Comprimento do reatorfig1 = plt.figure()dz =h/N    ########## arrumando os chutes iniciais###### z = 0xguess = np.array([P_G_H2_z0,                   C_L_H2_z0,                    C_L_S_z0,                        1e-5,                        1e-5,                        1e-7,                           0,                  C_L_H2S_z0,                  C_S_H2S_z0,                    C_L_N_z0,                    C_S_N_z0,                        1e-7,                  P_G_NH3_z0,                  C_L_NH3_z0,                 C_S_NH3_z0])pa     =          [P_G_H2_z0,                   C_L_H2_z0,                    C_L_S_z0,                  P_G_H2S_z0,                  C_L_H2S_z0,                    C_L_N_z0,                  P_G_NH3_z0,                  C_L_NH3_z0]        ######(Folha 1, PDF)#########################u[0] = (-u_G/dz) -(R*T*k_L_H2al/H_H2)       #<-----------u[1] = R*T*k_L_H2alu[2] = u_G/dz                              #<-----------       u[3] = (-u_L/dz) - k_L_H2al - k_S_H2*a_S    #<-----------u[4] = k_L_H2al/H_H2u[5] = k_S_H2*a_Su[6] = (u_L/dz)                             #<-----------u[7] = (-u_L/dz) - k_S_S*a_S                #<-----------u[8] = k_S_S*a_Su[9] = (u_L/dz)                             #<-----------u[10] = k_S_H2*a_Su[11] = -k_S_H2*a_Su[12] = -qsi*rho_B*n_HDTu[13] = k_S_S*a_Su[14] = -k_S_S*a_Su[15] = -qsi*rho_B*n_HDTu[16] = -k_HDSu[17] = m_HDSu[18] = n_HDS####### H2S (Folha 2, PDF)#######u[19] = (-u_G/dz) -(R*T*k_L_H2Sal/H_H2S)    #<-----------u[20] = R*T*k_L_H2Salu[21] = u_G/dz                              #<-----------u[22] = (-u_L/dz) - k_L_H2Sal - k_S_H2S*a_S #<-----------u[23] = k_L_H2Sal/H_H2Su[24] = k_S_H2S*a_Su[25] = (u_L/dz)                            #<-----------u[26] = k_S_H2S*a_Su[27] = -k_S_H2S*a_Su[28] = qsi*rho_B*n_HDT####### N (Folha 3, PDF)#######u[29] = (-u_L/dz) - k_S_N*a_S               #<-----------u[30] = k_S_N*a_Su[31] = (u_L/dz)                            #<-----------u[32] = k_S_N*a_Su[33] = -k_S_N*a_Su[34] = -qsi*rho_B*n_HDTu[35] = -k_HDNu[36] = m_HDNu[37] = n_HDN####### NH3 (Folha 4, PDF)#######u[38] = (-u_G/dz) -(R*T*k_L_NH3al/H_NH3)    #<-----------u[39] = R*T*k_L_NH3alu[40] = u_G/dz                              #<-----------u[41] = (-u_L/dz) - k_L_NH3al - k_S_NH3*a_S #<-----------u[42] = k_L_NH3al/H_NH3u[43] = k_S_NH3*a_Su[44] = (u_L/dz)                            #<-----------u[45] = k_S_NH3*a_Su[46] = -k_S_NH3*a_Su[47] = qsi*rho_B*n_HDTdef system0(x, p):    P_G_H2_z0  = p[0]    C_L_H2_z0  = p[1]    C_L_S_z0   = p[2]    P_G_H2S_z0 = p[3]    C_L_H2S_z0 = p[4]    C_L_N_z0   = p[5]    P_G_NH3_z0 = p[6]    C_L_NH3_z0 = p[7]######### Ver PDFs anexo ##########################################    return np.array(       [ u[0]*x[0] +  u[1]*x[1] +               u[2]*P_G_H2_z0,                                    # eq 1 Balanço na fase gasosa para H2         u[3]*x[1] +  u[4]*x[0] +   u[5]*x[2] + u[6]*C_L_H2_z0,                                    # eq 2 Balanço na fase líquida para H2         u[7]*x[3] +  u[8]*x[4] +               u[9]* C_L_S_z0,                                    # eq 3 Balanço na fase líquida para S        u[10]*x[1] + u[11]*x[2] +  u[12]*(x[5]+x[11]),                                             # eq 4 Balanço na fase sólida para H2        u[13]*x[3] + u[14]*x[4] +  u[15]*x[5],                                                     # eq 5 Balanço na fase sólida para S              x[5] +(u[16]*((x[3]**2)**(u[17]/2))*((x[2]**2)**(u[18]/2))/((1+K_H2S*x[8])**2)),     # eq 6 Lei de potência para HDS        u[19]*x[6] +   u[20]*x[7] +             u[21]*P_G_H2S_z0,                                  # eq 10 Balanço na fase gás para H2S        u[22]*x[7] +   u[23]*x[6] +u[24]*x[8] + u[25]*C_L_H2S_z0,                                  # eq 11 Balanço na fase líquida para H2S        u[26]*x[7] +   u[27]*x[8] +u[28]*x[5],                                                     # eq 12 Balanço na fase sólida para H2S        u[29]*x[9] + u[30]*x[10] + u[31]*C_L_N_z0,                                                 # eq 7 Balanço na fase líquida para N        u[32]*x[9] + u[33]*x[10] + u[34]*x[11],                                                    # eq 8 Balanço na fase sólida para N             x[11] + (u[35]*((x[10]**2)**(u[36]/2))*((x[2]**2)**(u[37]/2))),                       # eq 6 Lei de potência para HDN        u[38]*x[12] +  u[39]*x[13] +             u[40]*P_G_NH3_z0,                                 # eq 13 Balanço na fase gás para NH3        u[41]*x[13] +  u[42]*x[12] +u[43]*x[14] +u[44]*C_L_NH3_z0,                                 # eq 14 Balanço na fase líquida para NH3        u[45]*x[13] +  u[46]*x[14] +u[47]*x[11],                                                   # eq 15 Balanço na fase sólida para NH3         ]        )xsol0 = root(system0, xguess, method='hybr', args=pa) # res = 2.98116e-20Storage = np.zeros((N, 15))        #<-----------initial_guess = xsol0.xfor i in range(1, N):              #<-----------     Storage[0, :]  =  initial_guess     pa_args        = [initial_guess[0],                       initial_guess[1],                       initial_guess[3],                       initial_guess[6],                       initial_guess[7],                       initial_guess[9],                      initial_guess[12],                      initial_guess[13]]                                                      ####  atenção: essa linha é importante. Note que ela está substituíndo os z0 pela interação anterior. Veja o PDF para entender a numeração "n" de initial_guess[n]     # print(pa_args)     # input()     xsol0          = root(system0, initial_guess, method='hybr', args=pa_args)     initial_guess  = xsol0.x     Storage[i, :]  = initial_guess# ################## Ajustando o primeiro ponto no gráfico #################Storage[0][0]  = P_G_H2_z0Storage[0][1]  = C_L_H2_z0Storage[0][2]  = C_L_H2_z0Storage[0][3]  = C_L_S_z0Storage[0][4]  = C_L_S_z0Storage[0][6]  = P_G_H2S_z0Storage[0][7]  = C_L_H2S_z0Storage[0][8]  = C_S_H2S_z0Storage[0][9]  = C_L_N_z0Storage[0][10] = C_S_N_z0Storage[0][12] = P_G_NH3_z0Storage[0][13] = C_L_NH3_z0########## Convertendo as unidades##########ClS = (Storage[:, 3]*MM_S*(10**6))/DClN = (Storage[:, 9]*MM_N*(10**6))/DClH2  = (Storage[:, 1]*MM_H2)ClH2S = (Storage[:, 7]*MM_H2S)ClNH3 = (Storage[:, 13]*MM_NH3)fig2 = plt.figure()plt.plot(np.linspace(0, h, len(ClS)), ClS,'.-', color='black')      # o refere-se ao tipo de linhaplt.ylabel('Enxofre total (mg/Kg)')plt.xlabel('Tamanho do reator (cm)')ax = plt.gca()ay = plt.gca()formatter = ticker.ScalarFormatter(useMathText=True)formatter.set_scientific(True) formatter.set_powerlimits((-1,1)) ax.yaxis.set_major_formatter(formatter) plt.savefig('fig2.pdf', dpi=300)    ##<---------- salva a figfig3 = plt.figure()plt.plot(np.linspace(0, h, N), ClN, '.-', color='black')plt.ylabel('Nitrogênio Total (mg/Kg)')plt.xlabel('Tamanho do reator (cm)')ax = plt.gca()ay = plt.gca()formatter = ticker.ScalarFormatter(useMathText=True)formatter.set_scientific(True) formatter.set_powerlimits((-1,1)) ax.yaxis.set_major_formatter(formatter) plt.savefig('fig3.pdf', dpi=300)    ##<---------- salva a figfig3 = plt.figure()plt.plot(np.linspace(0, h, N), ClH2, '.-', color='black')plt.ylabel('Concentração de hidrogênio (g/cm³)')plt.xlabel('Tamanho do reator (cm)')ax = plt.gca()ay = plt.gca()formatter = ticker.ScalarFormatter(useMathText=True)formatter.set_scientific(True) formatter.set_powerlimits((-1,1)) ax.yaxis.set_major_formatter(formatter) plt.savefig('fig4.pdf', dpi=300)    ##<---------- salva a figfig4 = plt.figure()plt.plot(np.linspace(0, h, N), ClH2S,  'b.-', label='H$_{2}$S')plt.plot(np.linspace(0, h, N), ClNH3, 'r.-', label='NH$_{3}$')plt.ylabel('Gás dissolvido (g/cm³)')plt.xlabel('Tamanho do reator (cm)')plt.legend(loc='upper right', fontsize=10)ax = plt.gca()ay = plt.gca()formatter = ticker.ScalarFormatter(useMathText=True)formatter.set_scientific(True) formatter.set_powerlimits((-1,1)) ax.yaxis.set_major_formatter(formatter) plt.savefig('fig5.pdf', dpi=300)    ##<---------- salva a fig