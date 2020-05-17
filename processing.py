#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import preprocessing
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#Solving for global u using the global stiffness and force matrices
def global_u(n_e, order=1):
    K,f = preprocessing.global_matrices(n_e,order)
    n_n = order * n_e + 1
    u_solved = np.matmul(f[1:],np.linalg.inv(K[1:,1:])).reshape(1,n_n-1)
    u_fixed = np.zeros((1,1))
    u = np.concatenate((u_fixed,u_solved),axis=1)
    return(u)

#parameters for the plotted model
order = 2
n_e = 2 #number of elements
n_n = order * n_e + 1 #number of nodes

x_n = np.linspace(0,preprocessing.length,n_n) #x coordinates of the nodes
l_e = preprocessing.length / (n_e)
xg = np.linspace(0,preprocessing.length,1000) #range of x for plotting


u = global_u(n_e,order)

#initialize the output plots:
fig1 = plt.figure(num=1, figsize=(6.5, 3),dpi=300) #figure for u and sigma plots
u_ax = plt.subplot(1,2,1)
stress_ax = plt.subplot(1,2,2)

fig2 = plt.figure(num=2, figsize=(6.5, 3),dpi=300) # figure for difference plots
diff_u_ax = plt.subplot(1,2,1)
diff_stress_ax = plt.subplot(1,2,2)

#calculate the values of the shape functions across each element
for e in range(n_e):
    x_e = x_n[e*order:order*e+1+order]
    x = np.linspace(x_e[0],x_e[-1],100) #generate some x values within the element
    
    #calculate the analyticalsolutions across the element
    u_anal = (-5e-3*x**2 + 4e-2*x)*1e2 #cm
    stress_anal = (-10e-3*x + 4e-2)*preprocessing.modulus*1e-6 #MPa
    
    if order == 1:
        u_e = (((x_e[-1] - x)*u[0][e] + (x - x_e[0])*u[0][e+1])/l_e) *1e2 #cm
        dudx_e = np.zeros(len(x))+((-1)*u[0][e] + u[0][e+1])/l_e
        stress_e = dudx_e*preprocessing.modulus * 1e-6 #MPa
      
        u_ax.plot(x,u_e,label ='elem. '+str(e+1))
        stress_ax.plot(x,stress_e,label ='elem. ' +str(e+1))        
        diff_u_ax.plot(x,u_e-u_anal,label ='elem. ' +str(e+1))
        diff_stress_ax.plot(x,stress_e-stress_anal,label ='elem. ' +str(e+1))

    if order == 2:
        u_e = 1e2*(2/(l_e**2))*((x-x_e[1])*(x-x_e[2])*u[0][e*order]
        - 2*(x-x_e[0])*(x-x_e[2])*u[0][e*order+1] 
        + (x-x_e[0])*(x-x_e[1])*u[0][e*order+2]) #cm
        
        dudx_e = (2/(l_e**2))*((2*x-(x_e[1]+x_e[2]))*u[0][e*order]
        - 2*(2*x-(x_e[0]+x_e[2]))*u[0][e*order+1]
        + (2*x-(x_e[0]+x_e[1]))*u[0][e*order+2])
        
        stress_e = dudx_e*preprocessing.modulus * 1e-6 #MPa
        

        u_ax.plot(x,u_e,label ='elem. ' +str(e+1))
        stress_ax.plot(x,stress_e,label ='elem. ' +str(e+1))
        
        diff_u_ax.plot(x,u_e-u_anal,label ='elem. ' +str(e+1))
        diff_stress_ax.plot(x,stress_e-stress_anal,label ='elem. ' +str(e+1))

#plot the analytical solution across the entire domain
u_anal = (-5e-3*xg**2 + 4e-2*xg)*1e2 #cm
stress_anal = (-10e-3*xg + 4e-2)*preprocessing.modulus*1e-6 #MPa
u_ax.plot(xg,u_anal,':',color='gray',label='$u_{Exact}$')
stress_ax.plot(xg,stress_anal,':',color='gray',label='$\sigma_{Exact}$')

#set the essential figure components
u_ax.legend(fontsize=8)
u_ax.set_title('Displacement - 2nd order',size=10)
u_ax.set_xlabel('Position (m)',size=9)
u_ax.set_ylabel('$u$ (cm)',size=9)
u_ax.xaxis.set_tick_params(labelsize=8)
u_ax.yaxis.set_tick_params(labelsize=8)

stress_ax.legend(fontsize=8)
stress_ax.set_title('Stress - 2nd order',size=10)
stress_ax.set_xlabel('Position (m)',size=9)
stress_ax.set_ylabel('$\sigma$ (MPa)',size=9)
stress_ax.xaxis.set_tick_params(labelsize=8)
stress_ax.yaxis.set_tick_params(labelsize=8)

#adjusting spacing...
fig1.subplots_adjust(wspace=0.3)
fig1.subplots_adjust(top=0.93)
fig1.subplots_adjust(left=0.11)
fig1.subplots_adjust(bottom=0.145)
fig1.subplots_adjust(right=0.97)

diff_u_ax.legend(loc=1,fontsize=8)
diff_u_ax.set_title('FEM-Exact displacement difference',size=10)
diff_u_ax.set_xlabel('Position (m)',size=9)
diff_u_ax.set_ylabel('$\Delta u$ (cm)',size=9)

diff_u_ax.xaxis.set_tick_params(labelsize=8)
diff_u_ax.yaxis.set_tick_params(labelsize=8)

#for the 2nd order plot:
diff_u_ax.set_ylabel('$\Delta u$ (cm*$10^{-15}$)',size=9)
diff_u_ax.yaxis.offsetText.set_visible(False)

diff_stress_ax.legend(loc=1,fontsize=8)
diff_stress_ax.set_title('FEM-Exact stress difference',size=10)
diff_stress_ax.set_xlabel('Position (m)',size=9)
diff_stress_ax.set_ylabel('$\Delta \sigma$ (MPa)',size=9)

diff_stress_ax.xaxis.set_tick_params(labelsize=8)
diff_stress_ax.yaxis.set_tick_params(labelsize=8)

#for the second order plot:
diff_stress_ax.set_ylabel('$\Delta \sigma$ (MPa*$10^{-14}$)',size=9)
diff_stress_ax.yaxis.offsetText.set_visible(False)

fig2.subplots_adjust(wspace=0.3)
fig2.subplots_adjust(top=0.9)
fig2.subplots_adjust(left=0.11)
fig2.subplots_adjust(bottom=0.145)
fig2.subplots_adjust(right=0.97)