#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def global_u(n_e, order=1):
    K,f = preprocessing.global_matrices(n_e,order)
    n_n = order * n_e + 1
    u_solved = np.matmul(f[1:],np.linalg.inv(K[1:,1:])).reshape(1,n_n-1)
    u_fixed = np.zeros((1,1))
    u = np.concatenate((u_fixed,u_solved),axis=1)
    return(u)


order = 1
n_e = 4
n_n = order * n_e + 1

x_n = np.linspace(0,preprocessing.length,n_n) #x coordinates of the nodes
l_e = preprocessing.length / (n_e)

u = global_u(n_e,order)
#x = np.linspace(0,preprocessing.length,1000)

for e in range(n_e):
    x_e = x_n[e*order:order*e+1+order]
    x = np.linspace(x_e[0],x_e[-1],100)
    print(x_e)
    u_anal = -5e-3*x**2 + 4e-2*x
#    u_anal = 
    if order == 1:
        u_e = ((x_e[-1] - x)*u[0][e] + (x - x_e[0])*u[0][e+1])/l_e
        plt.plot(x,u_e,label = str(e+1))
        plt.plot(x,u_anal,':',color='gray')

        
    if order == 2:
        u_e = 2*((x-x_e[1])*(x-x_e[2])*u[0][e*order] -2 *(x-x_e[0])*(x-x_e[2])*u[0][e*order+1] + (x-x_e[0])*(x-x_e[1])*u[0][e*order+2])/(l_e**2)
        plt.plot(x,u_e,label = str(e+1))
        plt.plot(x,u_anal,':',color='gray')
plt.legend(title='Element Number')
plt.title('Displacement Graph')
plt.xlabel('Position (m)')
plt.ylabel('Displacement (m)')