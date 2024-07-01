#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:23:59 2023

@author: vdmsermoud
"""
import numpy as np
from numba import jit
import scipy.integrate as integrate

# Function to evaluate the Heaviside function for each point of the 1D grid
@jit(nopython=True)
def W_Heaviside(H,Z_vec,Nump,Sigma_ff,dz):
    WH = np.zeros((Nump,Nump))
    for j in range(Nump):
        Zinf = Z_vec[j]-0.5*Sigma_ff
        Zmax = Z_vec[j]+0.5*Sigma_ff
        kinf = max(int(Zinf/dz+1.e-8),0)
        ksup = min(int(Zmax/dz+1.e-8),Nump-1)
        for i in range(Nump):
            if (i >= kinf and i <= ksup):
                WH[i,j] = 1.
            else:
                WH[i,j] = 0.
    return WH

# Function to evaluate the weight for each point of the 1D grid
@jit(nopython=True)
def Omega_calc(H,Nump,Sigma_ff,Z_vec):
    omega0 = W_Heaviside(H,Z_vec,Nump,Sigma_ff,Z_vec[1]-Z_vec[0])
    omega = np.zeros((Nump,Nump,6))
    omega[:,:,0] = omega0[:,:]/Sigma_ff
    omega[:,:,1] = omega0[:,:]*0.5
    omega[:,:,2] = omega0[:,:]*np.pi*Sigma_ff
    omega[:,:,3] = omega0[:,:]*np.pi
    for j in range(Nump):
        for i in range(Nump):
            aux = ((0.5*Sigma_ff)**2-(Z_vec[i]-Z_vec[j])**2)
            omega[i,j,3] = omega[i,j,3]*aux
            omega[i,j,5] = omega0[i,j]*(Z_vec[i]-Z_vec[j])*2.*np.pi
    omega[:,:,4] = omega[:,:,5]/(2.*np.pi*Sigma_ff)
    return omega

@jit(nopython=True)
def calc_N(H,Omega_vec,density,Nump,dz,Sigma_ff,Z_vec):
    N_alpha = np.zeros((Nump,6))
    conv = np.zeros((Nump,6))
    for n in range(6):
        for j in range(Nump):    
            Zinf = Z_vec[j]-0.5*Sigma_ff
            Zmax = Z_vec[j]+0.5*Sigma_ff
            kinf = max(int(Zinf/dz+1.e-8),0)
            ksup = min(int(Zmax/dz+1.e-8),Nump-1)
            for i in range(Nump):
                if i==kinf or i==ksup:
                    conv[j,n] += 0.5*Omega_vec[i,j,n]*density[i] 
                else:
                    conv[j,n] += Omega_vec[i,j,n]*density[i] 
    N_alpha = conv*dz
    return N_alpha

@jit(nopython=True)
def d_Phi(H,Omega_vec,Nump,rho_vec,dz,Sigma_ff,Z_vec):
    N_fmt = calc_N(H,Omega_vec,rho_vec,Nump,dz,Sigma_ff,Z_vec)
    dPhi = np.zeros((Nump,6))
    for i in range(Nump):
        if N_fmt[i,3]> 1.e-8:
            Temp = 1.-N_fmt[i,3]
            dPhi[i,0] = -np.log(Temp)
            dPhi[i,1] = N_fmt[i,2]/Temp
            dPhi[i,2] = (N_fmt[i,1]/Temp)+(np.log(Temp)/N_fmt[i,3]+
                        1.0/Temp**2)*(N_fmt[i,2]**2-
                        N_fmt[i,5]**2)/(12.0*np.pi*N_fmt[i,3])
            dPhi[i,3] = N_fmt[i,0]/Temp+(N_fmt[i,1]*N_fmt[i,2]-N_fmt[i,4]*
                    N_fmt[i,5])/Temp**2-(np.log(Temp)/(18.0*np.pi*
                    N_fmt[i,3]**3)+1.0/(36.0*np.pi*N_fmt[i,3]**2*Temp)+
                    (1.0-3.0*N_fmt[i,3])/(36.0*np.pi*N_fmt[i,3]**2*
                    Temp**3))*(N_fmt[i,2]**3-3.0*N_fmt[i,2]*N_fmt[i,5]**2)
            dPhi[i,4] = -N_fmt[i,5]/Temp
            dPhi[i,5] = -N_fmt[i,4]/Temp-(np.log(Temp)/N_fmt[i,3]+1.0
                    /Temp**2)*N_fmt[i,2]*N_fmt[i,5]/(6.0*np.pi*N_fmt[i,3])
    return dPhi


@jit(nopython=True)
def calc_dFHS(H,Omega_vec,rho_vec,Nump,dz,T,Sigma_ff,Z_vec):
    dPhi = d_Phi(H,Omega_vec,Nump,rho_vec,dz,Sigma_ff,Z_vec)
    conv = np.zeros((Nump,6))
    FHS = np.zeros((Nump))
    for n in range(6):
        if n == 4 or n == 5:
            Omega_vec[:,:,n] *= -1
        for j in range(Nump):
            Zinf = Z_vec[j]-0.5*Sigma_ff
            Zmax = Z_vec[j]+0.5*Sigma_ff
            kinf = max(int(Zinf/dz+1.e-8),0)
            ksup = min(int(Zmax/dz+1.e-8),Nump-1)
            for i in range(Nump):
                if i==kinf or i==ksup:
                    conv[j,n] += 0.5*Omega_vec[i,j,n]*dPhi[i,n]
                else:
                    conv[j,n] += Omega_vec[i,j,n]*dPhi[i,n]
    for i in range(Nump):
        FHS[i] = np.sum(conv[i,:])*dz
    return FHS*T

@jit(nopython=True)
def Phi(H,N_poid,Nump,rho_vec,dz,Sigma_ff,Z_vec):
    N_fmt = calc_N(H,N_poid,rho_vec,Nump,dz,Sigma_ff,Z_vec)
    Phi = np.zeros((Nump))
    Phi_S = np.zeros((Nump))
    Phi_V = np.zeros((Nump))
    for i in range(Nump):
        if N_fmt[i,3]> 1.e-8:
            Temp = 1.-N_fmt[i,3]
            
            Phi_S[i] = -N_fmt[i,0]*np.log(Temp)+N_fmt[i,1]*N_fmt[i,2]/Temp
            +N_fmt[i,2]**3*(np.log(Temp)/(36.*np.pi*N_fmt[i,3]**2)
            +1./(36.*np.pi*N_fmt[i,3]*Temp**2))
            
            Phi_V[i] = -N_fmt[i,4]*N_fmt[i,5]/Temp-N_fmt[i,2]*(np.log(Temp)
            /(12.*np.pi*N_fmt[i,3]**2)
            +1./(12.*np.pi*N_fmt[i,3]*Temp**2))*N_fmt[i,5]*N_fmt[i,5]
            
    Phi = Phi_S+Phi_V
    return Phi
    
@jit(nopython=True)
def calc_FHS(H,N_poid,rho_vec,Nump,dz,T,Sigma_ff,Z_vec):
    Phi_loc = Phi(H,N_poid,Nump,rho_vec,dz,Sigma_ff,Z_vec)
    Phi_loc[-1] *= 0.5
    Phi_loc[0] *= 0.5
    FHS=np.sum(Phi_loc)*dz
    return T*FHS
