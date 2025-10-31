# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:33:30 2025

@author: Termohidraulica
"""

from pyREFPROP import rhof_P, rhog_P, muf_P, sigma_P, kf_P, Hf_P, Hg_P
import pandas as pd
import numpy as np

def get_data_and_dimensional_matrix(data_path = "chf_public.csv"):

    #Carga de la base de datos
    #-,-,m,m,kPa,kg/m^2/s,-,kJ/kg,C,kW/m^2,kW/m^2
    data_chf = pd.read_csv(data_path, skiprows=[1])
    data_chf.describe()
    data_chf.keys()
    
    data_chf['g'] = 9.81                                     # [m/s2]  : L*(T**-2)
    data_chf['rhof'] = rhof_P(data_chf['Pressure'])          # [kg/m3] : M*(L**-3)
    data_chf['rhog'] = rhog_P(data_chf['Pressure'])          # [kg/m3] : M*(L**-3)
    data_chf['muf'] = muf_P(data_chf['Pressure'])*1e-6       # 
    data_chf['sigma'] = sigma_P(data_chf['Pressure'])
    data_chf['kf'] = kf_P(data_chf['Pressure'])
    data_chf['hf'] = Hf_P(data_chf['Pressure'])
    data_chf['hg'] = Hg_P(data_chf['Pressure'])
    data_chf['rhof - rhog'] = data_chf['rhof'] - data_chf['rhog']
    data_chf['hfg'] = data_chf['hg'] - data_chf['hf']
    data_chf['1-Xout'] = 1 - data_chf['Outlet Quality']
    #print(data_chf.keys())
    
    data_selected = data_chf[
        [
         'Tube Diameter',     # [m] : L
         'Mass Flux',          # [kg/m**2/s] : M*(L**-2)*(T**-1)
         '1-Xout',    # [-]
         'g',                 # [m/s**2] : L*(T**-2)
         'rhof',              # [kg/m**3] : M*(L**-3)
         'rhog',              # [kg/m**3] : M*(L**-3)
         'rhof - rhog',       # [kg/m**3] : M*(L**-3)
         'muf',                # [Pa*s] = [kg/m/s] : M*(L**-1)*(T**-1)
         'sigma',             # [N/m] = [kg/s**2] : M*(T**-2)
         'hf',                # [kJ/kg]=[N*m/kg]=[m**2/s**2] : (L**2)*(T**-2)
         'hg',                # (M**2)*(T**-2)
         'hfg',               # (M**2)*(T**-2)
         'CHF',               # [kW/m2]=[J/s/m2]=[Kg/s**3] = M*(T**-3)
         #'Pressure',         # kPa : k[kg/m/s**2] : M*(L**-1)*(T**-2)
         #'kf',               # [W/m/K]=[J/s/m/K]=[Kg*m/s**3/K] = M*L*(T**-3)*(Θ**-1) # No lo utilizo porque no hay otro parametro que utilice la temperatura
         ]
        ]
    
    D = np.array([
        [1,-2, 0, 1,-3,-3,-3,-1, 0, 2, 2, 2], # L : Length
        [0,-1, 0,-2, 0, 0, 0,-1,-2,-2,-2,-2], # T : Time
        [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], # M : Mass
        # Θ : Temperature
        ])
    
    Bo_chf = data_selected['CHF']/data_selected['Mass Flux']/data_selected['hfg']
    return data_selected, D, Bo_chf