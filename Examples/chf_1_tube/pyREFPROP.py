# -*- coding: utf-8 -*-
"""
Saturation properties
"""

import pandas as pd
from scipy import interpolate

'''
-------------------------------------------------------------------------------
                          Liquid       Vapor        Liquid       Vapor        
Temperature  Pressure     Density      Density      Enthalpy     Enthalpy     
(C)          (kPa)        (kg/m^3)     (kg/m^3)     (kJ/kg)      (kJ/kg)      

------------------------------------------------------------------------------

Liquid       Vapor        Liquid       Vapor        Liquid       Vapor   
Entropy      Entropy      Therm. Cond. Therm. Cond. Viscosity    Viscosity       
(kJ/kg-K)    (kJ/kg-K)    (W/m-K)      (W/m-K)      (uPa-s)      (uPa-s)      

-------------------------------------------------------------------------------

Surf. Tension  Liquid Cp
(N/m)          (kJ/kg-K)

-------------------------------------------------------------------------------

'''
data_pressure = pd.read_csv("pressure_table2_refprop.txt",
                            skiprows=[0,1,2,3,4], sep=  '\s+', decimal=",",
                            names = ['Temperature', 'Pressure',
                                     'Liquid Density', 'Vapor Density',
                                     'Liquid Enthalpy', 'Vapor Enthalpy',
                                     'Liquid Entropy', 'Vapor Entropy',
                                     'Liquid Therm Cond', 'Vapor Term Cond',
                                     'Liquid Viscosity', 'Vapor Viscosity',
                                     'Surf Tension', 'Liquid Cp'])

data_temperature = pd.read_csv("temperature_table2_refprop.txt",
                            skiprows=[0,1,2,3,4], sep=  '\s+', decimal=",",
                            names = ['Temperature', 'Pressure',
                                     'Liquid Density', 'Vapor Density',
                                     'Liquid Enthalpy', 'Vapor Enthalpy',
                                     'Liquid Entropy', 'Vapor Entropy',
                                     'Liquid Therm Cond', 'Vapor Term Cond',
                                     'Liquid Viscosity', 'Vapor Viscosity',
                                     'Surf Tension'])


rhof_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Liquid Density'])
rhog_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Vapor Density'])

Hf_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Liquid Enthalpy'])
Hg_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Vapor Enthalpy'])
Hf_T = interpolate.interp1d(data_temperature['Temperature'], data_temperature['Liquid Enthalpy'])

kf_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Liquid Therm Cond'])

muf_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Liquid Viscosity'])

sigma_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Surf Tension'])

cpf_P = interpolate.interp1d(data_pressure['Pressure'], data_pressure['Liquid Cp'])
