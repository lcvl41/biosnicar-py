import yaml
from ruamel.yaml import YAML
import numpy as np
import math as m
import pandas as pd
input_file = "inputs.yaml" 
data_file = pd.read_csv("crust_dev_params.csv")
#data_out = pd.DataFrame(columns = ['sw', 'lw', 'latent', 'conv', 'cond'])
data_out=[]

def calculate_energy_fluxes(meteo_params):
    
	AIR_T_0 = meteo_params['AIR_T_0'] + 273.15
	AIR_T_Z = meteo_params['AIR_T_Z'] + 273.15
	AIR_P_Z = meteo_params['AIR_P_Z']
	RH = meteo_params['RH']
	AIR_VAP_P_0 = 100*6.1078*m.exp((17.269*(AIR_T_0-273.25))/(AIR_T_0-273.15+237.3))
	AIR_VAP_P_Z = RH*6.1078*m.exp((17.269*(AIR_T_Z-273.15))/(AIR_T_Z-273.15+237.3))
	WIND_SPEED = meteo_params['WIND_SPEED']
	HEAT_CAP = meteo_params['HEAT_CAP'] # water or snow
	T_RAIN_SNOW_FALL = meteo_params['T_RAIN_SNOW_FALL'] + 273.15
	FLUX_RAIN_SNOW_FALL = meteo_params['FLUX_RAIN_SNOW_FALL']
	INCOMING_LONGWAVE = meteo_params['INCOMING_LONGWAVE'] 
	OUTGOING_LONGWAVE = meteo_params['OUTGOING_LONGWAVE']
	INCOMING_SHORTWAVE = meteo_params['INCOMING_SHORTWAVE']
	BBA = meteo_params['BBA']
	DENSITY = meteo_params['DENSITY']

	with open("inputs.yaml" , "r") as ymlfile:
        	inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
	z0 = inputs['CRUST_DEV']['Z0'] 
	z = inputs['CRUST_DEV']['HEIGHT']  
	rho = inputs['CRUST_DEV']['RHO_AIR']
	v = inputs['CRUST_DEV']['KIN_VIS']
	g = inputs['CRUST_DEV']['ACC_G']  
	k = inputs['CRUST_DEV']['VK_CST']
	Cp = inputs['CRUST_DEV']['CP']
	a = inputs['CRUST_DEV']['EMP_CST']
	epsilon = inputs['CRUST_DEV']['EPSILON']
	lbd = inputs['CRUST_DEV']['LAMBDA']
	sigma = inputs['CRUST_DEV']['SB_CONST']
    
	######### CALULATION FLUXES IN W M-2 or J S-1 M-2
	### RADIATIVE FLUX = contributing to melt
	radiative_flux_lw = - OUTGOING_LONGWAVE + INCOMING_LONGWAVE #- sigma * 0.97 * 273.15**4 
	radiative_flux_sw = BBA * INCOMING_SHORTWAVE

	### CONDUCTIVE FLUX = contributing to surf lowering
	conductive_flux = HEAT_CAP * FLUX_RAIN_SNOW_FALL * (T_RAIN_SNOW_FALL - 273.15) 
    
	### CONVECTIVE FLUX = contributing to surface lowering
	# initialize conv flux and length scale with a * z / L = 0 
	friction_velocity = k * WIND_SPEED / (m.log(z/z0))
	REN = friction_velocity * z0 / v 
	zt = m.exp(m.log(z0) + 0.317 - 0.565 * m.log(REN) - 0.183 * (m.log(REN)**2))
	ze = m.exp(m.log(z0) + 0.396 - 0.512 * m.log(REN) - 0.180 * (m.log(REN)**2))
	convective_flux = (rho * Cp * k**2 * (WIND_SPEED*(AIR_T_Z - AIR_T_0)) / 
                                       ((m.log(z/z0)) * 
                                        (m.log(z/zt))))
	L = (1 / convective_flux * rho * Cp * (AIR_T_Z - AIR_T_0) / ( g * k ) *
         (k * WIND_SPEED / (m.log(z/z0)))**3)
	# loop to converge to correct value for zt, we, L and conv flux
	error = 10
	while error > 0.001: 
       		# Schuster 2001 : tau = k * WIND_SPEED / (m.log(z/z0) + (a* z / L))
        	friction_velocity = k * WIND_SPEED / (m.log(z/z0) + (a * z / L))
        	REN = friction_velocity * z0 / v 
        	zt = m.exp(m.log(z0) + 0.317 - 0.565 * m.log(REN) - 0.183 * (m.log(REN)**2))
        	ze = m.exp(m.log(z0) + 0.396 - 0.512 * m.log(REN) - 0.180 * (m.log(REN)**2))
        	convective_flux = (rho * Cp * k**2 * (WIND_SPEED*(AIR_T_Z - AIR_T_0)) / 
                                       ((m.log(z/z0) + (a * z / L)) * 
                                        (m.log(z/zt) + (a * z / L))))
        	L_new = (1 / (convective_flux * g * k) * rho * Cp * (AIR_T_Z - AIR_T_0) *
         	(friction_velocity)**3)
        	error = L_new - L
        	L = L_new
	### LATENT FLUX = contributing to surface lowering
	latent_flux = rho * epsilon * lbd * k * k * WIND_SPEED*\
			(AIR_VAP_P_Z - AIR_VAP_P_0) /\
			(AIR_P_Z*((m.log(z/z0) + (a * z / L)) * (m.log(z/ze) + (a * z / L))))
	melt =  (radiative_flux_sw+latent_flux+convective_flux) * 60 / 334000 / DENSITY	
    
	return radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux, melt


for index, row in data_file.iterrows():
        radiative_flux_sw, radiative_flux_lw, conductive_flux,\
        convective_flux, latent_flux, melt = calculate_energy_fluxes(row)
        data_out.append(np.array([radiative_flux_sw, radiative_flux_lw, latent_flux, 
        convective_flux,conductive_flux,melt]).T)
        
data_out2d = np.vstack(data_out)
data_outdf = pd.DataFrame(data_out2d, columns=['sw', 'lw', 'latent', 'conv', 
'cond', 'melt']).to_csv('seb_data_no_weathering.csv')


