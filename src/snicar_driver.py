#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("./src")
from setup_snicar import *
from classes import *
from column_OPs import *
from biooptical_funcs import *
from toon_rt_solver import toon_solver
from adding_doubling_solver import adding_doubling_solver
from validate_inputs import *
from display import *


# define input file
input_file = "./src/inputs.yaml"

###################
# BIO-OPTICAL MODEL
###################

# optionally run the bio-optical model to add new impurity optical properties to
# the BioSNICAR database. Commented out by default as we expect our default lap
# database to be sufficient for most users.

# run_biooptical_model(input_file)

###########################
# RADIATIVE TRANSFER MODEL
###########################

# first build classes from config file and validate their contents
(
    ice,
    illumination,
    rt_config,
    model_config,
    plot_config,
    impurities,
) = setup_snicar(input_file)

# validate inputs to ensure no invalid combinations have been chosen
status = validate_inputs(ice, rt_config, model_config, illumination, impurities)

# now get the optical properties of the ice column
ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
tau, ssa, g, L_snw = mix_in_impurities(
    ssa_snw, g_snw, mac_snw, ice, impurities, model_config
)

# now run one or both of the radiative transfer solvers
outputs1 = adding_doubling_solver(tau, ssa, g, L_snw, ice, illumination, model_config)

outputs2 = toon_solver(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config)

# plot and print output data
plot_albedo(plot_config, model_config, outputs1.albedo)
display_out_data(outputs1)


#%%
###########################
# CRUST DEV
###########################

# meteorological params throughout the day
data_file = pd.read_csv("./src/crust_dev_params.csv")
#albedo = np.ndarray(shape=(480,2))
for index, row in data_file.iterrows():
    ###### CALL CRUST MODEL
    # re_calculate density, bbl size and dz from energy inputs and 
    # ice conditions that are in the yaml file
    # ! this func should take meteorological data as inputs and change 
    # the ice params in the yaml file !
    radiative_flux, conductive_flux, convective_flux, latent_flux = calculate_energy_fluxes(row) 
    update_snicar_parameters(radiative_flux, conductive_flux, convective_flux, latent_flux, row)

    ##### CALL SNICAR
    # build classes from new inputs.yaml file and validate their contents
    (ice, illumination, rt_config, model_config,plot_config,impurities,
    ) = setup_snicar(input_file)
    status = validate_inputs(ice, rt_config, model_config, illumination, 
                            impurities)

    # now get the optical properties of the ice column
    ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
    tau, ssa, g, L_snw = mix_in_impurities(
    ssa_snw, g_snw, mac_snw, ice, impurities, model_config
    )
    # now generate and plot outputs
    # can we update the path to irradiance to the irradiance measured from ASD?
    outputs = adding_doubling_solver(tau, ssa, g, L_snw, ice, 
                                    illumination, model_config)

    update_albedo(outputs) 
    #albedo[:,index] = outputs.albedo
    plot_albedo(plot_config, model_config, outputs.albedo)
    
    #display_out_data(outputs1)

    
    
#%%
# fetch initial conditions in the yaml file (in functions afterwards)
# --> NEED TO CREATE A "CRUST_DV" BLOCK
# !! this need to include initial conditions for radiative forcing, 
# all snicar params and meterological params
input_file = "./src/inputs.yaml" 
import numpy as np
import yaml
import math as m
from ruamel.yaml import YAML

def calculate_energy_fluxes(meteo_params):
    
    AIR_T_0 = meteo_params['AIR_T_0'] + 273.15
    AIR_T_Z = meteo_params['AIR_T_Z'] + 273.15
    AIR_P_Z = meteo_params['AIR_P_Z']
    AIR_VAP_P_0 = meteo_params['AIR_VAP_P_0']
    AIR_VAP_P_Z = meteo_params['AIR_VAP_P_Z']
    WIND_SPEED = meteo_params['WIND_SPEED']
    HEAT_CAP = meteo_params['HEAT_CAP'] # water or snow
    T_RAIN_SNOW_FALL = meteo_params['T_RAIN_SNOW_FALL'] + 273.15
    FLUX_RAIN_SNOW_FALL = meteo_params['FLUX_RAIN_SNOW_FALL']
    IRRADIANCE = meteo_params['IRRADIANCE'] 
    INCOMING_LONGWAVE = meteo_params['INCOMING_LONGWAVE'] 

    with open("./src/inputs.yaml" , "r") as ymlfile:
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
    BBA = inputs['CRUST_DEV']['BBA']
    sigma = inputs['CRUST_DEV']['SB_CONST']
    
    ######### CALULATION FLUXES IN W M-1 or J S-1 M-2
    ### RADIATIVE FLUX = contributing to melt
    radiative_flux = BBA * IRRADIANCE - sigma * 0.97 * 273.15**4 + INCOMING_LONGWAVE
    
    ### CONDUCTIVE FLUX = contributing to surf lowering
    conductive_flux = HEAT_CAP * FLUX_RAIN_SNOW_FALL * (T_RAIN_SNOW_FALL - 273.15) 
    
    ### CONVECTIVE FLUX = contributing to surface lowering
    zt = m.exp(m.log(z0) + 0.317 - 0.565 * m.log(z0/v) - 0.183 * (m.log(z0/v)**2))
    
    convective_flux = (rho * Cp * k**2 * (WIND_SPEED*(AIR_T_Z - AIR_T_0)) / 
                                       ((m.log(z/z0)) * 
                                        (m.log(z/zt))))
    L = (1 / convective_flux * rho * Cp * (AIR_T_Z - AIR_T_0) / ( g * k ) *
         (k * WIND_SPEED / (m.log(z/z0)))**3)
    error = 10
    while error > 0.001: 
        L_new = (1 / convective_flux * rho * Cp * (AIR_T_Z - AIR_T_0) / ( g * k ) *
         (k * WIND_SPEED / (m.log(z/z0) + (a* z / L)))**3)
        convective_flux = (rho * Cp * k**2 * (WIND_SPEED*(AIR_T_Z - AIR_T_0)) / 
                                       ((m.log(z/z0) + (a * z / L)) * 
                                        (m.log(z/zt) + (a * z / L))))
        error = L_new - L
        L = L_new

    ### LATENT FLUX = contributing to surface lowering
    ze = m.exp(m.log(z0) + 0.396 - 0.512 * m.log(z0/v) - 0.180 * (m.log(z0/v)**2))
    latent_flux =  rho * epsilon * lbd * k * k * WIND_SPEED*\
                    (AIR_VAP_P_Z - AIR_VAP_P_0) /\
                    (AIR_P_Z*((m.log(z/z0) + (a * z / L)) * (m.log(z/ze) + (a * z / L))))
    
    return radiative_flux, conductive_flux, convective_flux, latent_flux

def update_snicar_parameters(radiative_flux, conductive_flux, 
                             convective_flux, latent_flux, meteo_params):
    
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    density = inputs['ICE']['RHO'][1]
    bbl_size = inputs['ICE']['RDS'][1]
    dz = inputs['ICE']['DZ'][1]
    
    # # CALCULATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES
    # kJ h-1 m-2 kJ -1 kg --> kg h-1 m-2 then divided by dz
    # this gives the kg lost per hour per m3 
    # then this is subtracted to old density to get new one
    new_density = density - (radiative_flux * 3.600 / 334 / dz)
    new_bbl_size = 5000
    new_dz = 0.1
    if new_density < 300:
        new_density = 890
        new_dz = 0.1
    if new_density > 916:
        new_density = 916
    
    # UPDATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES

    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['ICE']['RHO'][1] = float(round(new_density, 0))
    output['ICE']['RDS'][0] = new_bbl_size
    output['ICE']['RDS'][1] = new_bbl_size
    output['ICE']['DZ'][1] = new_dz
    output['RTM']['SZA'] = int(meteo_params['SZA'])
    output['RTM']['DIRECT'] = int(meteo_params['DIRECT'])
    with open('./src/inputs.yaml', 'w') as f:
        yml.dump(output, f)

def update_albedo(outputs):
    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['CRUST_DEV']['BBA'] = float(round(outputs.BBA, 5))
    with open('./src/inputs.yaml', 'w') as f:
        yml.dump(output, f)
        
        
        
        
        
    