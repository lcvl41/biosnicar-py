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
# plot_albedo(plot_config, model_config, outputs1.albedo)
# display_out_data(outputs1)


#%%
###########################
# CRUST DEV
###########################

# meteorological params throughout the day
data_file = '.csv' 

for meteo_params in data_file: 
    
    ###### CALL CRUST MODEL
    # re_calculate density, bbl size and dz from energy inputs and 
    # ice conditions that are in the yaml file
    # ! this func should take meteorological data as inputs and change 
    # the ice params in the yaml file !

    update_ice_parameters(meteo_params) 

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

    inputs['CRUST_DEV']['BBA'] = outputs.BBA

    plot_albedo(plot_config, model_config, outputs.albedo)

    display_out_data(outputs1)

    
    
#%%
# fetch initial conditions in the yaml file (in functions afterwards)
# --> NEED TO CREATE A "CRUST_DV" BLOCK
# !! this need to include initial conditions for radiative forcing, 
# all snicar params and meterological params
input_file = "./src/inputs.yaml" 
import numpy as np
import yaml
import math as m

def update_ice_parameters():
    meteo_params = np.array([0,10,1,3,2,5,0,0,0])
    AIR_T_0 = meteo_params[0] + 273.15
    AIR_T_Z = meteo_params[1] + 273.15
    AIR_P_Z = meteo_params[2]
    AIR_VAP_P_0 = meteo_params[3]
    AIR_VAP_P_Z = meteo_params[4]
    WIND_SPEED = meteo_params[5]
    HEAT_CAP = meteo_params[6] # water or snow
    T_RAIN_SNOW_FALL = meteo_params[7] + 273.15
    FLUX_RAIN_SNOW_FALL = meteo_params[8]
    z0 = 0.00246 # surface roughness from Schuster 2001
    z = 1 # height of sensor 
    rho = 1.2690 # air density, ideally as a function of T
    v = 13.72 * 10**6 # kinematic viscosity of air m2 s-1
    g = 0.9 # m s-2
    k = 0.41 # von Karman's cst
    Cp = 1010 # specific heat of air J kg-1 K-1 
    a = 5 # empirical correction (??)
    epsilon = 0.018016  / 0.0289652 # ratio molecular weight of water vapour to air
    lbd  = 250.1 # latent heat of vap kg kJ -1 at 0 deg C
    #import yaml
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
            
    BBA = inputs['CRUST_DEV']['BBA']
    #maybe the irradiance goes in the meteo params ?
    IRRADIANCE = inputs['CRUST_DEV']['IRRADIANCE'] 
    
    ### RADIATIVE FLUX 
    radiative_flux = BBA * IRRADIANCE
    
    ### CONDUCTIVE FLUX 
    conductive_flux = HEAT_CAP * FLUX_RAIN_SNOW_FALL * (T_RAIN_SNOW_FALL - 273.15)
    
    ### CONVECTIVE FLUX 
    zt = m.exp(m.log(z0) + 0.317 - 0.565 * m.log(z0/v) - 0.183 * (m.log(z0/v)**2))
    
    convective_flux = 1
    L = 5
    error = 10
    while error > 0.001: 
        L_new = (1 / convective_flux * rho * Cp * (AIR_T_Z - AIR_T_0) / ( g * k ) *
         (k * WIND_SPEED / (m.log(z/z0) + (a*z / L)))**3)
        convective_flux = (rho * Cp * k**2 * (WIND_SPEED*(AIR_T_Z - AIR_T_0)) / 
                                       ((m.log(z/z0) + (a * z / L)) * 
                                        (m.log(z/zt) + (a * z / L))))
        error = L_new - L
        L = L_new

    ### LATENT FLUX 
    ze = m.exp(m.log(z0) + 0.396 - 0.512 * m.log(z0/v) - 0.183 * (m.log(z0/v)**2))
    latent_flux =  (rho * epsilon * lbd * k**2 * 
                    (WIND_SPEED*(AIR_VAP_P_Z - AIR_VAP_P_0)) / 
                    AIR_P_Z*((m.log(z/z0) + (a * z / L)) * (m.log(z/ze) + (a * z / L))))
    
    # CALCULATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES
    density = 700 
    bbl_size = 5000
    dz = 0.08
    
    # UPDATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES

    from ruamel.yaml import YAML
    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['CRUST_DEV']['DENSITY'] = density
    output['CRUST_DEV']['BBL_GRAIN_SIZE'] = bbl_size
    output['CRUST_DEV']['DEPTH'] = dz
    with open('./src/new.yaml', 'w') as f:
        yml.dump(output, f)



    
    