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

#outputs2 = toon_solver(tau, ssa, g, L_snw, ice, illumination, model_config, rt_config)

# plot and print output data
plot_albedo(plot_config, model_config, outputs1.albedo)
print(outputs1.absorbed_flux_per_layer)
display_out_data(outputs1)


#%%
###########################
# CRUST DEV
###########################

# meteorological params throughout the day
data_file = pd.read_csv("./src/crust_dev_params.csv")
nb_bbl = get_nb_bbl()
#albedo = np.ndarray(shape=(480,2))

for index, row in data_file.iterrows():
    
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
    
    ###### CALL CRUST MODEL
    # re_calculate density, bbl size and dz from energy inputs and 
    # ice conditions that are in the yaml file
    # ! this func should take meteorological data as inputs and change 
    # the ice params in the yaml file !
    radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux = calculate_energy_fluxes(row, outputs) 
    
    update_snicar_parameters(radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux, row, nb_bbl)

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

def calculate_energy_fluxes(meteo_params, outputs):
    
    AIR_T_0 = meteo_params['AIR_T_0'] + 273.15
    AIR_T_Z = meteo_params['AIR_T_Z'] + 273.15
    AIR_P_Z = meteo_params['AIR_P_Z']
    AIR_VAP_P_0 = meteo_params['AIR_VAP_P_0']
    AIR_VAP_P_Z = meteo_params['AIR_VAP_P_Z']
    WIND_SPEED = meteo_params['WIND_SPEED']
    HEAT_CAP = meteo_params['HEAT_CAP'] # water or snow
    T_RAIN_SNOW_FALL = meteo_params['T_RAIN_SNOW_FALL'] + 273.15
    FLUX_RAIN_SNOW_FALL = meteo_params['FLUX_RAIN_SNOW_FALL']
    #IRRADIANCE = meteo_params['IRRADIANCE'] this goes into snicar
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
    # BBA = inputs['CRUST_DEV']['BBA'] 
    sigma = inputs['CRUST_DEV']['SB_CONST']
    
    ######### CALULATION FLUXES IN W M-1 or J S-1 M-2
    ### RADIATIVE FLUX = contributing to melt
    radiative_flux_sw = outputs.absorbed_flux_per_layer[1] 
    radiative_flux_lw = - sigma * 0.97 * 273.15**4 + INCOMING_LONGWAVE
    
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
    
    latent_flux =  rho * epsilon * lbd * k * k * WIND_SPEED*\
                    (AIR_VAP_P_Z - AIR_VAP_P_0) /\
                    (AIR_P_Z*((m.log(z/z0) + (a * z / L)) * (m.log(z/ze) + (a * z / L))))
    
    return radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux

def get_nb_bbl():
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    density = inputs['ICE']['RHO'][1]
    bbl_size = inputs['ICE']['RDS'][1]
    nb_bbl = (917 - density) / 917 / (4/3 * m.pi * (bbl_size * 10**(-6))**3)
    return nb_bbl


def update_snicar_parameters(radiative_flux_sw, radiative_flux_lw, conductive_flux, 
                             convective_flux, latent_flux, meteo_params, nb_bbl):
    
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    density = inputs['ICE']['RHO'][1]
    bbl_size = inputs['ICE']['RDS'][1]
    bbl_size_calc = inputs['ICE']['RDS_CALC']
    dz = inputs['ICE']['DZ'][1]

    # # CALCULATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES
    # kJ h-1 m-2 kJ -1 kg --> kg h-1 m-2 then divided by dz
    # this gives the kg melt per hour per m3 
    # this melt is corrected for the amount remaining in the 
    # ice using Cooper et al. 2017 equation for eff porosity
    # to calculate the effective meltwater evacuated
    # then this is subtracted to old density to get new one
    volumic_melt = (radiative_flux_sw * 3.600 / 334 / dz)
    porosity = -0.97*density/1000 + 0.89
    new_density = density - volumic_melt*porosity
    # then the amount of ice that is lost is lost at the interface of the 
    # bubbles, so the new bbl size is calculated from removing ice around:
    # nb_bbl per m3 = vol_air m3 per m3 / vol_bbl
    # volume lost per bubble: 
    # mass lost per m3 / density of ice kg m3 = m3 lost per m3
    # / nb_bbl per m3 = m3 lost / bbl
    vol_gained_per_bubble = (radiative_flux_sw * 3.600 / 334 / dz) / 917 / nb_bbl

    # vol ice lost per bubble also writes: 
    # v = -4/3 * pi * old_radius**3 + 4/3 * pi * new radius**3
    # thus new radius writes: 
    new_bbl_size_calc = round((vol_gained_per_bubble / (4/3 * m.pi) + (bbl_size_calc*10**(-6))**3)**(1/3)*10**(6)/100)*100

    # this stays at 10cm for now - need to be re-thought
    # -> tricky to modulate this in function of extinction coeff bc 
    # depth becomes too high for high density... 
    new_dz = 0.1 
    
    # increases when density decreases bc melt happens at 
    # multiple scattering within bubbles
    file_ice = str("./Data/OP_data/480band/bubbly_ice_files/bbl_{}.nc").format(new_bbl_size_calc)
    if os.path.isfile(file_ice):
        new_bbl_size = new_bbl_size_calc
    else: 
        new_bbl_size = bbl_size
    
    if new_density < 350: #if too low, collapse
    # the density increases to (?) and  
    # the new dz is calculated from mass balance
        new_density = 600
        new_dz = density * dz / new_density
    if new_density > 890: # if too high, stays at 890 but dz decreases due to melt
        new_density = 890
        new_dz = 0.9 * dz
    
    # UPDATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES

    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['ICE']['RHO'][1] = float(round(new_density, 0))
    output['ICE']['RDS'][0] = new_bbl_size
    output['ICE']['RDS'][1] = new_bbl_size
    output['ICE']['RDS_CALC'] = new_bbl_size_calc
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
        
        
        
        
        
    