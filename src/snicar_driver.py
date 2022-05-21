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

#%%

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
display_out_data(outputs1)

#%%
###########################
# CRUST DEV
###########################

#5200000 dust
#algae 29440

# the initial crust structure should be updated in the inputs.yaml file
# from the SNICAR inversion before running this code.
# then calculate nb of bubbles in the crust from initial crust structure:
nb_bbl = get_nb_bbl()
ablation = 0
albedo_array = np.array(([np.zeros(480)]*6))
#wvl= np.arange(0.205,5,0.01)

# fetch meteorological params throughout the day
data_file = pd.read_csv("./src/crust_dev_params.csv")

for index, row in data_file.iterrows():
    
    ##### CALL SNICAR
    # build classes from inputs.yaml file and validate their contents
    # (!!) path to illumination files should be changed to read SW radiation 
    # (!!) measured in the field instead
    (ice, illumination, rt_config, model_config,plot_config,impurities,
    ) = setup_snicar(input_file)
    
    status = validate_inputs(ice, rt_config, model_config, illumination, 
                            impurities)

    # now get the optical properties of the ice column
    ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
    tau, ssa, g, L_snw = mix_in_impurities(
    ssa_snw, g_snw, mac_snw, ice, impurities, model_config
    )
    
    # now run the AD solver to get albedo and associated variables
    outputs = adding_doubling_solver(tau, ssa, g, L_snw, ice, 
                                    illumination, model_config)
    albedo_array[index,:]=outputs.albedo
    
    ###### CALL CRUST MODEL
    # re_calculate density, bbl size and dz from energy inputs and 
    # ice conditions that are in the yaml file
    # ! this func should take meteorological data as inputs and change 
    # the ice params in the yaml file !
    radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux,radiative_flux_sw_spectral = calculate_energy_fluxes(row, outputs) 
    
    abl = update_snicar_parameters(radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux,
                                   radiative_flux_sw_spectral,row, nb_bbl)
    ablation = ablation + abl
    update_albedo(outputs) 
    #albedo[:,index] = outputs.albedo
    
    plot_albedo(plot_config, model_config, outputs.albedo)
    
    #display_out_data(outputs1)

    
    
#%%
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
    sigma = inputs['CRUST_DEV']['SB_CONST']
    
    ######### CALULATION FLUXES IN W M-2 or J S-1 M-2
    ### RADIATIVE FLUX = contributing to melt
    radiative_flux_sw = outputs.absorbed_flux_per_layer[1] 
    radiative_flux_sw_spectral = outputs.absorbed_spectral_flux_per_layer[:,1] 
    radiative_flux_lw = - sigma * 0.97 * 273.15**4 + INCOMING_LONGWAVE
    
    ### CONDUCTIVE FLUX = contributing to surf lowering
    conductive_flux = HEAT_CAP * FLUX_RAIN_SNOW_FALL * (T_RAIN_SNOW_FALL - 273.15) + outputs.absorbed_flux_per_layer[0] 
    
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
    
    return radiative_flux_sw, radiative_flux_lw, conductive_flux, convective_flux, latent_flux,radiative_flux_sw_spectral

def get_nb_bbl():
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    density = inputs['ICE']['RHO'][1]
    bbl_size = inputs['ICE']['RDS'][1]
    nb_bbl = (917 - density) / 917 / (4/3 * m.pi * (bbl_size * 10**(-6))**3)
    return nb_bbl

    


def update_snicar_parameters(radiative_flux_sw, radiative_flux_lw, conductive_flux, 
                             convective_flux, latent_flux,
                             radiative_flux_sw_spectral,meteo_params, nb_bbl):
    
    with open("./src/inputs.yaml" , "r") as ymlfile:
            inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    density = inputs['ICE']['RHO'][1]
    bbl_size = inputs['ICE']['RDS'][1]
    dz = inputs['ICE']['DZ'][1]
    
    # Water table 3% lower each hour
    new_dz = dz/0.97
    
    # Ice porosity (Cooper et al. 2018)
    porosity = -0.97*density/1000 + 0.89
     
    # Surface flux from shortwave radiation
    k = 1/dz * outputs.flx_extinction
    integral =  (-1/k * np.exp(-k*0.005) + 1/k) / (-1/k * np.exp(-k*dz)+ 1/k)
    surface_sw_flux = np.sum(radiative_flux_sw_spectral * integral)
    # Total surface melt
    surface_fluxes = latent_flux + conductive_flux + convective_flux + radiative_flux_lw + surface_sw_flux
    surface_melt = surface_fluxes * 3.600 / 334  # kg ice per m2
    # Volumic mass of ice ablated percolating down
    mass_volumic_pecolating = porosity * surface_melt / dz # kg ice per m3
    mass_ablated = (1-porosity) * surface_melt # kg ice per m2
     # Volumic mass of ice internally melted (kg m-3)
    ## J s-1 m-2 to J h-1 m-2 divided by J kg-1 yields kg h-1 m-2
    ## then divided by dz gives the kg melt per hour per m3 
    ## this melt is corrected for the amount remaining in the 
    ## ice using Cooper et al. 2017 equation for eff porosity
    ## to calculate the effective meltwater evacuated
    ## then this is subtracted to old density to get new one
    mass_volumic_internal_evacuated = porosity * ((radiative_flux_sw - surface_sw_flux) * 3600 / (334*10**3) / (dz-0.005))
    
    # Calculate new density from mass of ice removed and added
    new_density = density - mass_volumic_internal_evacuated + mass_volumic_pecolating
    
    # Calculate new water table from mass of ice ablated
    ablation = mass_ablated / density # m3 removed per m2
    new_dz = new_dz - ablation
    
    # then the amount of ice that is lost is lost at the interface of the 
    # bubbles, so the new bbl size is calculated from removing ice around:
    # nb_bbl per m3 = vol_air m3 per m3 / vol_bbl
    # volume lost per bubble: 
    # mass lost per m3 / density of ice kg m3 = m3 lost per m3
    # / nb_bbl per m3 = m3 lost / bbl converted to um3 / m3
    vol_gained_per_bubble = (mass_volumic_internal_evacuated-mass_volumic_pecolating) / 917 / nb_bbl *10**(18)
    # vol gained per bubble also writes: 
    # v = -4/3 * pi * old_radius**3 + 4/3 * pi * new radius**3
    # thus new radius writes: 
    new_bbl_size = round( 
                                ((vol_gained_per_bubble / (4/3 * m.pi) +
                                 bbl_size**3)**(1/3))/500
                              )*500

    # increases when density decreases bc melt happens at 
    # multiple scattering within bubbles
    #file_ice = str("./Data/OP_data/480band/bubbly_ice_files/bbl_{}.nc").format(new_bbl_size_calc)
    
    #if os.path.isfile(file_ice):
    #    new_bbl_size = new_bbl_size_calc
    #else: 
    #    new_bbl_size = bbl_size
    
    if new_density < 350: #if too low, collapse
    # the density increases to (?) and  
    # the new dz is calculated from mass balance
        new_density = 600
        new_dz = density * dz / new_density
    if new_density > 890: # if too high, stays at 915 and dz needs to be changed
    # but thats ambiguous bc can be due to percolation or surf melt... 
        new_density = 890
    if new_dz <0:
        new_density=890
        new_dz=0.1
        
    
    # UPDATE DENSITY, BBL SIZE, DEPTH FROM THE DIFFERENT FLUXES

    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['ICE']['RHO'][1] = float(round(new_density, 0))
    output['ICE']['RDS'][0] = new_bbl_size
    output['ICE']['RDS'][1] = new_bbl_size
    output['ICE']['DZ'][1] = float(round(new_dz, 4))
    output['RTM']['SOLZEN'] = int(meteo_params['SOLZEN'])
    output['RTM']['DIRECT'] = int(meteo_params['DIRECT'])
    with open('./src/inputs.yaml', 'w') as f:
        yml.dump(output, f)
    
    return ablation
    

def update_albedo(outputs):
    yml = YAML()
    yml.preserve_quotes = True
    yml.boolean_representation = ['False', 'True']
    output = yml.load(open("./src/inputs.yaml"))
    output['CRUST_DEV']['BBA'] = float(round(outputs.BBA, 5))
    with open('./src/inputs.yaml', 'w') as f:
        yml.dump(output, f)
        
        
        
        
        
    