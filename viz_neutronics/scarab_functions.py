import numpy as np
import json
import re
import scarabee as scrb
from viz_neutronics.plottingFunctions import readOutputs




def writeSolverToJSON(solver, x, y, z, outputs_filepath):

    pin_power, x_loc, y_loc = solver.pin_power(z)

    dictionary = { # all values are strings so that they get printed into a readable format
        'x': x.tolist(),
        'y': y.tolist(),
        'z': z.tolist(),
        'keff': solver.keff,
        'keff_tolerance': solver.keff_tolerance,
        'avg_flux': solver.avg_flux().tolist(),
        'avg_power': solver.avg_power().tolist(),
        'flux': solver.flux(x,y,z).tolist(),
        'flux_tolerance': solver.flux_tolerance,
        'power_homog': solver.power(x, y, z).tolist(),
        'pin_power': pin_power.tolist(),
        'pin_xloc': x_loc.tolist(),
        'pin_yloc': y_loc.tolist(),
        'ngroups': solver.ngroups
    }

    with open(outputs_filepath + 'EPR_output.json', 'w') as f:
        json.dump(dictionary, f, indent=4)


def readSolverFromJSON(outputFile):
  
    res = readOutputs('outputs/' + outputFile)

    print("Beginning some useful post-processing of Scarabée results:")

    flux_dict = {}

    print(" Separating flux array into multigroup flux dictionary res.flux['flux0'] etc.")
    for i in range(res.ngroups):
        key = 'flux' + str(i)
        value = np.array(res.flux)[i,:,:,0] #[i,:,:]
        flux_dict[key] = value
    setattr(res, 'flux', flux_dict)

    avg_power = np.array(res.avg_power)
    mean_power = np.mean(avg_power)

    print(" Normalising the average power by its mean value, ", mean_power)

    avg_power /= mean_power # Normalise
    avg_power = avg_power[:,:,0]
    setattr(res, 'avg_power', avg_power)


    # pin_power, x_loc, y_loc = solver.pin_power(z)
    print(" Normalising pin power")
    print("  -->Set 0 power pins to NaN")
    pin_power = np.array(res.pin_power)[:,:,0]
    msk = np.where(pin_power == 0.)
    nmsk = np.where(pin_power != 0.)
    pin_power[msk] = np.nan

    avg = np.mean(pin_power[nmsk])
    pin_power /= avg
    setattr(res, 'pin_power', pin_power)

    setattr(res, 'power_homog', np.array(res.power_homog)[:,:,0])


    return res




def transXS_inscatter(hom_XS : scrb.CrossSection, num_groups : int):
    """Takes Scarabée homogenised cross section and number of groups as input, 
    extracts the inscatter transport cross section for all groups and makes a ndarray.

    Args:
        hom_XS (scrb.CrossSection): Homogenised cross sections set
        num_groups (int): Number of energy groups
    """


    Etr_inscatter = []
    for energy_group in range(num_groups):
        Etr_inscatter.append(hom_XS.Etr(energy_group))
    Etr_inscatter = np.array(Etr_inscatter)
    return Etr_inscatter

def transXS_flux_limited(hom_XS : scrb.CrossSection, hom_flux : np.ndarray[np.float64]):
    """Takes Scarabée homogenised cross section and flux as input, returns a flux-limited transport cross section.

    Args:
        hom_XS (scrb.CrossSection): Homogenised cross sections set
        hom_flux (np.ndarray[np.float64]): Homogenised flux array
    """

    ngroups = hom_flux.shape[0]
    s1_array = np.zeros((ngroups, ngroups))
    product = np.zeros(ngroups)
    delta_tr_array = np.zeros(ngroups)
    Et_array = np.zeros(ngroups)

    for energy_out in range(ngroups):
        product[energy_out] = 0
        Et_array[energy_out] = hom_XS.Et(energy_out)
        for energy_in in range(ngroups):
            s1_array[energy_in,energy_out] = hom_XS.Es(1,energy_in,energy_out)
            product[energy_out] += s1_array[energy_in,energy_out] * hom_flux[energy_in]
        delta_tr_array[energy_out] = np.sum(product[energy_out]) / hom_flux[energy_out]
        
    Etr_fl_array = Et_array - delta_tr_array
    return Etr_fl_array


def transXS_outscatter(hom_XS : scrb.CrossSection, hom_flux : np.ndarray[np.float64]):
    """Takes Scarabée homogenised cross section and flux as input, returns an outscatter transport cross section.

    Args:
        hom_XS (scrb.CrossSection): Homogenised cross sections set
        hom_flux (np.ndarray[np.float64]): Homogenised flux array
    """

    ngroups = hom_flux.shape[0]
    s1_array = np.zeros((ngroups, ngroups))
    Et_array = np.zeros(ngroups)

    for energy_out in range(ngroups):
        Et_array[energy_out] = hom_XS.Et(energy_out)
        for energy_in in range(ngroups):
            s1_array[energy_in,energy_out] = hom_XS.Es(1,energy_in,energy_out)
      
    s1g_array = np.sum(s1_array,1)

    delta_tr_array = s1g_array

    Etr_os_array = Et_array - delta_tr_array
    
    return Etr_os_array


def display_material_isotopes(material : scrb.Material):
    print('\n' + material.name)
    for nuclide in material.composition.nuclides:
        print(nuclide.name, material.atom_density(nuclide.name))



def findKeff_fromFuelAssembly(scarab_text_log : str):
    """This works specifically for PWR assembly output log files. 
    Taking the second instance of keff seems to work, but 
    sometimes when running a full core, the individual fuel 
    assembly logs seem to get lost. I think the current set-up 
    in EPR core works well for now.

    Args:
        scarab_text_log (str): .txt file generated by Scarabee for a fuel assembly 

    Returns:
       float: The k-eff (or k-inf) for the fuel assembly
    """
    file = open(scarab_text_log, 'r')   # Read scarabee keff from text log
    keff_list = []
    counter=0
    for line in file:
        if re.search('Kinf', line):
            keff_list.append(float(line.split()[-1]))
            if counter==1: # take the second instance of kinf
                break
            else:
                counter+=1
    keff_scarab = keff_list[-1]  # take the last iteration value
    file.close()
    return keff_scarab

