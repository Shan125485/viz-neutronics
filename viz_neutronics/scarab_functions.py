import numpy as np
from scipy.interpolate import LinearNDInterpolator
import json
import os
import matplotlib.pyplot as plot
import re
import scarabee as scrb
from viz_neutronics.plottingFunctions import readOutputs, findKeffMC, plotSpatialTallyMC, rmsError, visualiseQuarter




def writeSolverToJSON(solver, x_pin, y_pin, z_pin, x_assem, y_assem, z_assem, outputs_filepath : str):

    print('extracting pin power')
    pin_power, x_loc, y_loc = solver.pin_power(z_pin)
    print('Making dictionary')
    dictionary = { # all values are strings so that they get printed into a readable format
        'x_pin': x_pin.tolist(),
        'y_pin': y_pin.tolist(),
        'z_pin': z_pin.tolist(),
        'x_assem': x_assem.tolist(),
        'y_assem': y_assem.tolist(),
        'z_assem': z_assem.tolist(),
        'keff': solver.keff,
        'keff_tolerance': solver.keff_tolerance,
        'avg_flux': solver.avg_flux().tolist(),
        'avg_power': solver.avg_power().tolist(),
        'flux_pinwise': solver.flux(x_pin,y_pin,z_pin).tolist(),
        'flux_assembly_wise': solver.flux(x_assem,y_assem,z_assem).tolist(),
        'flux_tolerance': solver.flux_tolerance,
        'power_pinwise': solver.power(x_pin, y_pin, z_pin).tolist(),
        'power_assembly_wise': solver.power(x_assem, y_assem, z_assem).tolist(),
        'pin_power': pin_power.tolist(),
        'pin_xloc': x_loc.tolist(),
        'pin_yloc': y_loc.tolist(),
        'ngroups': solver.ngroups
    }
    print('dumping to json')

    with open(outputs_filepath + 'Solver_output.json', 'w') as f:
        json.dump(dictionary, f)


def readSolverFromJSON(outputFile : str, print_output=True):
  
    res = readOutputs(outputFile, print_output=print_output)

    print("Beginning some useful post-processing of Scarabée results:")

    flux_pinwise_dict = {}
    flux_assembly_wise_dict = {}
    avg_flux_dict = {}

    print(" Separating flux array into multigroup flux dictionary res.flux['flux0'] etc.")
    print(" Separating avg_flux array into multigroup flux dictionary res.avg_flux['avg_flux0'] etc.")
    for i in range(res.ngroups):
        key = 'flux_pinwise' + str(i)
        value = np.array(res.flux_pinwise)[i,:,:,0] #[i,:,:]
        flux_pinwise_dict[key] = value

        key_assem = 'flux_assembly_wise' + str(i)
        value_assem = np.array(res.flux_assembly_wise)[i,:,:,0] #[i,:,:]
        flux_assembly_wise_dict[key_assem] = value_assem

        key_avg = 'avg_flux' + str(i)
        value_avg = np.array(res.avg_flux)[i,:,:,0] #[i,:,:]
        avg_flux_dict[key_avg] = value_avg
    setattr(res, 'flux_pinwise', flux_pinwise_dict)
    setattr(res, 'flux_assembly_wise', flux_assembly_wise_dict)
    setattr(res, 'avg_flux', avg_flux_dict)

    avg_power = np.array(res.avg_power)
    mean_power = np.mean(avg_power)

    print(" Normalising the average power by its mean value, ", mean_power)

    avg_power /= mean_power # Normalise
    avg_power = avg_power[:,:,0]
    setattr(res, 'avg_power', avg_power)


    print(" Normalising pin power by its mean")
    print("  -->Set 0 power pins to NaN")
    pin_power = np.array(res.pin_power)[:,:,0]
    msk = np.where(pin_power == 0.)
    nmsk = np.where(pin_power != 0.)
    pin_power[msk] = np.nan

    avg = np.mean(pin_power[nmsk])
    pin_power /= avg
    setattr(res, 'pin_power', pin_power)

    setattr(res, 'power_pinwise', np.array(res.power_pinwise)[:,:,0])
    setattr(res, 'power_assembly_wise', np.array(res.power_assembly_wise)[:,:,0])


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

def plotFissRateCompareAssem_MOCMC(outputFileMC : str, binFileMOC : str, scarab_text_log : str, name='assembly'):
    """_summary_

    Args:
        outputFileMC (_type_): _description_
        binFileMOC (_type_): _description_
        scarab_text_log (_type_): _description_
        name (_type_, optional): _description_. Defaults to 'assembly'.
    """    """"""

    resScarab  = scrb.DiffusionData.load(binFileMOC) # read Scarabee output
   
    # keff comparison
    keff_SCONE, keff_scarab, keff_diff = compareKeff_MOCMC(outputFileMC, scarab_text_log)

    # Fission rate comparison
    tallyname = 'pinFiss'
    fissrateMC, fissrateMC_std = plotSpatialTallyMC(outputFileMC, tallyname, plotting=False)
    fissrateScarab = resScarab.form_factors
    diff = (fissrateScarab - fissrateMC) / fissrateMC 

    # Extract Metadata
    max_error = np.max(diff[~np.isnan(diff)]) # exclude nan values for max calculation but keep them in the plot for clear visualisation of gadolinia and guide tubes
    min_error = np.min(diff[~np.isnan(diff)])

    rms = rmsError(fissrateMC, fissrateScarab)  

    # plotting
    fig, ax = plot.subplots()
    val = ax.imshow(abs(diff), cmap='Reds', origin='lower')
    fig.colorbar(val, ax=ax).minorticks_on()
    
    fig.suptitle(name  + ': Scarabee MOC vs SCONE MC, '
    '(Abs) relative difference in fission rate. \nMax error={:.2%}, '
    'min error={:.2%}, \nRMSE={:.2%}. \n$k_{{MC}}={:.5f}$, '
    '$k_{{MOC}}={:.5f}$, diff={:.0f} pcm'.format(max_error, min_error, rms, keff_SCONE, keff_scarab, keff_diff))
    plot.savefig('outputs/' + 'compare_MOC_MC_' + tallyname + '_' + name +  '.svg')
    plot.close()



def compareKeff_MOCMC(outputFileMC : str, scarab_text_log : str):
    """Returns keff from SCONE output json, Scarabee log .text file, and calculates the difference in pcm 

    Args:
        outputFileMC (str): _description_
        scarab_text_log (str): _description_
d
    Returns:
        _type_: _description_
    """    
    keff_SCONE, keff_SCONE_std = findKeffMC(outputFileMC)
    keff_scarab = findKeff_fromFuelAssembly(scarab_text_log)
    keff_diff = (keff_scarab - keff_SCONE) * 1e5 # pcm
    return keff_SCONE, keff_scarab, keff_diff


def compareKeff_nodalMC(outputFileMC : str, outputJSONScarab : str):
    """Difference between keff from scarabée nodal code and SCONE MC

    Args:
        outputFileMC (str): SCONE output JSON file
        outputJSONScarab (str): Scarabée output JSON file (from nodal calculation)

    Returns:
        _type_: _description_
    """
    resScarab = readSolverFromJSON(outputJSONScarab)
    keff_Scarab = resScarab.keff
    keff_SCONE, keff_SCONE_std = findKeffMC(outputFileMC)
    diff_keff = (keff_Scarab - keff_SCONE) * 1e5  # in pcm
    return diff_keff

def plotDiffusionData(binFileMOC : str, scarab_text_log : str, param : str, name : str, plotting=True):
    resScarab  = scrb.DiffusionData.load(binFileMOC)
    value = getattr(resScarab, param)

    if plotting:

        keff_scarab = findKeff_fromFuelAssembly(scarab_text_log)

        # plotting
        fig, ax = plot.subplots()
        val = ax.imshow(value, cmap='magma', origin='lower')
        fig.colorbar(val, ax=ax).minorticks_on()

        fig.suptitle(name  + ': Scarabee MOC ' + param + '\n$k_{{MOC}}={:.5f}$'.format(keff_scarab))

        plot.savefig('outputs/' + name + '_' + param + '.png')
        plot.close()
    return value

def plotAllFromJSON(outputFileScarab : str, plotting=True):
    """A quick visualisation of all Scarabee output quantities from a JSON file before significant postprocessing

    Args:
        outputFileScarab (_type_): _description_
    """    """"""
    res = readSolverFromJSON(outputFileScarab)

    outputs_filepath = 'outputs/resPlots/'
    if not os.path.exists(outputs_filepath):
        os.makedirs(outputs_filepath)


    print('\nLogging: NOTE - plotting =', plotting)
    print('\nkeff:', res.keff)
    print('keff_tolerance:', res.keff_tolerance)
    print('flux_tolerance:', res.flux_tolerance)
    print('ngroups:', res.ngroups)
    print('x_pin dimensions:', len(res.x_pin), 'elements from', res.x_pin[0], 'to', res.x_pin[-1])
    print('y_pin dimensions:', len(res.y_pin), 'elements from', res.y_pin[0], 'to', res.y_pin[-1])
    print('z_pin dimensions:', len(res.z_pin), 'elements from', res.z_pin[0], 'to', res.z_pin[-1])
    print('x_assem dimensions:', len(res.x_assem), 'elements from', res.x_assem[0], 'to', res.x_assem[-1])
    print('y_assem dimensions:', len(res.y_assem), 'elements from', res.y_assem[0], 'to', res.y_assem[-1])
    print('z_assem dimensions:', len(res.z_assem), 'elements from', res.z_assem[0], 'to', res.z_assem[-1])
    print('pin_xloc dimensions:', len(res.pin_xloc), 'elements from', res.pin_xloc[0], 'to', res.pin_xloc[-1])
    print('pin_xloc dimensions:', len(res.pin_yloc), 'elements from', res.pin_xloc[0], 'to', res.pin_yloc[-1])

    print('avg_flux dimensions:')
    for key in res.avg_flux:
        value = res.avg_flux[key]
        saveFile = outputs_filepath + key + '.svg'
        if plotting:
            fig, ax = plot.subplots()
            val = ax.imshow(value, cmap='magma')
            fig.colorbar(val, ax=ax).minorticks_on()
            plot.title(key)
            plot.savefig(saveFile)
            plot.close()
        print('-->', key, value.shape, ':: plot saved to', saveFile)

    print('flux dimensions:')
    for key in res.flux_pinwise:
        # print('-->', key, res.flux[key].shape)

        value = res.flux_pinwise[key]
        saveFile = outputs_filepath  + key + '.svg'
        if plotting:
            fig, ax = plot.subplots()
            val = ax.imshow(value, cmap='magma')
            fig.colorbar(val, ax=ax).minorticks_on()
            plot.title(key)
            plot.savefig(saveFile)
            plot.close()
        print('-->', key, value.shape, ':: plot saved to', saveFile)

    for key in res.flux_assembly_wise:
        # print('-->', key, res.flux[key].shape)

        value = res.flux_assembly_wise[key]
        saveFile = outputs_filepath + key + '.svg'
        
        if plotting:
            fig, ax = plot.subplots()
            val = ax.imshow(value, cmap='magma')
            fig.colorbar(val, ax=ax).minorticks_on()
            plot.title(key)
            plot.savefig(saveFile)
            plot.close()
        print('-->', key, value.shape, ':: plot saved to', saveFile)

    if plotting:
        fig, ax = plot.subplots()
        val = ax.imshow(res.pin_power, cmap='magma')
        fig.colorbar(val, ax=ax).minorticks_on()
        plot.title("Pin Power Distribution")
        plot.savefig(outputs_filepath + 'pin_power.svg')
        plot.close()
    print('pin_power dimensions: ' + str(res.pin_power.shape) + ':: plot saved to ' + outputs_filepath + 'pin_power.svg')

    if plotting:
        fig, ax = plot.subplots()
        val = ax.imshow(res.power_pinwise, cmap='magma')
        fig.colorbar(val, ax=ax).minorticks_on()
        plot.title("Homogenised pinwise power distribution")
        plot.savefig(outputs_filepath + 'power_pinwise.svg')
        plot.close()
    print('power_pinwise dimensions: ' + str(res.power_pinwise.shape) + ':: plot saved to ' + outputs_filepath + 'power_pinwise.svg')

    if plotting:
        fig, ax = plot.subplots()
        val = ax.imshow(res.power_assembly_wise, cmap='magma')
        fig.colorbar(val, ax=ax).minorticks_on()
        plot.title("Homogenised assembly-wise power distribution")
        plot.savefig(outputs_filepath + 'power_assembly_wise.svg')
        plot.close()
    print('power_assembly_wise dimensions: ' + str(res.power_assembly_wise.shape) + ':: plot saved to ' + outputs_filepath + 'power_assembly_wise.svg')

    if plotting:
        fig, ax = plot.subplots()
        val = ax.imshow(res.avg_power, cmap='magma')
        fig.colorbar(val, ax=ax).minorticks_on()
        plot.title("Averaged power distribution")
        plot.savefig(outputs_filepath + 'avg_power.svg')
        plot.close()
    print('avg_power dimensions: ' + str(res.avg_power.shape) + ':: plot saved to ' + outputs_filepath + 'avg_power.svg')

    return res
    




def plotSpatialParamJSON(outputJSONScarab, paramName, normalise_by_mean='all', vmax=None, vmin=None, plotting=True, visualise_quarter=False, aspect_ratio=1, indices = None, mgID = None):
    """Plots quantity in Cartesian space along with the uncertainty

    Args:
        outputFileScarab (str):             Name of output MC file.
        paramName (str):                    Name of tally to be plotted, set by user in the MC input file.
        normalise_by_mean (str, optional):  Normalise the tally value to the mean value. Defaults to 'all'.
                                                If 'all', the mean of all values is used. 
                                                If 'non-zero', then only non-zero values are used to calculate the mean. 
                                                If None, no normalisation is applied.
        response_index (int, optional):     If multiple responses have been tallied for this clerk, use this index to specify which one to plot. Defaults to 0.
        plotting (Bool) :                   If True, plots and saves a figure. Otherwise just returns the values.
        indices :                           If None, then no slicing is applied. Otherwise, it should be a list of tuples containing the indices (as many tuples as there are dimensions. The slicing is applied before visualise_quarter)
    """
    print('\n#########\nLogging: plotting {} from scarabee file {} '.format(paramName, outputJSONScarab))
    resScarab = readSolverFromJSON(outputJSONScarab, print_output=True)

    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Check parameter type  
    if type(getattr(resScarab, paramName)) == dict:
        dataDict  = getattr(resScarab, paramName)
        attrFoundFlag = False
        for key in dataDict.keys():
            if key[-1] == str(int(mgID)):
                print('Parsing results to find match to mgID {}, {}'.format(mgID, key))
                attrFoundFlag = True
                value = dataDict[key]
        if attrFoundFlag:
            pass
        else:
            raise ValueError('Parameter {} with mgID {} for plotting was not found'.format(paramName, mgID))
        
    else: 
        value = np.array(getattr(resScarab, paramName))
        mgID = '' # set for plotting label
    std = np.zeros(value.shape)   # dummy variable as input to visualiseQuarter
    keff = resScarab.keff

    # apply slicing
    print('Apply slicing as specified in input')
    if indices == None:
        value = value
    elif indices is not None and not (isinstance(indices, list) and len(indices) == 2 and all(isinstance(t, tuple) and len(t) == 2 for t in indices)):
        raise ValueError("value must be None or a list of two 2-element tuples")
    else:
        print('2D slicing applied according to indices', indices)
        value = value[indices[0][0]: indices[0][1], indices[1][0]: indices[1][1]]
        
    value, std, quarter_label = visualiseQuarter(value, std, visualise_quarter=visualise_quarter)

    # try to remove data from plot if the tally is 0
    value_plot = np.copy(np.where(value < 1e-18, np.nan, value))

    # now normalise by the mean value of non-zero tallies
    if normalise_by_mean=='non-zero':
        value_no_nan = value_plot[~np.isnan(value_plot)]
        value_mean = np.mean(value_no_nan)
        print('Normalising by the mean of non-zero values,', value_mean)
        value_plot = value_plot / value_mean
    elif normalise_by_mean=='all':
        value_temp = np.where(np.isnan(value_plot), 0, value_plot)
        print('Need to set NaNs to 0s again')
        mean = np.mean(value_temp)
        print('Normalising by the mean of all values,', mean)
        value_plot=value_plot/ mean
    elif normalise_by_mean==None:
        'No normalisation applied'
        value_plot = value_plot

    # Turn NaNs back into 0s for plotting    
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)
    
    if plotting:

        fig, ax1 = plot.subplots()
     
        val_plot = ax1.imshow(value_plot, cmap='magma', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
        fig.colorbar(val_plot, ax=ax1).minorticks_on()

        ax1.set_title('{} \nNormalised by the mean\n of {} values'.format(paramName+'_' + str(mgID), normalise_by_mean))

        fig.suptitle('Scarabee ' + paramName + ', $k_{{eff}}$={:.5f}'.format(keff) + '\n'  + quarter_label )
    
  
        plot.savefig(newpath + '/' + paramName + '_' + str(mgID) + '_nodal.svg')
        plot.close()

    return value_plot

def plotSpatialParam_CompareNodalMC(outputFileMC, outputFileNodal, tallyName_MC, paramNameNodal, normalise_by_mean='all',  response_index_MC = 0, vmax=None, vmin=None, plotting=True, visualise_quarter_MC=False, visualise_quarter_nodal=False, aspect_ratio=1, indicesNodal = None, mgID_nodal = None, annotate=False):
    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    value_MC, std_MC = plotSpatialTallyMC(outputFileMC, tallyName=tallyName_MC, normalise_by_mean=normalise_by_mean, response_index=response_index_MC, visualise_quarter=visualise_quarter_MC, plotting=False)
    value_nodal = plotSpatialParamJSON(outputFileNodal, paramNameNodal, normalise_by_mean=normalise_by_mean, plotting=False, visualise_quarter=visualise_quarter_nodal, indices = indicesNodal, mgID = mgID_nodal)

    if mgID_nodal==None:
        mgID_nodal = '' # set for plotting label 
    
    # extract labels:
    dummy_value = np.ones((2,2))
    dummy_std = np.ones((2,2))
    dummy_value, dummy_std, quarter_label = visualiseQuarter(dummy_value, dummy_std, visualise_quarter=visualise_quarter_MC) # TEMP this is only for MC!!!!

    # extract keff values from the outputs

    keff_MC, stdKeffMC = findKeffMC(outputFileMC)
    keff_nodal = readSolverFromJSON(outputFileNodal).keff

   

    
    # Calculate quantities
    rel_diff = (value_nodal - value_MC) / value_MC
    # rel_diff_unc = np.sqrt((std_RR/value_MC**2)**2 +(std_MC * value_RR/value_MC**2) ** 2)

    # calculate statistics

    rel_diff = np.copy(np.where(np.isnan(value_MC), 0, rel_diff))   # deal with nans in the original arrays


    rel_diff_for_max = np.where(value_MC< 1e-16,0, rel_diff)    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    max_error = np.max(rel_diff_for_max)

    min_error = np.min(rel_diff_for_max)


    rmse = rmsError(value_MC, value_nodal, relative_to='max_value')


    # Turn NaNs back into 0s for plotting    
    rel_diff = np.where(np.isnan(rel_diff), 0, rel_diff)


    plot.rcParams['figure.constrained_layout.use'] = True

    if plotting:

        fig, ax1 = plot.subplots()
        # check dimension:


        val_plot = ax1.imshow(rel_diff *100, cmap='rainbow', origin='lower', aspect=aspect_ratio)

        fig.colorbar(val_plot, ax=ax1, label='%').minorticks_on()

        if annotate==True:
            for (j,i),label in np.ndenumerate(rel_diff):
                ax1.text(i,j,'{:.2f}'.format(label*100),ha='center',va='center', fontsize=4, color='white')
                # ax1.annotate('{:.0%}'.format(label), [i,j], ha='center',va='center', fontsize=5.5, color='white')



        ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))

      

        fig.suptitle('(Nodal -MC) / MC: relative difference in {}. \nMax error={:.3%}, min error={:.3%}, \nRMSE={:.3%}.\n$k_{{MC}}={:.5f}$, $k_{{nodal}}={:.5f}$, diff={:.0f} pcm'.format(paramNameNodal, max_error, min_error, rmse, keff_MC, keff_nodal, 1e5*(keff_nodal-keff_MC)) + '\n'  + quarter_label  )

  
        plot.savefig(newpath + '/compare_MCNodal' + paramNameNodal + str(mgID_nodal) + '.svg')