import json
import numpy as np
import matplotlib.pyplot as plot
from itertools import cycle
import os

from viz_neutronics.input2json import parse_text_to_dict, save_to_json, stringTuple_to_array, dict2obj# run from outside module
# from input2json import parse_text_to_dict, save_to_json,  dict2obj # run from within module


# Global plot settings
plot.rcParams['figure.constrained_layout.use'] = True   # Better subplot spacing
plot.rcParams['axes.grid'] = False                       # Gridline
plot.rcParams['lines.linewidth'] = 1.5                # Line width. Default is 1.5       


# variables
lines = ["-","--","-.",":"]
colours = ['royalblue', 'orange', 'forestgreen', 'firebrick', 'goldenrod', 'darkviolet']
linecycler = cycle(lines)
colourcycler = cycle(colours)


def readInputs(inputFile): 
    # read in inputs
    print('Reading in input file', inputFile, 'as a dictionary')
    inputDict = parse_text_to_dict(inputFile)

    print('Saving input dictionary to input.json')
    save_to_json(inputDict, 'input.json')

    print('Input dictionary keys are:\n')
    for key in inputDict.keys():
        print('-->', key)
    print('\nConverting dictionaries into objects: inputs and outputs')
    inputs = dict2obj(inputDict)
    return inputs

def readOutputs(output_file, print_output=True):
    print('\n\nLoading {} into an output dictionary'.format(output_file))
    # returns output JSON object as python dictionary
    with open(output_file) as f:
        outputDict = json.load(f)

    if print_output:
        print('Output dictionary keys are:\n')
        for key in outputDict.keys():
            print('-->', key)
    
    outputs = dict2obj(outputDict)
    return outputs


def plotShannon(inputFile, outputFile):
    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    inputs = readInputs(inputFile)
    outputs = readOutputs(outputFile)
    shannonEntropy = outputs.inactive.shannon.shannonEntropy
    inactiveCycles = inputs.inactive 
    activeCycles = inputs.active

    fig, ax = plot.subplots()
    ax.plot(shannonEntropy[:inactiveCycles])

    ax.set_ylabel('Shannon entropy')
    ax.set_xlabel('Iteration')
    plot.title(str(inactiveCycles) + ' inactive cycles, ' + str(activeCycles) + ' active cycles')
    
    plot.savefig(newpath + '/Shannon_entropy.svg')


def plotScatteringMatrices(outputFile):

    outputs = readOutputs(outputFile)
    # plot P0 uncertainty colourmaps
    P0 = np.array(outputs.active.scatteringMatrices.P0)[:,0]
    P0_std = np.array(outputs.active.scatteringMatrices.P0)[:,1]

    numGroups = int(np.sqrt(P0.shape))

    P0 = np.reshape(P0, (numGroups, numGroups))
    P0_std = np.reshape(P0_std, (numGroups, numGroups))
    relUnc = np.where(P0_std==0, 0, P0_std / P0)

    fig, ((axP0, axStd), (axRelUnc, ax4)) = plot.subplots(2,2)
    ax4.set_axis_off()
    # ax1.imshow(P0, extent=[0, 1, 0, 1])
    P0_scale = axP0.imshow(P0)
    P0_std_scale = axStd.imshow(P0_std)
    P0_relUnc_scale = axRelUnc.imshow(relUnc)

    fig.colorbar(P0_relUnc_scale, ax=axRelUnc)
    fig.colorbar(P0_scale, ax=axP0)
    fig.colorbar(P0_std_scale, ax=axStd)
    # flux plot, separate into fast and thermal (1eV)

    axP0.set_title('P0')
    axStd.set_title('P0 std')
    axRelUnc.set_title('relative uncertainty')

    axP0.set_aspect('equal')
    axStd.set_aspect('equal')
    axRelUnc.set_aspect('equal')

    # fig.colorbar(P0)
    fig.suptitle("Slab with vacuum boundaries, {} groups".format(numGroups))


    plot.savefig('P0_colourmap.svg')


def plotCellPosRR(outputFileRR):
    """Plots the cell positions in the random ray output file. Useful for debugging.

    Args:
        outputFileRR (str): Name of the random ray output file.
    """
    outputs = readOutputs(outputFileRR)
    cellPos = np.array(outputs.position.position)
    print(cellPos[:,2].shape)
    fig, (axx, axy, axz) = plot.subplots(3,1)
    # ax.plot(cellPos[:,0], cellPos[:,1], 'o', markersize=1)
    i=11
    counter = np.arange(cellPos.shape[0]-i)
    
    axx.scatter(counter,cellPos[i:,0])
    axy.scatter(counter,cellPos[i:,1])
    axz.scatter(counter,cellPos[i:,2])
    # ax1.scatter(cellPos[:,2],cellPos[:,1])
    # ax.set_xlabel('x position')
    axx.set_ylabel('x position')
    axy.set_ylabel('y position')
    axz.set_ylabel('z position')
    fig.suptitle('Cell positions in random ray output file')

    plot.savefig('outputs/cell_positions.svg')

def plotDirectionalFluxRR(outputFileRR, direction):
    """Plots the right or left flux in the random ray output file. Useful for debugging.

    Args:
        outputFileRR (str): Name of the random ray output file.
    """
    if direction not in ['left', 'right']:
        raise ValueError("Direction must be either 'left' or 'right'.")
    if direction == 'right':
        outputs = readOutputs(outputFileRR)
        rightflux = np.array(outputs.right_flux.right_flux_g1)
        print('right flux shape:', rightflux.shape)

        fig, (ax_val, ax_std) = plot.subplots(2)
        ax_val.plot(rightflux[:,0])
        ax_std.plot(rightflux[:,1])
        ax_val.set_title('value')
        ax_std.set_title('std')

        fig.suptitle('right flux in random ray')

        plot.savefig('outputs/right_flux.svg')
    elif direction == 'left':
        outputs = readOutputs(outputFileRR)
        leftflux = np.array(outputs.left_flux.left_flux_g1)
        print('left flux shape:', leftflux.shape)

        fig, (ax_val, ax_std) = plot.subplots(2)
        ax_val.plot(leftflux[:,0])
        ax_std.plot(leftflux[:,1])
        ax_val.set_title('value')
        ax_std.set_title('std')

        fig.suptitle('left flux in random ray')

        plot.savefig('outputs/left_flux.svg')

def plotMultiMapTallyMC(outputFileMC, tallyName,response_index = 0, normalise_by_mean='all', plotting=True, mapOrder = [1,2,3], layout = 'horizontal', mode = 'line', aspect_ratio = 1):
    """Tries to plot up to three dimensions. mapOrder corresponds to the level of plotting. Dimension 1 is the x axis of line plots. 
    Dimension 2 will be lines shown on the same axes. Dimension 3 will define a new set of axes.

    Args:
        outputFileMC (_type_): _description_
        tallyName (_type_): _description_
        response_index (int, optional): _description_. Defaults to 0.
        normalise_by_mean (str, optional): _description_. Defaults to 'all'.
        plotting (bool, optional): _description_. Defaults to True.
        mapOrder (list, optional): _description_. Defaults to [''].
    """
    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    outputs = readOutputs(outputFileMC)

    result = np.array(getattr(outputs.active, tallyName).Res)
    mapNames = np.array(getattr(outputs.active, tallyName).multiMapOrder)
    value = result[...,response_index,0]
    std = result[...,response_index,1] / value

    # try to remove data from plot if the tally is 0
    value_plot = np.copy(np.where(value < 1e-18, np.nan, value))
    std_plot = np.copy(np.where(value < 1e-18, np.nan, std))

    # now normalise by the mean value of non-zero tallies
    if normalise_by_mean=='non-zero':
        value_no_nan = value_plot[~np.isnan(value_plot)]
        value_mean = np.mean(value_no_nan)
        print('Normalising by the mean of non-zero values,', value_mean)
        value_plot = value_plot / value_mean
    elif normalise_by_mean=='all':
        mean = np.mean(value)
        print('Normalising by the mean of all values,', mean)
        value_plot=value_plot/ mean
    elif normalise_by_mean==None:
        'No normalisation applied'
        value_plot = value_plot

    # Turn NaNs back into 0s for plotting    
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)
    std_plot = np.where(np.isnan(std_plot), 0, std_plot)
    
    if plotting:
        mapOrder = np.array(mapOrder)
        # reshape the array according to the user input. First index defines the number of figures, 2nd index the number of traces, 3rd index the length of x axis.
        value_plot = np.moveaxis(value_plot, [0,1,2], mapOrder)
        std_plot = np.moveaxis(std_plot, [0,1,2], mapOrder)
        mapNames = np.flip(mapNames)[mapOrder]

        num_figs = value_plot.shape[0]
        num_traces = value_plot.shape[1]

        plot.rcParams['figure.constrained_layout.use'] = True

        if mode == 'line':
            for i in range(0, num_figs):
                print('Plotting figure {:.0f} of {:.0f}'.format(i+1, num_figs))
                if layout == 'horizontal':
                    fig, (ax1, ax2) = plot.subplots(1,2)
                elif layout == 'vertical':
                    fig, (ax1, ax2) = plot.subplots(2,1)
                
                for j in range(0, num_traces):
                    style = next(linecycler)
                    colour = next(colourcycler)
                    print('-> Plotting trace {:.0f} of {:.0f}'.format(j+1, num_traces))
                    ax1.plot(value_plot[i,j], label=j+1, linestyle=style, color=colour)
                    ax2.plot(std_plot[i,j], linestyle=style, color=colour)

                ax1.set_xlim(left=0)
                ax2.set_xlim(left=0)

                ax1.set_ylabel(tallyName)
                ax2.set_ylabel('std')

                ax1.set_xlabel(mapNames[2])
                ax2.set_xlabel(mapNames[2])

              

                if layout == 'horizontal':
                    ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
                    ax2.set_title('Standard deviation\n(relative uncertainty)')

                elif layout == 'vertical':
                    ax1.set_title('Value - normalised by the mean of {} values'.format(normalise_by_mean))
                    ax2.set_title('Standard deviation (relative uncertainty)')

                plot.figlegend(title=mapNames[1], loc='lower right')
                fig.suptitle('Monte Carlo ' + tallyName + ' ' + mapNames[0] + str(i+1) )

       
                plot.savefig(newpath + '/' + tallyName + str(int(response_index)) + '_' + mapNames[0] + str(i+1) + '_MC.svg')
                
                
        
        elif mode == 'image':
            for i in range(0, num_figs):
                print('Plotting figure {:.0f} of {:.0f}'.format(i+1, num_figs))
                if layout == 'horizontal':
                    fig, (ax1, ax2) = plot.subplots(1,2)
                elif layout == 'vertical':
                    fig, (ax1, ax2) = plot.subplots(2,1)

                val_plot = ax1.imshow(value_plot[i], cmap='Blues', origin='lower', aspect=aspect_ratio)
                uncertainty_plot = ax2.imshow(std_plot[i], cmap='Reds', origin='lower', aspect=aspect_ratio)

                # Set axes labels
                ax1.set_ylabel(mapNames[1])
                ax2.set_ylabel(mapNames[1])

                ax1.set_xlabel(mapNames[2])
                ax2.set_xlabel(mapNames[2])


                cbar1 = fig.colorbar(val_plot, ax=ax1) #.minorticks_on()
                cbar1.ax.set_ylabel(tallyName, rotation=270)
                cbar1.minorticks_on()
                cbar2 = fig.colorbar(uncertainty_plot, ax=ax2) #.minorticks_on()
                cbar2.ax.set_ylabel('std', rotation=270)
                cbar2.minorticks_on()
                fig.suptitle('Monte Carlo ' + tallyName + ', for bin: ' + mapNames[0] + str(i+1) )

                if layout == 'horizontal':
                    ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
                    ax2.set_title('Standard deviation\n(relative uncertainty)')

                elif layout == 'vertical':
                    ax1.set_title('Value - normalised by the mean of {} values'.format(normalise_by_mean))
                    ax2.set_title('Standard deviation (relative uncertainty)')

     
                plot.savefig(newpath + '/' + tallyName + str(int(response_index)) + '_' + mapNames[0] + str(i+1) + '_MC.svg')

    return value_plot, std_plot


def plotSpatialTallyMC(outputFileMC, tallyName, normalise_by_mean='all', response_index = 0, vmax=None, vmin=None, vmax_unc=None, vmin_unc=None, plotting=True, visualise_quarter=False, aspect_ratio=1, layout = 'horizontal', remove_edges_2D=False):
    """Plots quantity in Cartesian space along with the uncertainty

    Args:
        outputFileMC (str): Name of output MC file.
        tallyName (str): Name of tally to be plotted, set by user in the MC input file.
        normalise_by_mean (str, optional): Normalise the tally value to the mean value. Defaults to 'all'.
                            If 'all', the mean of all values is used. 
                            If 'non-zero', then only non-zero values are used to calculate the mean. 
                            If None, no normalisation is applied.
        response_index (int, optional): If multiple responses have been tallied for this clerk, use this index to specify which one to plot. Defaults to 0.
        plotting (Bool) : If True, plots and saves a figure. Otherwise just returns the values.
    """

    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    outputs = readOutputs(outputFileMC, print_output=False)

    result = np.array(getattr(outputs.active, tallyName).Res)
    value = result[...,response_index,0]
    std = result[...,response_index,1] / value

    keff, stdKeff = findKeffMC(outputFileMC) # extract keff

    # symmetry check
    # print('MC value symmetry check:')
    # value_MC_flip = np.flip(value,0)
    # print(value_MC_flip)
    # print(100*(value_MC_flip - value_MC_flip.T)/value_MC_flip)


    # print('MC symmetry check:')
    # print(np.allclose(value_MC_flip, value_MC_flip.T, rtol=1e-3, atol=1e-3))

    value, std, removed_edges_label = removeEdges2D(value, std, remove_edges_2D=remove_edges_2D)
   
    value, std, quarter_label = visualiseQuarter(value, std, visualise_quarter=visualise_quarter)

   
    
    

    # try to remove data from plot if the tally is 0
    value_plot = np.copy(np.where(value < 1e-18, np.nan, value))
    std_plot = np.copy(np.where(value < 1e-18, np.nan, std))

    # now normalise by the mean value of non-zero tallies
    if normalise_by_mean=='non-zero':
        value_no_nan = value_plot[~np.isnan(value_plot)]
        value_mean = np.mean(value_no_nan)
        print('Normalising by the mean of non-zero values,', value_mean)
        value_plot = value_plot / value_mean
    elif normalise_by_mean=='all':
        mean = np.mean(value)
        print('Normalising by the mean of all values,', mean)
        value_plot=value_plot/ mean
    elif normalise_by_mean==None:
        'No normalisation applied'
        value_plot = value_plot

    # Turn NaNs back into 0s for plotting    
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)
    std_plot = np.where(np.isnan(std_plot), 0, std_plot)
    
    if plotting:
        if layout == 'horizontal':
            # fig, (ax1, ax2) = plot.subplots(1,2)
            fig, ax1 = plot.subplots() # TEMP
        elif layout == 'vertical':
            fig, (ax1, ax2) = plot.subplots(2,1)
        
        if value_plot.ndim < 2:
            # This is not 2D data, assume 1-dimensional.
            ax1.plot(value_plot, color='blue')
            ax2.plot(std_plot, color='red')

            ax1.set_xlim(left=0)
            ax2.set_xlim(left=0)

        else:

            # val_plot = ax1.imshow(value_plot, cmap='Blues', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
            # val_plot = ax1.imshow(value_plot, cmap='viridis', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
            val_plot = ax1.imshow(value_plot, cmap='magma', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
            # uncertainty_plot = ax2.imshow(std_plot, cmap='Reds', vmax=vmax_unc, vmin=vmin_unc, origin='lower', aspect=aspect_ratio)
            # uncertainty_plot = ax2.imshow(std_plot, cmap='Greens', vmax=vmax_unc, vmin=vmin_unc, origin='lower', aspect=aspect_ratio)
            fig.colorbar(val_plot, ax=ax1).minorticks_on()
            # fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

        # ax1.set_title('Fission rate') # TEMP
        ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
        # ax2.set_title('Standard deviation\n(relative uncertainty)')

        fig.suptitle('Monte Carlo ' + tallyName + ', $k_{{eff}}$={:.5f} +/- {:.0f} pcm'.format(keff, 1e5 * stdKeff) + '\n'  + quarter_label  + '\n'  + removed_edges_label )
    
  
        plot.savefig(newpath + '/' + tallyName + str(int(response_index)) + '_MC.svg')
        

    return value_plot, std_plot

def plotSpatialMaterialTallyMC(outputFileMC, tallyName, materialName, normalise_by_mean='all', response_index = 0, vmax=None, vmin=None, vmax_unc=None, vmin_unc=None):
    """Plots quantity in Cartesian space along with the uncertainty. Works when a spacemap is defined first, then material map, in the MC input file tally.

    Args:
        outputFileMC (str): Name of output MC file.
        tallyName (str): Name of tally to be plotted, set by user in the MC input file.
        normalise_by_mean (str, optional): Normalise the tally value to the mean value. Defaults to 'all'.
                            If 'all', the mean of all values is used. 
                            If 'non-zero', then only non-zero values are used to calculate the mean. 
                            If None, no normalisation is applied.
        response_index (int, optional): If multiple responses have been tallied for this clerk, use this index to specify which one to plot. Defaults to 0.
    """

    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    outputs = readOutputs(outputFileMC)
    
    # find index of material
    MaterialBins = np.array(getattr(outputs.active, tallyName).MaterialBins)
    mat_index = np.where(MaterialBins== materialName)
    print(mat_index)

    result = np.array(getattr(outputs.active, tallyName).Res)
    value = result[mat_index[0],...,response_index,0][0]
    std = result[mat_index[0],...,response_index,1][0] / value

    keff, stdKeff = findKeffMC(outputFileMC) # extract keff
    

    # try to remove data from plot if the tally is 0
    value_plot = np.copy(np.where(value < 1e-18, np.nan, value))
    std_plot = np.copy(np.where(value < 1e-18, np.nan, std))

    # now normalise by the mean value of non-zero tallies
    if normalise_by_mean=='non-zero':
        value_no_nan = value_plot[~np.isnan(value_plot)]
        value_mean = np.mean(value_no_nan)
        print('Normalising by the mean of non-zero values,', value_mean)
        value_plot = value_plot / value_mean
    elif normalise_by_mean=='all':
        mean = np.mean(value)
        print('Normalising by the mean of all values,', mean)
        value_plot=value_plot/ mean
    elif normalise_by_mean==None:
        'No normalisation applied'
        value_plot = value_plot
    
    # Turn NaNs back into 0s for plotting    
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)
    std_plot = np.where(np.isnan(std_plot), 0, std_plot)
    

    fig, (ax1, ax2) = plot.subplots(1,2)
    # val_plot = ax1.imshow(value_plot, cmap='hot', vmax=vmax, vmin=vmin)
    val_plot = ax1.imshow(value_plot, cmap='Blues', vmax=vmax, vmin=vmin, origin='lower')
    uncertainty_plot = ax2.imshow(std_plot, cmap='Reds', vmax=vmax_unc, vmin=vmin_unc, origin='lower')
    fig.colorbar(val_plot, ax=ax1).minorticks_on()
    fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

    ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
    ax2.set_title('Standard deviation\n(relative uncertainty)')

    fig.suptitle('Monte Carlo ' + tallyName + ', $k_{{eff}}$={:.5f} +/- {:.0f} pcm'.format(keff, 1e5 * stdKeff))
  
    plot.savefig(newpath + '/' + tallyName + '_' + materialName + '_MC.svg')


def plotSpatialTallyRR(outputFileRR, tallyName, normalise_by_mean='all', vmax=None, vmin=None, vmax_unc=None, vmin_unc=None, plotting=True, visualise_quarter=False, aspect_ratio=1, orientation='horizontal', remove_edges_2D=False):
    """Plots quantity in Cartesian space along with the uncertainty

    Args:
        outputFileRR (str): Name of output RR file.
        tallyName (str): Name of tally to be plotted, only two options - fiss1G or flux1G.
        normalise_by_mean (str, optional): Normalise the tally value to the mean value. Defaults to 'all'.
                            If 'all', the mean of all values is used. 
                            If 'non-zero', then only non-zero values are used to calculate the mean. 
                            If None, no normalisation is applied.
        plotting (Bool) : If True, plots and saves a figure. Otherwise just returns the values.
    """

    # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    outputs = readOutputs(outputFileRR)
    result = getattr(outputs, tallyName)
    value = np.array(getattr(result, tallyName))[...,0]
    std = np.array(getattr(result, tallyName))[...,1]  

    keff, stdKeff = findKeffRR(outputFileRR) # extract keff

    value, std, removed_edges_label = removeEdges2D(value, std, remove_edges_2D=remove_edges_2D)

    value, std, quarter_label = visualiseQuarter(value, std, visualise_quarter=visualise_quarter)

    # try to remove data from plot if the tally is 0
    value_plot = np.copy(np.where(value < 1e-18, np.nan, value))
    std_plot = np.copy(np.where(value < 1e-18, np.nan, std))

    

    # now normalise by the mean value of non-zero tallies
    if normalise_by_mean=='non-zero':
        value_no_nan = value_plot[~np.isnan(value_plot)]
        value_mean = np.mean(value_no_nan)
        print('Normalising by the mean of non-zero values,', value_mean)
        value_plot = value_plot / value_mean
    elif normalise_by_mean=='all':
        mean = np.mean(value)
        print('Normalising by the mean of all values,', mean)
        value_plot=value_plot/ mean
    elif normalise_by_mean==None:
        'No normalisation applied'
        value_plot = value_plot
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)

    # Turn NaNs back into 0s for plotting    
    value_plot = np.where(np.isnan(value_plot), 0, value_plot)
    std_plot = np.where(np.isnan(std_plot), 0, std_plot)


    if plotting: 

        if orientation == 'horizontal':
            fig, (ax1, ax2) = plot.subplots(1,2)
        elif orientation == 'vertical':
            fig, (ax1, ax2) = plot.subplots(2,1)

        # check dimension:
        if value_plot.ndim < 2:
            # This is not 2D data, assume 1-dimensional.
            ax1.plot(value_plot, color='blue')
            ax2.plot(std_plot,color='red')
         
        
        else:

            # val_plot = ax1.imshow(value_plot, cmap='Blues', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
            val_plot = ax1.imshow(value_plot, cmap='viridis', vmax=vmax, vmin=vmin, origin='lower', aspect=aspect_ratio)
            uncertainty_plot = ax2.imshow(std_plot, cmap='Reds', vmax=vmax_unc, vmin=vmin_unc, origin='lower', aspect=aspect_ratio)
            fig.colorbar(val_plot, ax=ax1).minorticks_on()
            fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

        ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
        ax2.set_title('Standard deviation\n(relative uncertainty)')

        fig.suptitle('Random ray ' + tallyName + ', $k_{{eff}}$={:.5f} +/- {:.0f} pcm'.format(keff, 1e5 * stdKeff)+ '\n'  + quarter_label  + '\n'  + removed_edges_label  )
    

        plot.savefig(newpath + '/' + tallyName + '_RR_' + tallyName + '.svg')
        
    
    return value_plot, std_plot

def plotSpatialTallyCompare_MCRR(outputFileMC, outputFileRR, tallyName_MC, tallyName_RR, normalise_by_mean='all', response_index_MC = 0, plotting=True, visualise_quarter=False, aspect_ratio=1, orientation = 'horizontal', sideByside=False, i=0, j=None, remove_edges_2D=False):
    """Plot which compares a Monte Carlo and random ray output.

    Args:
        outputFileMC (str): Monte Carlo output filepath
        outputFileRR (str): Random ray output filepath
        tallyName_MC (str): The tally name in the Monte Carlo file
        tallyName_RR (str): The tally name in the random ray file
        normalise_by_mean (str, optional): Normalise the tally value to the mean value. Defaults to 'all'.
                            If 'all', the mean of all values is used. 
                            If 'non-zero', then only non-zero values are used to calculate the mean. 
                            If None, no normalisation is applied. 
        response_index_MC (int, optional): If multiple responses have been tallied for this clerk, use this index to specify which one to plot. Defaults to 0.
        plotting (bool, optional): If True, plots and saves a figure. Otherwise just returns the values. Defaults to True.
        visualise_quarter (bool or str, optional): Defaults to False.If False, plot normally. Otherwise displays plot reflected in both y and x axes, according to the string:
                            Could be ['top-left', 'top-right', 'bottom-left', 'bottom-right'].
        aspect_ratio (int, optional): The ratio of y pixels vs x pixels when displayed. Defaults to 1.
        orientation (str, optional): Determines whether value plot and uncertainty plot are adjacent horizontally ('horizontal') or vertically ('vertical). 
                                Defaults to 'horizontal'.
        sideByside (bool, optional): For a 1D plot only. If 1D plot is detected, and this is True, then instead of plotting the relative difference, plots the values themselves on the same axes.
                                Defaults to False.
        i (int, optional): Starting index for slicing a 1D output array. Defaults to 0.
    """
     # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    value_MC, std_MC = plotSpatialTallyMC(outputFileMC, tallyName=tallyName_MC, normalise_by_mean=normalise_by_mean, response_index=response_index_MC, visualise_quarter=visualise_quarter, plotting=False, remove_edges_2D=remove_edges_2D)
    value_RR, std_RR = plotSpatialTallyRR(outputFileRR, tallyName=tallyName_RR, normalise_by_mean=normalise_by_mean, visualise_quarter=visualise_quarter, plotting=False, remove_edges_2D=remove_edges_2D)

    # extract labels:
    dummy_value = np.ones((2,2))
    dummy_std = np.ones((2,2))
    dummy_value, dummy_std, removed_edges_label = removeEdges2D(dummy_value, dummy_std, remove_edges_2D=remove_edges_2D)
    dummy_value, dummy_std, quarter_label = visualiseQuarter(dummy_value, dummy_std, visualise_quarter=visualise_quarter)
    print('Removed edges label:', removed_edges_label)
    print('Quarter label:', quarter_label)

    # check symmetry
    print('MC value symmetry check:')
    value_MC_flip = np.flip(value_MC,0)
    value_RR_flip = np.flip(value_RR,0)
    print(100*(value_MC_flip - value_MC_flip.T)/value_MC_flip)

    print('RR value symmetry check:')
    print(100*(value_RR_flip - value_RR_flip.T)/value_RR_flip)

    print('MC symmetry check:')
    print(np.allclose(value_MC_flip, value_MC_flip.T, rtol=5e-3, atol=0))
    print('RR symmetry check:')
    print(np.allclose(value_RR_flip, value_RR_flip.T, rtol=5e-3, atol=0))


    # extract keff values from the outputs

    keff_MC, stdKeffMC = findKeffMC(outputFileMC)
    keff_RR, stdKeffRR = findKeffRR(outputFileRR)

    # If slicing in 1D:
    value_MC = np.copy(value_MC)[i:j] 
    value_RR = np.copy(value_RR)[i:j]
    std_MC = np.copy(std_MC)[i:j]
    std_RR = np.copy(std_RR)[i:j]

    # TEMP TODO turn slicing option 2D

    # k = 8
    # l = -8
    # value_MC = np.copy(value_MC)[:l, k:] 
    # value_RR = np.copy(value_RR)[:l, k:]
    # std_MC = np.copy(std_MC)[:l,k:]
    # std_RR = np.copy(std_RR)[:l,k:]
    # k = 153
    # l = 153
    # value_MC = np.copy(value_MC)[8:l,:k] 
    # value_RR = np.copy(value_RR)[:l,:k]
    # std_MC = np.copy(std_MC)[:l,:k]
    # std_RR = np.copy(std_RR)[:l,:k]

    
    
    # Calculate quantities
    rel_diff = (value_RR - value_MC) / value_MC
    rel_diff_unc = np.sqrt((std_RR/value_MC**2)**2 +(std_MC * value_RR/value_MC**2) ** 2)

    # calculate statistics

    rel_diff = np.copy(np.where(np.isnan(value_MC), 0, rel_diff))   # deal with nans in the original arrays
    rel_diff_unc = np.copy(np.where(np.isnan(value_MC), 0, rel_diff_unc))

    rel_diff_for_max = np.where(value_MC< 1e-16,0, rel_diff)    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    max_error = np.max(rel_diff_for_max)

    min_error = np.min(rel_diff_for_max)
    max_abs_err = np.max(np.abs([max_error, min_error]))

    rmse = rmsError(value_MC, value_RR, relative_to='max_value')
    meanErr = meanError(value_MC, value_RR)
    # max_error = np.max(np.abs(rel_diff))

    # Turn NaNs back into 0s for plotting    
    rel_diff = np.where(np.isnan(rel_diff), 0, rel_diff)
    rel_diff_unc = np.where(np.isnan(rel_diff_unc), 0, rel_diff_unc)

    plot.rcParams['figure.constrained_layout.use'] = True

    if plotting:
        if orientation == 'horizontal':
            # fig, (ax1, ax2) = plot.subplots(1,2)
            fig, ax1 = plot.subplots() # TEMP
        elif orientation == 'vertical':
            fig, (ax1, ax2) = plot.subplots(2,1)
        # check dimension:
        if rel_diff.ndim < 2:
            # This is not 2D data, assume 1-dimensional.

            # May need to plot the values themselves rather than the difference for a 'side-by'side' comparison
            if sideByside:
                ax1.plot(value_MC, color='black', linestyle='dashed', label='MC')
                ax1.plot(value_RR, color='green', label='RR')

                ax2.plot(std_MC, color='black',linestyle='dashed')
                ax2.plot(std_RR, color='green')

                ax1.set_ylabel(tallyName_RR)
                ax2.set_ylabel('standard deviation')
                plot.figlegend()

            elif sideByside== False:
                ax1.plot(rel_diff, color='blue')
                ax2.plot(rel_diff_unc, color='red')

                ax1.set_ylabel('Relative diff')
                ax2.set_ylabel('Combined uncertainty')       


            # put x axis at y=0
            ax1.spines['bottom'].set_position(('data', 0.0000))
            ax2.spines['bottom'].set_position(('data', 0.0000))

            ax1.set_xlim(left=0)
            ax2.set_xlim(left=0)

            
        
        else: # 2D data
            # val_plot = ax1.imshow(rel_diff, cmap='RdBu_r', vmax=max_abs_err, vmin=-max_abs_err, origin='lower', aspect=aspect_ratio)
            # val_plot = ax1.imshow(rel_diff, cmap='jet', origin='lower', aspect=aspect_ratio)
            val_plot = ax1.imshow(rel_diff, cmap='viridis', origin='lower', aspect=aspect_ratio)
            # val_plot = ax1.imshow(rel_diff, cmap='rainbow', origin='lower', aspect=aspect_ratio)
            # val_plot = ax1.imshow(rel_diff, cmap='magma', origin='lower', aspect=aspect_ratio)

            # uncertainty_plot = ax2.imshow(rel_diff_unc, cmap='Reds', origin='lower', aspect=aspect_ratio)
            fig.colorbar(val_plot, ax=ax1).minorticks_on()
            # fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

        # ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
        ax1.set_title('Fission rate') # TEMP
        # ax2.set_title('Standard deviation\n(relative uncertainty)')

      

        fig.suptitle('(RR -MC) / MC: relative difference in {}. \nMax error={:.3%}, min error={:.3%}, \nRMSE={:.3%}, mean error={:.3%} relative to max MC {}.\n$k_{{MC}}={:.5f}$, $k_{{RR}}={:.5f}$, diff={:.0f} pcm'.format(tallyName_RR, max_error, min_error, rmse, meanErr, tallyName_MC, keff_MC, keff_RR, 1e5*(keff_RR-keff_MC)) + '\n'  + quarter_label  + '\n'  + removed_edges_label )

  
        plot.savefig(newpath + '/compare_MCRR' + tallyName_RR + '.svg')


def plotSpatialTallyCompare_MCMC(outputFileMC1, outputFileMC2, tallyName_MC1, tallyName_MC2, normalise_by_mean='all', response_index_MC1 = 0, response_index_MC2 = 0, plotting=True, visualise_quarter=False, aspect_ratio=1, orientation = 'horizontal', sideByside=False, i=0, j=None):
    """Plot which compares a Monte Carlo and random ray output.

    Args:
        outputFileMC (str): Monte Carlo output filepath
        outputFileRR (str): Random ray output filepath
        tallyName_MC (str): The tally name in the Monte Carlo file
        tallyName_RR (str): The tally name in the random ray file
        normalise_by_mean (str, optional): Normalise the tally value to the mean value. Defaults to 'all'.
                            If 'all', the mean of all values is used. 
                            If 'non-zero', then only non-zero values are used to calculate the mean. 
                            If None, no normalisation is applied. 
        response_index_MC (int, optional): If multiple responses have been tallied for this clerk, use this index to specify which one to plot. Defaults to 0.
        plotting (bool, optional): If True, plots and saves a figure. Otherwise just returns the values. Defaults to True.
        visualise_quarter (bool or str, optional): Defaults to False.If False, plot normally. Otherwise displays plot reflected in both y and x axes, according to the string:
                            Could be ['top-left', 'top-right', 'bottom-left', 'bottom-right'].
        aspect_ratio (int, optional): The ratio of y pixels vs x pixels when displayed. Defaults to 1.
        orientation (str, optional): Determines whether value plot and uncertainty plot are adjacent horizontally ('horizontal') or vertically ('vertical). 
                                Defaults to 'horizontal'.
        sideByside (bool, optional): For a 1D plot only. If 1D plot is detected, and this is True, then instead of plotting the relative difference, plots the values themselves on the same axes.
                                Defaults to False.
        i (int, optional): Starting index for slicing a 1D output array. Defaults to 0.
    """
     # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    value1, std1 = plotSpatialTallyMC(outputFileMC1, tallyName=tallyName_MC1, normalise_by_mean=normalise_by_mean, response_index=response_index_MC1, visualise_quarter=visualise_quarter, plotting=False)
    value2, std2 = plotSpatialTallyMC(outputFileMC2, tallyName=tallyName_MC2, normalise_by_mean=normalise_by_mean, response_index=response_index_MC2, visualise_quarter=visualise_quarter, plotting=False)

    # extract keff values from the outputs

    keff1, stdKeff1 = findKeffMC(outputFileMC1)
    keff2, stdKeff2 = findKeffMC(outputFileMC2)

    # If slicing:
    value1 = np.copy(value1)[i:j] 
    value2 = np.copy(value2)[i:j]
    std1 = np.copy(std1)[i:j]
    std2 = np.copy(std2)[i:j]
    
    # Calculate quantities
    rel_diff = (value2 - value1) / value1
    rel_diff_unc = np.sqrt((std2/value1**2)**2 +(std1 * value2/value1**2) ** 2)

    # calculate statistics

    rel_diff = np.copy(np.where(np.isnan(value1), 0, rel_diff))   # deal with nans in the original arrays
    rel_diff_unc = np.copy(np.where(np.isnan(value1), 0, rel_diff_unc))

    rel_diff_for_max = np.where(value1< 1e-16,0, rel_diff)    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    max_error = np.max(rel_diff_for_max)

    min_error = np.min(rel_diff_for_max)
    max_abs_err = np.max(np.abs([max_error, min_error]))

    rmse = rmsError(value1, value2, relative_to='max_value')
    meanErr = meanError(value1, value2)
    # max_error = np.max(np.abs(rel_diff))

    # Turn NaNs back into 0s for plotting    
    rel_diff = np.where(np.isnan(rel_diff), 0, rel_diff)
    rel_diff_unc = np.where(np.isnan(rel_diff_unc), 0, rel_diff_unc)

    plot.rcParams['figure.constrained_layout.use'] = True

    if plotting:
        if orientation == 'horizontal':
            # fig, (ax1, ax2) = plot.subplots(1,2)
            fig, ax1 = plot.subplots() # TEMP
        elif orientation == 'vertical':
            fig, (ax1, ax2) = plot.subplots(2,1)
        # check dimension:
        if rel_diff.ndim < 2:
            # This is not 2D data, assume 1-dimensional.

            # May need to plot the values themselves rather than the difference for a 'side-by'side' comparison
            if sideByside:
                ax1.plot(value1, color='black', linestyle='dashed', label='MC1')
                ax1.plot(value2, color='green', label='MC2')

                ax2.plot(std1, color='black',linestyle='dashed')
                ax2.plot(std2, color='green')

                ax1.set_ylabel(tallyName_MC2)
                ax2.set_ylabel('standard deviation')
                plot.figlegend()

            elif sideByside== False:
                ax1.plot(rel_diff, color='blue')
                ax2.plot(rel_diff_unc, color='red')

                ax1.set_ylabel('Relative diff')
                ax2.set_ylabel('Combined uncertainty')       


            # put x axis at y=0
            ax1.spines['bottom'].set_position(('data', 0.0000))
            ax2.spines['bottom'].set_position(('data', 0.0000))

            ax1.set_xlim(left=0)
            ax2.set_xlim(left=0)

            
        
        else:
            val_plot = ax1.imshow(rel_diff, cmap='RdBu_r', vmax=max_abs_err, vmin=-max_abs_err, origin='lower', aspect=aspect_ratio)
            # val_plot = ax1.imshow(rel_diff, cmap='viridis', origin='lower', aspect=aspect_ratio)

            # uncertainty_plot = ax2.imshow(rel_diff_unc, cmap='Reds', origin='lower', aspect=aspect_ratio)
            fig.colorbar(val_plot, ax=ax1).minorticks_on()
            # fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

        ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
        # ax2.set_title('Standard deviation\n(relative uncertainty)')

      

        fig.suptitle('(val2 -val1) / val1: relative difference in {}. \nMax diff={:.3%}, min diff={:.3%}, \nRMS diff={:.3%}, mean diff={:.3%} relative to max MC1 {}.\n$k_{{1}}={:.5f}$, $k_{{2}}={:.5f}$, diff={:.0f} pcm'.format(tallyName_MC2, max_error, min_error, rmse, meanErr, tallyName_MC1, keff1, keff2, 1e5*(keff2-keff1)))

  
        plot.savefig(newpath + '/compare_MCMC' + tallyName_MC1+ str(int(response_index_MC1)) + '.svg')

def plotSpatialTallyCompare_MCMG(outputFileCE, outputFileMG, tallyName_CE, tallyName_MG, normalise_by_mean='all', response_index_CE = 0, response_index_MG = 0,  visualise_quarter=False ):
    """Compares the output of two Monte Carlo files, one using continuous energy and the other multigroup

    Args:
        outputFileMC (_type_): _description_
        outputFileRR (_type_): _description_
        tallyName_MC (_type_): _description_
        tallyName_RR (_type_): _description_
        normalise_by_mean (str, optional): _description_. Defaults to 'all'.
        response_index_MC (int, optional): _description_. Defaults to 0.
        visualise_quarter (bool, optional): _description_. Defaults to False.
    """
     # make an output folder to store outputs
    newpath = 'outputs'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    value_CE, std_CE = plotSpatialTallyMC(outputFileCE, tallyName_CE, normalise_by_mean, response_index_CE, visualise_quarter=visualise_quarter, plotting=False)
    value_MG, std_MG = plotSpatialTallyMC(outputFileMG, tallyName_MG, normalise_by_mean, response_index_MG, visualise_quarter=visualise_quarter, plotting=False)

    # Calculate quantities
    rel_diff = (value_MG - value_CE) / value_CE
    rel_diff_unc = (std_MG - std_CE) / std_CE

    # calculate statistics
    value_CE = np.copy(np.where(np.isnan(value_CE), 0, value_CE))
    value_MG = np.copy(np.where(np.isnan(value_MG), 0, value_MG))

    rel_diff_for_max = np.where(value_CE< 1e-16,0, rel_diff)    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    max_error = np.max(rel_diff_for_max)
    min_error = np.min(rel_diff_for_max)
    max_abs_err = np.max(np.abs([max_error, min_error]))

    rmse = rmsError(value_CE, value_MG, relative_to='max_value')
    meanErr = meanError(value_CE, value_MG)
 



    fig, (ax1, ax2) = plot.subplots(1,2)
    # val_plot = ax1.imshow(value_plot, cmap='hot', vmax=vmax, vmin=vmin)
    val_plot = ax1.imshow(rel_diff, cmap='RdBu_r', vmax=max_abs_err, vmin=-max_abs_err, origin='lower')

    uncertainty_plot = ax2.imshow(rel_diff_unc, cmap='hot', origin='lower')
    fig.colorbar(val_plot, ax=ax1).minorticks_on()
    fig.colorbar(uncertainty_plot, ax=ax2).minorticks_on()

    ax1.set_title('Value \nNormalised by the mean\n of {} values'.format(normalise_by_mean))
    ax2.set_title('Standard deviation\n(relative uncertainty)')

    fig.suptitle('(MG-CE) / CE: relative difference in {}. \nMax error={:.3%}, min error={:.3%}, \nRMSE={:.3%}, mean error={:.3%} relative to max MC {}.'.format(tallyName_CE, max_error, min_error, rmse, meanErr, tallyName_CE))

    plot.savefig(newpath + '/compare_MCMG_' + tallyName_CE + '.svg')


def plotFissionRatesMC(outputFile, normalise_plot=False, target=100):
    fissRate, fissRate_std = findFissRateMC(outputFile)

    fig, ax1 = plot.subplots()
    if normalise_plot == True:
        # fissRate = fissRate / np.max(fissRate)
        fissRate = normalise(fissRate, target)
        
    val = ax1.imshow(fissRate)
    fig.colorbar(val, ax=ax1)
    fig.suptitle('Monte Carlo fission rate')
    plot.savefig('Fission_rate_MC.svg')

def plotFluxMC(outputFile, normalise_plot=False, target=100):
    flux, flux_std = findFluxMC(outputFile)

    fig, ax1 = plot.subplots()
    if normalise_plot == True:
      
        flux = normalise(flux, target)
        
    val = ax1.imshow(flux)
    fig.colorbar(val, ax=ax1)
    fig.suptitle('Monte Carlo flux')
    plot.savefig('Flux_MC.svg')

def plotFissionRatesRR(outputFile, normalise_plot=False, target=100):
    # outputs = readOutputs(outputFile)
    
    fissRate, fissRate_std = findFissRateRR(outputFile)

    
    fig, ax1 = plot.subplots()

    if normalise_plot == True:
        # fissRate = fissRate / np.max(fissRate)
        fissRate = normalise(fissRate, target)
        
    val = ax1.imshow(fissRate)
    fig.colorbar(val, ax=ax1)

    fig.suptitle('Random ray fission rate')
    plot.savefig('Fission_rate_RR.svg')


def plotFissionRatesRR_radial(outputFile, normalise_plot=False, target=100):
    # outputs = readOutputs(outputFile)
    # or line plot?
    fissRate, fissRate_std, radialBounds = findFissRateRR_radial(outputFile)
    if normalise_plot == True:
        fissRate = normalise(fissRate, target)

    r = (np.array(radialBounds)[...,0] + np.array(radialBounds)[...,1]) / 2  

    fig, ax = plot.subplots()
    ax.plot(r, fissRate, 'bo-')
    ax.set_ylabel('Fission rate')
    ax.set_xlabel('radius (cm)')
    ax.set_title('Fission rate against radius')
  
    plot.savefig('Fission_rate_RR.svg')



def plotFissionRatesMC_radial(outputFile, normalise_plot=False, target=100):
    # outputs = readOutputs(outputFile)
    # or line plot?
    fissRate, fissRate_std, radialBounds = findFissRateMC_radial(outputFile)
    if normalise_plot == True:
        fissRate = normalise(fissRate, target)

    r = (np.array(radialBounds)[...,0] + np.array(radialBounds)[...,1]) / 2  

    fig, ax = plot.subplots()
    ax.plot(r, fissRate, 'bo-')
    ax.set_ylabel('Fission rate')
    ax.set_xlabel('radius (cm)')
    ax.set_title('Fission rate against radius')

    plot.savefig('Fission_rate_MC.svg')

    
def plotFissionRatesCompareMC_RR(outputFileMC,outputFileRR, target=100):

    fissRateMC = normalise(findFissRateMC(outputFileMC)[0], target)
    fissRateRR = normalise(findFissRateRR(outputFileRR)[0], target)   
    
    # Calculate quantities
    rel_diff = (fissRateRR - fissRateMC) / fissRateMC

    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    rel_diff_for_max = np.where(fissRateMC< 1e-16,0, rel_diff)
    max_error = np.max(np.abs(rel_diff_for_max))

    rmse = rmsError(fissRateMC, fissRateRR, target, relative_to='max_value')
    meanErr = meanError(fissRateMC, fissRateRR, target)
    # max_error = np.max(np.abs(rel_diff))

    # plot relative difference
    fig, ax1 = plot.subplots()
            
    val = ax1.imshow(rel_diff, cmap='turbo')
    cb = fig.colorbar(val, ax=ax1, format='{x:.3f}')
    cb.set_label('Relative difference')

    fig.suptitle('(RR -MC) / MC: relative difference in fission rate.Max error={:.1%}, \nRMSE={:.2%}, mean error={:.2%} relative to max MC fission rate'.format(max_error, rmse, meanErr))
    plot.savefig('Fission_rate_rel_diff.svg')
    return rel_diff

def plotFissionRatesCompare_radial_MC_RR(outputFileMC,outputFileRR, target=100):

    fissRateMC, fissRateMC_std, radialBoundsMC = findFissRateMC_radial(outputFileMC)
    fissRateRR, fissRateRR_std, radialBoundsRR= findFissRateRR_radial(outputFileRR)
    
    fissRateMC = normalise(fissRateMC, target)
    fissRateRR = normalise(fissRateRR, target)

    r_MC = (np.array(radialBoundsMC)[...,0] + np.array(radialBoundsMC)[...,1]) / 2  
    r_RR = (np.array(radialBoundsRR)[...,0] + np.array(radialBoundsRR)[...,1]) / 2  


    # Calculate quantities
    rel_diff = (np.array(fissRateRR) - np.array(fissRateMC)) / np.array(fissRateMC)
    rel_diff = np.nan_to_num(rel_diff) * 100
    
    rmse = rmsError(fissRateMC, fissRateRR, target, relative_to='max_value')
    max_error = np.max(np.abs(rel_diff))


    fig, ax = plot.subplots()

    ax.plot(r_MC, rel_diff, 'bo-')

    ax.set_ylabel('% (RR-MC)/MC')
    ax.set_xlabel('radius (cm)')
    plot.title('Fission rate against radius.\nMax error={:.4f}%, RMSE={:.4e}'.format(max_error, rmse))

    plot.savefig('Relative_diff_fission_rate.svg')


def plotFluxSpectrumMC(outputFileMC):
    outputs = readOutputs(outputFileMC)
    fs = outputs.active.fluxSpectrum
    res = np.array(fs.Res)
    EnergyBounds_plot = np.flip(np.array(fs.EnergyBounds), 1)
    x = (EnergyBounds_plot[0] + EnergyBounds_plot[1]) / 2 # for plotting

    if hasattr(fs, 'MaterialBins'):
        material_list = np.array(fs.MaterialBins)
    else:
        material_list = np.array(['material'])
    
    i=0
    for material in material_list:
        
        [material] = material
        print(material)
        
        flux_res_material = res[i]
        flux_res = flux_res_material[..., 0][:,0]
   
        i = i+1

        fig, ax = plot.subplots()
        
        widths = (EnergyBounds_plot[1]-EnergyBounds_plot[0])

        # ax.bar(x, flux_res, width=widths,label=material, edgecolor='black')
        ax.bar(x, flux_res, width=widths, edgecolor='black')
 

        ax.set_xscale('log')

        ax.set_ylabel('Flux')
        ax.set_xlabel('MeV')
        ax.set_title('{:.0f} groups, flux for {}.'.format(len(EnergyBounds_plot[0]), material))
 
        plot.savefig('Flux_spectrum_' + material+'.svg')
       
       
        






def findFissRateRR(outputFileRR):
    outputs = readOutputs(outputFileRR)
    fissRate = np.array(outputs.fiss1G.fiss1G)[...,0]
    fissRate_std = np.array(outputs.fiss1G.fiss1G)[...,1]
 
    return fissRate, fissRate_std

def findFissRateRR_radial(outputFileRR):
    outputs = readOutputs(outputFileRR)
    fissRate = np.array(outputs.fiss1G.fiss1G)[...,0]
    fissRate_std = np.array(outputs.fiss1G.fiss1G)[...,1]

    radialBounds = outputs.fiss1G.radialMapRadialBounds

    return fissRate, fissRate_std, radialBounds



def findFissRateMC(outputFileMC):
    outputs = readOutputs(outputFileMC)
    reactionRate = np.array(outputs.active.pinFiss.Res)
    fissRate = reactionRate[...,1,0]
    fissRate_std = reactionRate[...,1,1]
    return fissRate, fissRate_std

def findFissRateMC_radial(outputFileMC):
    outputs = readOutputs(outputFileMC)
    reactionRate = np.array(outputs.active.pinFiss.Res)
    fissRate = reactionRate[...,1,0]
    fissRate_std = reactionRate[...,1,1]
    radialBounds = outputs.active.pinFiss.radialMapRadialBounds
    return fissRate, fissRate_std, radialBounds



def findFluxMC(outputFileMC):
    outputs = readOutputs(outputFileMC)
    reactionRate = np.array(outputs.active.pinFiss.Res)
    flux = reactionRate[...,0,0]
    flux_std = reactionRate[...,0,1]
    return flux, flux_std

def findKeffMC(outputFileMC):
    """Finds the k_eff value from the MC output file.

    Args:
        outputFileMC (str): The path to the MC output file.

    Returns:
        float: The k_eff value.
    """
    outputs = readOutputs(outputFileMC, print_output=False)
    keff = outputs.keff.K_EFF[0]
    std = outputs.keff.K_EFF[1]

    return keff, std

def findKeffRR(outputFileRR):
    """Finds the k_eff value from the random ray output file.

    Args:
        outputFileRR (str): The path to the random ray output file.

    Returns:
        float: The k_eff value.
    """
    outputs = readOutputs(outputFileRR, print_output=False)
    keff = outputs.keff.keff[0]
    std = outputs.keff.keff[1]
    return keff, std

def normalise(array, target):
    # array_norm = array / np.max(array)

    alpha = target / np.copy(np.sum(array))
    array_norm = np.copy(array) * alpha

    return array_norm

def rmsError(actual_result, predicted_result, target=None, relative_to=None):
    

    if target is not None:
        # normalise both results
        actual_result = normalise(actual_result, target)
        predicted_result = normalise(predicted_result, target)

    # print(predicted_result)
    # Calculate the mean squared error (MSE) by taking the mean of the squared differences
    meanSquaredError = ((predicted_result - actual_result) ** 2).mean()
    

    if relative_to == 'max_value':
        # Calculate the RMSE by taking the square root of the MSE
        rmse = np.sqrt(meanSquaredError) / np.max(actual_result)
    else:
        rmse = np.sqrt(meanSquaredError) #/ relative_to # if relative_to is a floar
    return rmse

def meanError(actual_result, predicted_result, target=None):

    if target is not None:
        # normalise both results
        actual_result = normalise(actual_result, target)
        predicted_result = normalise(predicted_result, target)

    # Calculate the mean squared error (MSE) by taking the mean of the squared differences
    meanError = (np.abs(predicted_result - actual_result)).mean()
    return meanError


def visualiseQuarter(value, std, visualise_quarter=False):
    """Adjust data assuming it is a quarter of the core, reflected in both x and y axes.

    Args:
        array (np.ndarray): The array to visualise.
        quarter (str, optional): The quarter to visualise. Defaults to 'top-left'.
    """
    
    if visualise_quarter=='top-right':
        ## The given array only covers the top-right quarter of the fuel assembly. For visualisation, need to reflect this in the array. Need to 'mask' the central fuel pins.
        # step 1, double the stats in the centre, and alter the relative uncertainty to reflect this
        value[0,:] = 2 * value[0,:]
        value[:, 0] = 2 * value[:, 0]

        std[0,:] = std[0,:] / 2
        std[:, 0] = std[:, 0] / 2

        # step 2 flip and concatenate
        value = np.concatenate([ np.flip(value, 0), value])
        value = np.concatenate([ np.flip(value, 1), value],1)
        std = np.concatenate([np.flip(std, 0), std])
        std = np.concatenate([np.flip(std, 1), std],1)

        # now remove the central row and column to avoid duplication
        midindex = value.shape[0] // 2
        value = np.delete(value, midindex, 0)
        value = np.delete(value, midindex, 1)
        std = np.delete(std, midindex, 0)
        std = np.delete(std, midindex, 1)

        # step 3 write a label
        quarter_label = '(visual adjusted for quarter geometry)'

    elif visualise_quarter == 'bottom-right':
        ## The given array only covers the bottom-right quarter of the fuel assembly. For visualisation, need to reflect this in the array. Need to 'mask' the central fuel pins.
        # step 1, double the stats in the centre, and alter the relative uncertainty to reflect this
        value[-1,:] = 2 * value[-1,:]
        value[:, 0] = 2 * value[:, 0]

        std[-1,:] = std[-1,:] / 2
        std[:, 0] = std[:, 0] / 2

        # step 2 flip and concatenate
        value = np.concatenate([ value,np.flip(value, 0) ])
        value = np.concatenate([ np.flip(value, 1), value],1)

        std = np.concatenate([ std, np.flip(std, 0)])
        std = np.concatenate([np.flip(std, 1), std],1)

        # now remove the central row and column to avoid duplication
        midindex = value.shape[0] // 2
        value = np.delete(value, midindex, 0)
        value = np.delete(value, midindex, 1)
        std = np.delete(std, midindex, 0)
        std = np.delete(std, midindex, 1)

        # step 3 write a label
        quarter_label = '(visual adjusted for quarter geometry)'

    elif visualise_quarter == 'top-right-only':
        # Only adjust the values along the centreline.

        value[0,:] = 2 * value[0,:]
        value[:, 0] = 2 * value[:, 0]

        std[0,:] = std[0,:] / 2
        std[:, 0] = std[:, 0] / 2

        # step 3 write a label
        quarter_label = '(visual adjusted for quarter geometry)'

    elif visualise_quarter == 'bottom-right-only':
        # Only adjust the values along the centreline.

        value[-1,:] = 2 * value[-1,:]
        value[:, 0] = 2 * value[:, 0]

        std[-1,:] = std[-1,:] / 2
        std[:, 0] = std[:, 0] / 2

        # step 3 write a label
        quarter_label = '(visual adjusted for quarter geometry)'
    else:
        quarter_label = ""
    
    return value, std, quarter_label


def removeEdges2D(value, std, remove_edges_2D=False):
     
    if remove_edges_2D == 'bottom-right':
       
        value = np.copy(value)[:-1,1:]
        std = np.copy(std)[:-1, 1:]
        removed_edges_label = '(edge elements removed for bottom-right quarter geometry)'

    elif remove_edges_2D == False:
        removed_edges_label = ''
    return value, std, removed_edges_label



if __name__=='__main__':

    # plotFissionRatesRR_radial('SimplePin_RR_output_radial.json', normalise_plot=True, target=100)
    # plotFissionRatesMC_radial('SimplePin_MC_output_radial.json', normalise_plot=True, target=100)
    # plotFissionRatesCompare_radial_MC_RR('SimplePin_MC_output_radial.json', 'SimplePin_RR_output_radial.json')
    # plotFluxSpectrumMC('SimplePin_MC_output_fluxSpectrum.json')
    # plotScatteringMatrices('SimplePin_MC_output_70G_problematic.json')
    # plotFluxSpectrumMC('SimplePin_MC_output_10^8.json')
    # plotShannon('SimpleSlab_MC', 'SimplePin_MC_output_10^8.json' )
    # plotFissionRatesMC('assembly_MC_output.json', 100)
    # plotFissionRatesRR('assembly_RR_output.json', 100)
    plotSpatialTallyMC('FuelAssembly_MC_output_spatial_tally.json', 'pinFissionRate', response_index=0, plotting=True)
    # plotSpatialMaterialTallyMC('FuelAssembly_MC_output_space_and_material.json', tallyName='u238Capture', materialName='UO2-31', normalise_by_mean='all')