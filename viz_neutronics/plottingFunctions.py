import json
import numpy as np
import matplotlib.pyplot as plot

from viz_neutronics.input2json import parse_text_to_dict, save_to_json, stringTuple_to_array, dict2obj# run from outside module
# from input2json import parse_text_to_dict, save_to_json,  dict2obj # run from within module



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

def readOutputs(output_file):
    print('\n\nLoading {} into an output dictionary'.format(output_file))
    # returns output JSON object as python dictionary
    with open(output_file) as f:
        outputDict = json.load(f)

    print('Output dictionary keys are:\n')
    for key in outputDict.keys():
        print('-->', key)
    
    outputs = dict2obj(outputDict)
    return outputs


def plotShannon(inputFile, outputFile):
    inputs = readInputs(inputFile)
    outputs = readOutputs(outputFile)
    shannonEntropy = outputs.inactive.shannon.shannonEntropy
    inactiveCycles = inputs.inactive 
    activeCycles = inputs.active

    fig, ax = plot.subplots()
    ax.plot(shannonEntropy[:inactiveCycles])
    ax.set_ylabel('Shannon entropy')
    plot.title(str(inactiveCycles) + ' inactive cycles, ' + str(activeCycles) + ' active cycles')
    plot.tight_layout()
    plot.savefig('Shannon_entropy.svg')


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
    plot.tight_layout()

    plot.savefig('P0_colourmap.svg')

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
    plot.grid()
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
    plot.grid()
    plot.savefig('Fission_rate_MC.svg')

    
def plotFissionRatesCompareMC_RR(outputFileMC,outputFileRR, target=100):

    fissRateMC = normalise(findFissRateMC(outputFileMC)[0], target)
    fissRateRR = normalise(findFissRateRR(outputFileRR)[0], target)   
    
    # Calculate quantities
    rel_diff = (fissRateRR - fissRateMC) / fissRateMC

    # in a fuel assembly, areas with 0 fission rate cause a divide by zero error in the relative difference. This isn't an issue for plotting, but for the max error 
    rel_diff_for_max = np.where(fissRateMC< 1e-16,0, rel_diff)
    max_error = np.max(np.abs(rel_diff_for_max))

    rmse = rmsError(fissRateMC, fissRateRR, target)
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
    
    rmse = rmsError(fissRateMC, fissRateRR, target)
    max_error = np.max(np.abs(rel_diff))


    fig, ax = plot.subplots()

    ax.plot(r_MC, rel_diff, 'bo-')

    ax.set_ylabel('% (RR-MC)/MC')
    ax.set_xlabel('radius (cm)')
    plot.title('Fission rate against radius.\nMax error={:.4f}%, RMSE={:.4e}'.format(max_error, rmse))
    plot.grid()
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
        plot.grid()
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

def normalise(array, target):
    # array_norm = array / np.max(array)

    alpha = target / np.copy(np.sum(array))
    array_norm = np.copy(array) * alpha

    return array_norm

def rmsError(actual_result, predicted_result, target = None):

    if target is not None:
        # normalise both results
        actual_result = normalise(actual_result, target)
        predicted_result = normalise(predicted_result, target)

    # Calculate the mean squared error (MSE) by taking the mean of the squared differences
    meanSquaredError = ((predicted_result - actual_result) ** 2).mean()

    # Calculate the RMSE by taking the square root of the MSE
    rmse = np.sqrt(meanSquaredError) / np.max(actual_result)
    return rmse

def meanError(actual_result, predicted_result, target = None):

    if target is not None:
        # normalise both results
        actual_result = normalise(actual_result, target)
        predicted_result = normalise(predicted_result, target)

    # Calculate the mean squared error (MSE) by taking the mean of the squared differences
    meanError = (np.abs(predicted_result - actual_result)).mean()
    return meanError


if __name__=='__main__':

    # plotFissionRatesRR_radial('SimplePin_RR_output_radial.json', normalise_plot=True, target=100)
    # plotFissionRatesMC_radial('SimplePin_MC_output_radial.json', normalise_plot=True, target=100)
    # plotFissionRatesCompare_radial_MC_RR('SimplePin_MC_output_radial.json', 'SimplePin_RR_output_radial.json')
    # plotFluxSpectrumMC('SimplePin_MC_output_fluxSpectrum.json')
    # plotScatteringMatrices('SimplePin_MC_output_70G_problematic.json')
    # plotFluxSpectrumMC('SimplePin_MC_output_10^8.json')
    # plotShannon('SimpleSlab_MC', 'SimplePin_MC_output_10^8.json' )
    plotFissionRatesMC('assembly_MC_output.json', 100)
    # plotFissionRatesRR('assembly_RR_output.json', 100)