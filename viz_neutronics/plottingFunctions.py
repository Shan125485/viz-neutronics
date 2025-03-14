import json
import numpy as np
import matplotlib.pyplot as plot

from viz_neutronics.input2json import parse_text_to_dict, save_to_json, stringTuple_to_array, dict2obj# run from outside module
#from input2json import parse_text_to_dict, save_to_json,  dict2obj # run from within module



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
    ax.plot(shannonEntropy[:inactiveCycles + activeCycles])
    ax.set_ylabel('Shannon entropy')
    plot.title(str(inactiveCycles) + ' inactive cycles, ' + str(activeCycles) + ' active cycles')
    plot.tight_layout()
    plot.savefig('Shannon_entropy')


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

    plot.savefig('P0_colourmap')

def plotFissionRatesMC(outputFile):
    outputs = readOutputs(outputFile)
    reactionRate = np.array(outputs.active.pinFiss.Res)
    flux = reactionRate[:,:,0,0]
    flux_std = reactionRate[:,:,0,1]
    fissRate = reactionRate[:,:,1,0]
    fissRate_std = reactionRate[:,:,1,1]
    X = np.array(outputs.active.pinFiss.XBounds)
    Y = np.array(outputs.active.pinFiss.XBounds)

    # Average coordinates to point to the centre of cell rather than bounbdaries
    X =  (X[:0] + X[:1]) / 2
    Y =  (Y[:0] + Y[:1]) / 2

    fig, ax1 = plot.subplots()
    val = ax1.imshow(fissRate)
    fig.colorbar(val, ax=ax1)
    fig.suptitle('Monte Carlo fission rate')
    plot.savefig('Fission_rate_MC')

def plotFissionRatesRR(outputFile):
    outputs = readOutputs(outputFile)

    
    
    fissRate = np.array(outputs.fiss1G.fiss1G)[...,0]
    print(fissRate.shape)
    fissRate_std = np.array(outputs.fiss1G.fiss1G)[...,1]
    X_fiss = (np.array(outputs.fiss1G.XBounds)[...,0] + np.array(outputs.fiss1G.XBounds)[...,1])/2
    Y_fiss = (np.array(outputs.fiss1G.YBounds)[...,0] + np.array(outputs.fiss1G.YBounds)[...,1])/2
    flux = np.array(outputs.flux1G.flux1G)[...,0]
    flux_std = np.array(outputs.flux1G.flux1G)[...,1]



    # fissRate_std = reactionRate[:,:,1,1]
    # X = np.array(outputs.active.pinFiss.XBounds)
    # Y = np.array(outputs.active.pinFiss.XBounds)

    # # Average coordinates to point to the centre of cell rather than bounbdaries
    # X =  (X[:0] + X[:1]) / 2
    # Y =  (Y[:0] + Y[:1]) / 2

    fig, ax1 = plot.subplots()
    val = ax1.imshow(fissRate)
    fig.colorbar(val, ax=ax1)

    fig.suptitle('Random ray fission rate')
    plot.savefig('Fission_rate_RR')

    
