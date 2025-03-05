import json
import pprint

# from viz_neutronics.input2json import parse_text_to_dict, save_to_json, stringTuple_to_array, dict2obj # run from outside module
from input2json import parse_text_to_dict, save_to_json, stringTuple_to_array, dict2obj # run from within module


inputFile = 'SimpleSlab_MC'
# output_file = 'SimpleSlab_MC_output_1G.json'
output_file = 'SimpleSlab_MC_output_wims172.json'

##############################################

# read in inputs
print('Reading in input file', inputFile, 'as a dictionary')
inputDict = parse_text_to_dict(inputFile)

print('Saving input dictionary to input.json')
save_to_json(inputDict, 'input.json')

print('Input dictionary keys are:\n')
for key in inputDict.keys():
    print('-->', key)


print('\n\nLoading {} into an output dictionary'.format(output_file))
# returns output JSON object as python dictionary
with open(output_file) as f:
    outputDict = json.load(f)

print('Output dictionary keys are:\n')
for key in outputDict.keys():
    print('-->', key)


print('\nConverting dictionaries into objects: inputs and outputs')
inputs = dict2obj(inputDict)
outputs = dict2obj(outputDict)

####################