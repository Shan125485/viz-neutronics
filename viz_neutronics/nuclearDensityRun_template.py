# TEMPLATE

from viz_neutronics.nuclearDensityCalculator import results
from viz_neutronics.nuclearDensityInputs import input


file_out = 'nuclearDensityOutputs.txt'


x = results(input)
x.display()
x.write_results(file_out)