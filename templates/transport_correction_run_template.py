# TEMPLATE
from viz_neutronics.transport_correction import generate_mg_XS


MC_output_file = "inputs/SimplePin_MC_output_64G.json"
tcType = 'outscatter'

generate_mg_XS(MC_output_file, tcType)

# this will create a directory called materialsInputs with the materials XSfile inside.