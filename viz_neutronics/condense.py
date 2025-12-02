import numpy as np


def condense(weight_param : np.ndarray, parameter: np.ndarray, group_structure: list):
    """Weights a parameter and condenses into the given group structure. Parameter, weighting parameter and group structure must all have the same total number of groups.

    Args:
        weight_param (np.ndarray): an array of flux (or current) for each energy group
        parameter (np.ndarray): an array of the parameter to be condensed at each energy group
        group_structure (list of tuples):of the form [(0,246), (247, 280)]
    """
    # Run input checks
    input_checks(weight_param, parameter, group_structure)


    condensed_param_list = []

    for group in group_structure:

        lower_bound = group[0]
        upper_bound = group[1] # these are inclusive bounds

        weight_param_segment = weight_param[lower_bound: upper_bound+1]
        parameter_segment = parameter[lower_bound: upper_bound+1]
            
        
        # first multiply
        product = weight_param_segment * parameter_segment
        
        # then sum
        weight_param_sum = np.sum(weight_param_segment)
        product_sum = np.sum(product)

        # then divide
        condensed_param = product_sum / weight_param_sum
        condensed_param_list.append(condensed_param)
 

    return condensed_param_list

def condense_D_coeff_with_uncertainty(weight_param, D, group_structure, transportXS_std=None, phi_std=None):
    """Given a diffusion coefficient array D and the transport cross section uncertainties, condense it according to the provided group structure.

    Args:
        weight_param (np.array): an array of flux (or current) for each energy group
        parameter (np.array): an array of the parameter to be condensed at each energy group
        group_structure (list of tuples):of the form [(0,246), (247, 280)]
    """
    
    input_checks(weight_param, D, group_structure)

    condensed_param_list = []
    D_std_list = []

    for group in group_structure:
        # print(group)
        lower_bound = group[0]
        upper_bound = group[1] # these are inclusive bounds

        weight_param_segment = weight_param[lower_bound: upper_bound+1]
        parameter_segment = D[lower_bound: upper_bound+1]

        # if type(transportXS_std) != None:
        if True:
            # Calculate the standard deviation for the diffusion coefficient in this group
            delta_transportXS = np.array(transportXS_std[lower_bound: upper_bound+1])
            delta_phi = phi_std[lower_bound: upper_bound+1]
            D_std = delta_D_G(parameter_segment, weight_param_segment, delta_transportXS, delta_phi )
            D_std_list.append(D_std)
            
        
        # first multiply
        product = weight_param_segment * parameter_segment
        
        # then sum
        weight_param_sum = np.sum(weight_param_segment)
        product_sum = np.sum(product)

        # then divide
        condensed_param = product_sum / weight_param_sum
        condensed_param_list.append(condensed_param)

    return condensed_param_list, D_std_list
    
def input_checks(weight_param, parameter, group_structure):
     # Check whether the weight_param and parameter arrays have the same length
    if weight_param.shape[0] != parameter.shape[0]:
        raise ValueError("weight_param and parameter arrays must have the same shape.")

    # The number of groups in the group structure must also be equal to the length of the parameter arrays:
    if group_structure[-1][-1]+1 != weight_param.shape[0]:
        raise ValueError("The group structure must cover the entire length of parameter arrays")
    
    # There should be no overlap in the values in the group structure:
    if any(group_structure[i][1] >= group_structure[i+1][0] for i in range(len(group_structure)-1)):
        raise ValueError("The group structure must not have overlapping ranges.")
    
    # Group structure tuples should be ordered from lowest to highest group indices:
    if any(group_structure[i][0] != (group_structure[i-1][1] + 1) for i in range(1, len(group_structure))):
        raise ValueError("The group structure must be ordered and continuous starting from group index 0.")

    # The first group structure value should start from 0
    if group_structure[0][0] != 0:
        raise ValueError("The group structure must start from group index 0.")


def delta_D_G(D, phi, delta_transportXS, delta_phi ):
    """Returns the standard deviation of the group condensed diffusion coefficient D_G given the uncertainty in transport cross section and flux

    Args:
        D (_type_): _description_
        phi (_type_): _description_
        delta_transportXS (_type_): _description_
        delta_phi (_type_): _description_

    Returns:
        _type_: _description_
    """
    delta_D_phi = diff_flux_product_unc(D, phi, delta_transportXS, delta_phi )

    term1 = (1/np.sum(phi)) ** 2    
    term2 = np.sum(delta_D_phi ** 2)
    term3 = (np.sum(D * phi) /(np.sum(phi) ** 2)) ** 2
    term4 = np.sum(delta_phi ** 2)

    delta_D_G = np.sqrt(term1 * term2 + term3 * term4)
    return delta_D_G

def diff_flux_product_unc(D, phi, delta_transportXS, delta_phi ):
    """For each sub-group g, calculate the standard deviation of the product 
    of the diffusion coefficient and flux in that group (D_g * phi_g).
    This relies on the quadrature rule for error propagation.


    Args:
        D (_type_): _description_
        phi (_type_): _description_
        delta_transportXS (_type_): _description_
        delta_phi (_type_): _description_

    Returns:
        _type_: _description_
    """
  

    val = D**2 * delta_phi**2 + (2/3 * phi * D ** 2)**2 * delta_transportXS**2

    return np.sqrt(val)


def transXS_2_D(transXS):
    return 1/(3 * transXS)


if __name__ == "__main__":
    pass