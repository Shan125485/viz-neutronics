# For ENDFB8 library



def find_ZAID_ext(temp_C: float):
    temp_K = temp_C + 273
    
    ext_array = ['05', '06', '00', '01', '02', '03', '04']
    temp_array = [0.1, 250, 293.6, 600, 900, 1200, 2500]

    # Out-of-range checks
    if temp_K > max(temp_array):
        raise ValueError('Target material temperature {} K is greater than the maximum tabulated temperature of {}'.format(
            temp_K, max(temp_array)))
    elif temp_K < min(temp_array):
        raise ValueError('Target material temperature {} K is less than the minimum tabulated temperature of {}'.format(
            temp_K, min(temp_array)))

    # Find closest tabulated value:
    idx, closest_temp = min(
        enumerate(temp_array),
        key=lambda x: abs(x[1] - temp_K)
    )
    ext = ext_array[idx]

    if abs(temp_K - closest_temp) > 1:  # If the difference is greater than 1 degree
        print('WARNING: Material extension of {} corresponds to a temperature of {} K. Target material temperature is {} K'.format(
            ext, closest_temp, temp_K))

    return str(ext)



def find_moder_extension(temp_C : float):
    temp_K = temp_C + 273

    # find closest match... from Appendix D of ENDF/B-VIII.0-based ACE files for thermal scattering data [19]
    ext_array = [40, 41, 42, 43, 44, 45, 46,
                 47, 48, 50, 51, 52, 53, 54, 55, 56, 57]
    temp_array = [294, 284, 300, 324, 350, 374, 400, 424,
                  450, 474, 500, 524, 550, 574, 600, 624, 650, 800]

    # Out-of-range checks
    if temp_K > max(temp_array):
        raise ValueError('Target moderator temperature {} K is greater than the maximum tabulated temperature of {}'.format(
            temp_K, max(temp_array)))
    elif temp_K < min(temp_array):
        raise ValueError('Target moderator temperature {} K is less than the minimum tabulated temperature of {}'.format(
            temp_K, min(temp_array)))

    # Find closest tabulated value:
    idx, closest_temp = min(
        enumerate(temp_array),
        key=lambda x: abs(x[1] - temp_K)
    )
    ext = ext_array[idx]

    if abs(temp_K - closest_temp) > 1:  # If the difference is greater than 1 degree
        print('WARNING: Moderator extension of {} corresponds to a temperature of {} K. Target moderator temperature is {} K'.format(
            ext, closest_temp, temp_K))

    return str(ext)



