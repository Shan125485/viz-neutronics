import h5py
import matplotlib.pyplot as plot
import xarray as xr
import numpy as np
import logging
logger = logging.getLogger(__name__)

# Lookup search for ENDFB8 library ZAID extensions


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
    logger.info('WARNING: Material extension of %s corresponds to a temperature of %s K. Target material temperature is %s K', ext, closest_temp, temp_K)
    # print('WARNING: Material extension of {} corresponds to a temperature of {} K. Target material temperature is {} K'.format(
    #     ext, closest_temp, temp_K))

    return str(ext), closest_temp


def find_moder_extension(temp_C: float):
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
    logger.info('WARNING: Moderator extension of %s corresponds to a temperature of %s K. Target moderator temperature is %s K', ext, closest_temp, temp_K)
    # print('WARNING: Moderator extension of {} corresponds to a temperature of {} K. Target moderator temperature is {} K'.format(
    #     ext, closest_temp, temp_K))

    return str(ext), closest_temp



def plot_h5_data(h5_lib_name, isotope, outputFilepath):

    f = h5py.File('/home/st712/MaterialsLibrary/hdf5_files/' + h5_lib_name + '.h5', 'r')
    isotope = 'H2_D2O'
        
    # print(list(f.keys()))
    dset = f[isotope]
    print(list(dset.keys()))



        
    data_dict = {}

    xr.set_options(display_expand_data = True)

    ds = xr.Dataset()
    ds_long = xr.Dataset()

    for key in dset.keys():
        print(key + ': ' + str(dset[key].shape))
        data_dict[key] = np.array(dset[key])

        if isotope in ['H1_H2O', 'H2_D2O']:
            if dset[key].shape[1] == 56:
                da = xr.DataArray(np.array(dset[key]))
                ds[key] = da

            elif dset[key].shape[1] == 834:
                da_long = xr.DataArray(np.array(dset[key]))
                ds_long[key] = da_long


    # print(data_dict)


    ds = ds.rename_dims(dims_dict={'dim_0':'temperature', 'dim_1': 'energy'})
    ds_long = ds_long.rename_dims(dims_dict={'dim_0':'temperature', 'dim_1': 'energy'})
    # print(ds.data_vars)


    for var_name in ds.data_vars:
        print(var_name)

        data = ds[var_name]
        dim_size = ds.sizes


        fig, ax = plot.subplots()

        for temp_pos in range(0,dim_size['temperature']):
            data.isel(temperature=temp_pos).plot(x='energy', label='temperature '+ str(temp_pos), ax=ax)

        title = '{} for {}'.format(var_name, isotope)
        ax.set_xlabel('Energy group')
        ax.invert_xaxis()
        ax.set_ylabel(var_name) 
        plot.title(title)
        # plot.legend()
        plot.savefig(outputFilepath + '/{}_{}.svg'.format(isotope, var_name))
        plot.close()

    for var_name in ds_long.data_vars:
        print(var_name)
        data = ds_long[var_name]
        dim_size = ds_long.sizes


        fig, ax = plot.subplots()

        for temp_pos in range(0,dim_size['temperature']):
            data.isel(temperature=temp_pos).plot(x='energy', label='temperature '+ str(temp_pos), ax=ax)

        title = '{} for {}'.format(var_name, isotope)
        ax.set_xlabel('Energy group')
        ax.invert_xaxis()
        ax.set_ylabel(var_name) 
        plot.title(title)
        # plot.legend()
        plot.savefig(outputFilepath + '/{}_{}.svg'.format(isotope, var_name))
        plot.close()

    return