import numpy as np
import pandas as pd
import matplotlib.pyplot as plot




def plot_polar_energy(csv_filename, mode='LAB'):
    df = pd.read_csv('inputs/' + csv_filename + '.csv')
    output_filename = 'outputs/' + csv_filename + '_' +mode + '_energy_angle.svg'

    if mode == 'LAB':
        theta = np.deg2rad(df[['scatter angle (LAB)']]).transpose()[:].to_numpy()[0,:]
        
    elif mode =='COM':
        print('COM')
        theta =np.deg2rad(df[['scatter angle (COM)']]).transpose()[:].to_numpy()[0,:]
    E = df['Ef/Ei']

    # reflect
    theta = np.concatenate((theta, np.flip(-theta)))
    E= np.concatenate((E, np.flip(E)))

    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta,E, label = mode, color='green' )
    plot.fill_between(theta, E, alpha=0.2, color='green')

    ax.set_ylabel('Final / initial energy',loc='bottom')

    ax.grid(True)

    
    # fig.legend()
    
    # plot.title(csv_filename + ' energy change angular distribution')
    plot.savefig(output_filename)
    print('Saved figure to ' + output_filename)

def theta2mu(theta_rad):
    """Convert angle in radians to cosine of angle"""
    return np.cos(theta_rad)

def calc_legendre_poly(mu, Pn_order = 0):
    """Takes in cosine of angle and returns arrays of the legendre polynomials of the given PN_order.
    Takes an integer order Pn_order up to Pn=4"""
    N = Pn_order
    if N >4:
        raise ValueError('Legendre polynomial order greater than 4 not implemented')
    elif N<0:
        raise ValueError('Legendre polynomial order must be non-negative')
    elif N==0:
        P0 = np.ones_like(mu)
        return P0
    elif N==1:
        P1 = mu
        return P1
    elif N==2:
        P2 = 0.5 * (3 * mu**2 - 1)
        return P2
    elif N==3:
        P3 = 0.5 * (5 * mu**3 - 3 * mu)
        return P3
    elif N==4:
        P4 = (35 * mu**4 - 30 * mu**2 + 3) / 8
        return P4
    else:
        raise ValueError('Legendre polynomial order must be an integer between 0 and 4')


def legendre_for_plotting(theta):
    theta_for_plotting = np.where(theta<0, -theta , theta) # reflect negative theta to positive side for plotting
    mu = theta2mu(-theta_for_plotting)
    print(mu)
    P0, P1, P2, P3, P4 = calc_legendre_poly(mu)
    return P0, P1, P2, P3, P4



def plot_legendre(npoints=500, Pn_order_list=[0,1,2,3,4]):
    """Plots the Legendre polynomials in polar coordinates up to N=4. Saved in outputs file.
    For reference, check here: https://demonstrations.wolfram.com/PolarPlotsOfLegendrePolynomials/

    Args:
        npoints (int, optional): The resolution of the plot - how many angles to plot between mu =-1 and mu=1. Defaults to 500.
        Pn_order_list (list, optional): List of integers between 0 and 4 which give the order of the Legendre functions to be plotted, . Defaults to [0,1,2,3,4].
    """

    output_filename = 'outputs/'  + 'legendre_' + str(Pn_order_list)+ '_polar_plot.svg'

    mu = np.linspace(-1,1,npoints)
    theta = np.arccos(mu)  # theta in radians from 0 to pi


    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
   
    for N in Pn_order_list:
        # Calculate Legendre polynomial for this order
        Pn = calc_legendre_poly(mu, N)

        # Where there are negative values of Pn, reflect theta to the other side and make Pn positive for plotting
        theta_plot = np.where(Pn<0, theta+np.pi, theta) 
        Pn_plot = np.where(Pn<0,-Pn,Pn) 

        # reflect to negative theta side for full polar plot
        theta_plot = np.concatenate((theta_plot, np.flip(-theta_plot)))
        Pn_plot = np.concatenate((Pn_plot, np.flip(Pn_plot)))
        
        # Plot and label
        ax.plot(theta_plot, Pn_plot, label = 'P'+str(N) )
        plot.fill_between(theta_plot, Pn_plot, alpha=0.6)

    ax.grid(False)

    fig.legend()
    plot.title('Legendre Polynomials Polar Plot')
    plot.savefig(output_filename)
    print('Saved figure to ' + output_filename)






if __name__=="__main__":
    plot_legendre(npoints=500, Pn_order_list=[0,2,4])
    # plot_polar_energy('H-1', mode='LAB')
    # plot_polar_energy('Fe-56')
    # plot_polar_energy('O-16')

