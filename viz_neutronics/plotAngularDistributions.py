import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


def plot_polar_Fe(csv_filename):
    df = pd.read_csv(csv_filename + '.csv')
    print(df.columns)
    print(df)

    theta_COM =np.deg2rad(df[['scatter angle (COM)']]).transpose()[:].to_numpy()[0,:]
    theta_LAB = np.deg2rad(df[['scatter angle (LAB)']]).transpose()[:].to_numpy()[0,:]
    r_high_E = df['E= 1.000069 MeV - angular differential XS (b/sr)']
    r_low_E = df['E=1 eV - Angular  differential XS (b/sr)']

    theta_LAB = np.concatenate((theta_LAB, np.flip(-theta_LAB)))
    r_high_E= np.concatenate((r_high_E, np.flip(r_high_E)))
    r_low_E= np.concatenate((r_low_E, np.flip(r_low_E)))

    print(theta_LAB.shape)
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_LAB,r_low_E, label = 'LAB, 1 eV' )
    plot.fill_between(theta_LAB, r_low_E, alpha=0.2)
    ax.plot(theta_LAB,r_high_E, label = 'LAB, 1 MeV' )
    ax.set_ylabel('barns/sr',loc='bottom')

    ax.grid(True)
    plot.fill_between(theta_LAB, r_high_E, alpha=0.5)
    
    fig.legend()

    plot.title(csv_filename + ' angular differential cross sections')
    plot.savefig(csv_filename + '_polar_plot.svg')

def plot_polar_O(csv_filename):
    df = pd.read_csv(csv_filename + '.csv')
    print(df.columns)
   

    theta_COM =np.deg2rad(df[['scatter angle (COM)']]).transpose()[:].to_numpy()[0,:]
    theta_LAB = np.deg2rad(df[['scatter angle (LAB)']]).transpose()[:].to_numpy()[0,:]
    # print(theta_LAB)
    # concatenate 
    theta_LAB = np.concatenate((theta_LAB, np.flip(-theta_LAB)))
    
    # print(theta_LAB)

    r_high_E = df['E= 1MeV - angular differential XS (b/sr)']
    r_low_E = df['E=100 eV - Angular differential XS (b/sr)2']

    r_high_E= np.concatenate((r_high_E, np.flip(r_high_E)))
    r_low_E= np.concatenate((r_low_E, np.flip(r_low_E)))

    print(theta_LAB.shape)
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_LAB,r_low_E, label = 'LAB, 1 eV' )
    plot.fill_between(theta_LAB, r_low_E, alpha=0.2)
    ax.plot(theta_LAB,r_high_E, label = 'LAB, 1 MeV' )
    ax.set_ylabel('barns/sr',loc='bottom')

    ax.grid(True)
    plot.fill_between(theta_LAB, r_high_E, alpha=0.5)
    
    fig.legend()

    plot.title(csv_filename + ' angular differential cross sections')
    plot.savefig(csv_filename + '_polar_plot.svg')

def plot_polar_H(csv_filename, mode='LAB'):
    df = pd.read_csv(csv_filename + '.csv')
    print(df.columns)
   

    if mode == 'LAB':
        theta = np.deg2rad(df[['scatter angle (LAB)']]).transpose()[:].to_numpy()[0,:]
        
    elif mode =='COM':
        print('COM')
        theta =np.deg2rad(df[['scatter angle (COM)']]).transpose()[:].to_numpy()[0,:]
        print(theta)
    theta = np.concatenate((theta, np.flip(-theta)))
    print(theta)
        
    r_high_E = df['E= 1 MeV - angular differential XS (b/sr)']
    r_low_E = df['E=1 eV - Angular  differential XS (b/sr)']

    r_high_E= np.concatenate((r_high_E, np.flip(r_high_E)))
    r_low_E= np.concatenate((r_low_E, np.flip(r_low_E)))

    print(theta.shape)
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta,r_low_E, label = mode+', 1 eV' )
    plot.fill_between(theta, r_low_E, alpha=0.2)
    ax.plot(theta,r_high_E, label = mode+', 1 MeV' )
    ax.set_ylabel('barns/sr',loc='bottom')

    ax.grid(True)
    plot.fill_between(theta, r_high_E, alpha=0.5)
    
    fig.legend()

    plot.title(csv_filename + ' angular differential cross sections')
    plot.savefig(csv_filename + '_' +mode + '_polar_plot.svg')

def plot_polar_energy(csv_filename, mode='LAB'):
    df = pd.read_csv(csv_filename + '.csv')
    print(df.columns)
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

    
    fig.legend()

    plot.title(csv_filename + ' energy change angular distribution')
    plot.savefig(csv_filename + '_' +mode + '_energy_angle.svg')


def plot_legendre(csv_filename):
    # check here https://demonstrations.wolfram.com/PolarPlotsOfLegendrePolynomials/

    df = pd.read_csv(csv_filename + '.csv')
    print(df.columns)
   

    mu = df['cosine (COM)']
    theta =np.deg2rad(df[['scatter angle (COM)']]).transpose()[:].to_numpy()[0,:]
    
    

    P0=df['P0']
    P1=df['P1']
    P2=df['P2']
    P3=df['P3']
    P4=df['P4']

    
    # correct for negative numbers

    theta1 = np.where(P1<0,theta + np.pi , theta)
    P1 = np.where(P1<0,-P1,P1)

    theta2 = np.where(P2<0,theta + np.pi , theta)
    P2 = np.where(P2<0,-P2,P2)

    theta3 = np.where(P3<0,theta + np.pi , theta)
    P3 = np.where(P3<0,-P3,P3)

    theta4 = np.where(P4<0,theta + np.pi , theta)
    P4 = np.where(P4<0,-P4,P4)

    # flip and concatenate
    theta = np.concatenate((theta, np.flip(-theta)))
    theta1 = np.concatenate((theta1, np.flip(-theta1)))
    theta2 = np.concatenate((theta2, np.flip(-theta2)))
    theta3 = np.concatenate((theta3, np.flip(-theta3)))
    theta4 = np.concatenate((theta4, np.flip(-theta4)))
    P0= np.concatenate((P0, np.flip(P0)))
    P1= np.concatenate((P1, np.flip(P1)))
    P2= np.concatenate((P2, np.flip(P2)))
    P3= np.concatenate((P3, np.flip(P3)))
    P4= np.concatenate((P4, np.flip(P4)))
    
    
    fig, ax = plot.subplots(subplot_kw={'projection': 'polar'})
    # ax.set_ylim(0,1)

    ax.plot(theta,P0, label = 'P0' )
    plot.fill_between(theta, P0, alpha=0.2)
    

    ax.plot(theta1,P1, label = 'P1' )
    plot.fill_between(theta1, P1, alpha=0.5)

    ax.plot(theta2,P2, label = 'P2' )
    plot.fill_between(theta2, P2, alpha=0.6)
   
    ax.plot(theta3,P3, label = 'P3' )
    plot.fill_between(theta3, P3, alpha=0.7)

    ax.plot(theta4,P4, label = 'P4' )
    plot.fill_between(theta4, P4, alpha=0.8)
   

    ax.grid(True)

    
    fig.legend()

    plot.title(csv_filename + ' angular differential cross sections')
    plot.savefig(csv_filename + '_' + '_polar_plot.svg')






if __name__=="__main__":
    # plot_polar_Fe('Fe-56')
    # plot_polar_O('O-16')
    # plot_polar_H('H-1', mode='COM')
    # plot_polar_H('H-1', mode='LAB')
    # plot_legendre('legendre')
    # plot_polar_energy('H-1')
    # plot_polar_energy('Fe-56')
    plot_polar_energy('O-16')

