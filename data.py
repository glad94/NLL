# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:29:03 2017

@author: Gerald Lim
"""
import unicodecsv
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
#=============================================================================
#Read my csv file containing the D-meson decay times/error
#=============================================================================
def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
    
Ddata = read_csv('lifetime.csv')


#Store my time and sigma data into respective lists
t = [float(data['t']) for data in Ddata]         #All the decay times
sigma = [float(data['sigma']) for data in Ddata] #Decay time uncertainties
#=============================================================================
#Plot my histogram of the D-meson decay time/error data
#=============================================================================
def plot(samp):
    '''
    Plots the histogram distribution of an input set of data (samp)
    Args:
        samp:   An array/list containing the data for which the histogram distribution 
                is plotted for 
    Example: plot(t); plot(sigma)
    '''
    N = len(samp)
    #Freedman-Diaconis to calculate number of bins to use
    IQR = stats.iqr(samp)
    binsize = 2*IQR/(N**(1./3))
    #Calculate the number of bins to use
    b = int((max(samp)-min(samp))/binsize)
 
    #Plotting of distribution 
    plt.figure(1)
    #Essentially labelling of axes depending if the plotted data is for t/ sigma
    if samp == t:
        plt.hist(samp, bins=b, label = r"$t_{i}$", normed = True)
        plt.xlabel("t (ps)")
        plt.title("Distribution of decay times $t_{i}$.")
        plt.legend()
    elif samp == sigma:
        plt.hist(samp, bins=b, label = r"$\sigma_{i}$")
        plt.xlabel("t (ps)")
        plt.title("Distribution of decay times $\sigma_{i}$.")
        plt.legend()
    plt.ylabel("Frequency") 
#=============================================================================
#Fit Function (Particle decay signal)
#=============================================================================
def fitSig(t, sigma, tau):
    '''
    Returns the Fit function/ PDF of a given t, sigma and tau value for the particle decay signal
    Args:
        t:      A (float) decay time value
        sigma:  A (float) uncertainty sigma value
        tau:    A (float) lifetime tau value
    Returns:
        fit: The evaluated value of the Fit function 
            
    Example: >>> fitSig(0.1,0.287,0.404)
            0.89212294152048199
    '''
    fit = (1./(2.*tau))*np.exp((sigma**2./(2.*tau**2.))-(t/tau))*special.erfc(((sigma/tau)-(t/sigma))/np.sqrt(2.))
    return fit
#=============================================================================
#Fit Function (Background signal)
#=============================================================================
def fitBG(t, sigma):
    '''
    Returns the Fit function of a given t, sigma value for the background signal
    Args:
        t:      A (float) decay time value
        sigma:  A (float) uncertainty sigma value
    Returns:
        fit: The evaluated value of the Fit function of the background signal
            
    Example: >>> fitBG(0.1,0.287)
            1.3081737508850964
    '''
    fit = np.exp(-(t**2.)/(2.*(sigma**2.)))/((np.sqrt(2*np.pi))*sigma)
    return fit
#=============================================================================
#Plots both fit functions against time  
#=============================================================================
def plotFit(t,sigma, tau):
    '''
    Plots the "fitSig" function for given list/array of t values, a sigma and tau value,
    against t
    Args:
        t:      An array of (float) decay time values
        sigma:  A (float) uncertainty sigma value
        tau:    A (float) lifetime tau value
    Example: T = T = np.linspace(-2,4,50)
             plotFit(T,0.287, 0.404)
    '''
    plt.figure(2)
    #plt.legend()
    #plt.legend(("Soo","f"), loc = "right")
    #plt.legend(("waa"), loc = "right")
    plt.plot(t, fitSig(t,sigma,tau), label = r"Fit Function")
    plt.title("Fit Function against $t$.")
    #plt.plot(t, fitBG(t,sigma), label = r"FitBG Function")
    #plt.title("FitBG Function against $t$.")
    plt.xlabel("t (ps)")
    plt.ylabel("Fit Function")


#=============================================================================
#Function Calls
#=============================================================================
if __name__ == "__main__":
    plot(t)    
    T = np.linspace(-2,4,50)
    plotFit(T,0.281744,0.404)  
    plotFit(T,0.001,0.404)      #~Zero sigma --> Negative Exponential
    plotFit(T,0.001,1.0)        #Greater tau --> Flatter peak
    plt.legend((r"$\sigma$ = 0.282, $\tau = 0.404$",r"$\sigma$ = 0.001, $\tau = 0.404$",r"$\sigma$ = 0.001, $\tau = 1.0$"))
# Inreasing tau causes a shorter/ narrower peak
# Increasing sigma causes a shorter peak, "spreads curve outwards left (negative time region)"
# At very small sigma, fitFunc is essentially an exponential curve (convolution is negligible)
