# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:31:26 2017

@author: Gerald Lim
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from data import t,sigma,fitSig,fitBG

#==============================================================================
#Cosh Function
#==============================================================================
def cosh(coord):
    '''
    Returns the cosh(x) function for a taken x value .
    Args:
        coord:      A list containing the floating point x value(s) for which cosh is computed
    Returns:
        np.cosh(x): The cosh(x) value for the input x argument
    Example:>>> Cosh(0.0)
            1.0
    '''
    x = coord[0]
    return np.cosh(x)

#=============================================================================
#NLL Function
#=============================================================================
def NLL(coord):
    '''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with a fixed set of 10000 t,sigma data (see "from data import...") and the
    arg (coord) value.
    Args:
        coord: A list containing the floating point tau value(s) for which the NLL is computed
    Returns:
        NLL: The value of the NLL computed from the (t,sigma) dataset and input tau argument
    Example:>>> NLL([0.404])
            6220.45362533
            >>> taulist = np.linspace(0.05,1,80)
            >>> NLL([taulist])
            array([ 24155.94552593,  20743.52029484,  18165.04338598, ...,
                   8687.56466064,   8750.85271077,   8814.00166559])
    '''
    tau = coord[0]
    #Likelihood
    LL = 0.
    for i in range(0,len(t)):
        #Calculate the PDF for each t,sigma value and the tau value
        P = fitSig(t[i],sigma[i],tau)
        #Convert the evaluated PDF for each (t,sigma) measurement into natural log form 
        logP = np.log(P)
        #Sum all the negative log P to give the LL (positive log likelihood)    
        LL += logP
    NLL = -LL
    return NLL 

#=============================================================================
#NLL Function (With fewer measurements included --> 5000,2500,1000
#=============================================================================
def NLL5000(coord):
    '''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with only the first 5000 data from a fixed set of t,sigma data 
    (see "from data import...") and the arg (coord) value.
    Args:
        coordlist:  A list containing the floating point tau value(s) for which the NLL is computed
                    and integer value for which the dataset to draw measurements from is limited to
    Returns:
        NLL:        The value of the NLL computed from the limited (t,sigma) dataset and input tau argument
    Example:>>> NLL5000([0.404])  #
            3091.461273131531
    '''
    tau = coord[0]
    #Likelihood
    LL = 0.
    for i in range(0,len(t[:5000])):
        #Calculate the PDF for each t,sigma value and the tau value
        P = fitSig(t[i],sigma[i],tau)
        #Convert the evaluated PDF for each (t,sigma) measurement into natural log form 
        logP = np.log(P)
        #Sum all the negative log P to give the LL (positive log likelihood)    
        LL += logP
    NLL = -LL
    return NLL 

def NLL2500(coord):
    '''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with only the first 2500 data from a fixed set of t,sigma data 
    (see "from data import...") and the arg (coord) value.
    '''
    tau = coord[0]
    #Likelihood
    LL = 0.
    for i in range(0,len(t[:2500])):
        #Calculate the PDF for each t,sigma value and the tau value
        P = fitSig(t[i],sigma[i],tau)
        #Convert the evaluated PDF for each (t,sigma) measurement into natural log form 
        logP = np.log(P)
        #Sum all the negative log P to give the LL (positive log likelihood)    
        LL += logP
    NLL = -LL
    return NLL 

def NLL1250(coord):
    '''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with only the first 1000 data from a fixed set of t,sigma data 
    (see "from data import...") and the arg (coord) value.
    '''
    tau = coord[0]
    #Likelihood
    LL = 0.
    for i in range(0,len(t[:1250])):
        #Calculate the PDF for each t,sigma value and the tau value
        P = fitSig(t[i],sigma[i],tau)
        #Convert the evaluated PDF for each (t,sigma) measurement into natural log form 
        logP = np.log(P)
        #Sum all the negative log P to give the LL (positive log likelihood)    
        LL += logP
    NLL = -LL
    return NLL 

#=============================================================================
#Plots the NLL function against tau
#=============================================================================
def NLLplot(tau):
    '''
    Plots the distribution of the Negative Log Likelihood (NLL) against tau, 
    for a taken list/array of tau values from PDFs evaluated with a fixed set of 
    t,sigma data (see "from data import...") and the arg (tau) values.
    Example: taulist = np.linspace(0.05,1,40)
             >>> NLLplot(taulist) 
    '''
    #Evaluate the NLL for each tau value in the list/array of taus
    NLLs = NLL([tau])
    plt.figure()
    #Plot the NLL distribution against tau
    plt.plot(tau, NLLs, label = r"Negative Log Likelihoods")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"NLL")
    plt.legend()
    plt.title(r"NLL($\tau$) against $\tau$.")
    
#=============================================================================
#NLL with the background + D-meson signal
#=============================================================================
def NLLBG(coord):
    '''
    Returns the Negative Log Likelihood (NLL) for a taken coordinate (coord) value from a PDF 
    evaluated with a fixed set of t,sigma data (see "from data import...") and the
    arg (coord) value.
    Args:
        coord: A 2D array/list containing the floating point tau and a(ratio of the particle to background signal)
        value in the format [tau,a] or np.array([tau,a])
    Returns:
        NLL: The value of the NLL computed from the (t,sigma) dataset and input coord argument
    Example:>>> NLLBG([0.404,1.0])
            6220.45362533
    '''
    #Unzip the coordinate list/array into the tau and a component
    tau,a = coord 
    #Log Likelihood
    LL = 0.
    for i in range(0,len(t)):
        #Calculate the PDF for each t,sigma value and the tau and a value
        P = a*fitSig(t[i],sigma[i],tau) + (1-a)*fitBG(t[i],sigma[i])
        logP = np.log(P)
        #Sum all the negative log P to give the LL (positive log likelihood)    
        LL += logP
    NLL = -LL
    return NLL   

#=============================================================================
#Plots the NLL function (with background signal) against tau and ratio a
#=============================================================================
def NLLplotBG(coord):
    '''
    Plots the 3D distribution of the Negative Log Likelihood (NLL) against tau and a, 
    for a taken list of 2 arrays containing tau and a values from PDFs evaluated with 
    a fixed set of t,sigma data (see "from data import...") and the arg (coord) values.
    Args:
        coord: A 2D array/list containing the arrays of tau and a
        values in the format [my_array_of_taus,my_array_of_as] 
    Example: taulist = np.linspace(0.05,1,40)
             alist = np.linspace(0.01,1,40)
             >>> NLLplotBG([taulist,alist]) 
    '''
    #Unzip the coordinate list/array into the separate arrays
    tau, a = coord 
    Tau, A = np.meshgrid(tau , a)
    NLLs = NLLBG([Tau, A])
    
    #Plot the NLL distribution against tau and a
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Tau, A, NLLs, cmap=cm.jet)
    #Colour!!
    fig.colorbar(surf)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$a$')
    ax.set_zlabel(r'NLL')
    ax.set_title(r"NLL($\tau,a$) plotted against $\tau$ and $a$:")
    
#=============================================================================
#Plots the contour plot of NLLBG function against tau and ratio a
#=============================================================================
def NLLBGcontour(coord):
    '''
    Plots the contour plot of the Negative Log Likelihood (NLL) against tau and a, 
    for a taken list of 2 arrays containing tau and a values from PDFs evaluated with 
    a fixed set of t,sigma data (see "from data import...") and the arg (coord) values.
    This is useful to "zoom" into the minimum of the NLLBG which depends on 2 free param
    Args:
        coord: A 2D array/list containing the arrays of tau and a
        values in the format [my_array_of_taus,my_array_of_as] 
    Example: taucont = np.linspace(0.404,0.415,80)
             acont = np.linspace(0.976,0.99,80)
             >>> NLLBGcontour([taucont,acont])  
    '''
    #Unzip the coordinate list/array into the separate arrays
    tau, a = coord 
    Tau, A = np.meshgrid(tau , a)
    NLLs = NLLBG([Tau, A])
    
    plt.figure()
    conplot = plt.contour(Tau,A,NLLs,11)
    plt.clabel(conplot, inline = 1, fontsize = 10)
    plt.title(r"Contour Plot of NLL against $\tau$ and $a$")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$a$")
    plt.show()

#=============================================================================
#Function Calls
#==============================================================================

#Array of tau values to iterate over
taulist = np.linspace(0.05,1,80)
#Array of a values to iterate over 
alist = np.linspace(0.01,1,80) 

#For contour plot
taucont = np.linspace(0.404,0.415,80) 
acont = np.linspace(0.976,0.99,80)
if __name__ == "__main__":
    NLLplot(taulist)    
    NLLplotBG([taulist,alist]) 
    NLLBGcontour([taucont,acont]) 
    