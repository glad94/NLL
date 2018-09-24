# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:35:54 2017

@author: Gerald Lim
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from NLL import cosh,NLL, NLL5000, NLL2500, NLL1250, NLLBG
#Note: This is different from numpy.random 
import operator
import scipy as sp
import time

#==============================================================================
#Parabolic Minimiser
#==============================================================================
def paramin(func,x):  
    '''
    Prints the minimum x-position of an input function "func" along with the minimum
    value of the input function "func" computed via Parabolic minimisation; 
    and the number of iterations taken to converge to that minimum position.
    Args:
        func:       The mathematical function to be minimised. 
        x:          An array of discrete x values for which the argument function is 
                    a function of. 
    Prints:
        x3:         The minimum "x"-coordinate of the minimised function
        y3:         The minimum function value at x3 i.e. func(x3)
        iteration:  The number of iterations taken to converge to the minimum x3
    Returns:
        min_x:      The list of all the iterated minimum x-coordinates including the initial 3 points of the parabolic interpolant 
        min_y:      The list of all the iterated minimum function values including the initial 3 points of the parabolic interpolant 
    Example: >>> xlist = np.linspace(-4,4,40) 
             >>> paramin(cosh,xlist)
             Parabolic:  x                Cosh(x)
             Minimum:  -5.55571895195e-12   1.0
             Iteration:  5
             ([0.51282051282051277, 0.30769230769230749, 0.1025641025641022, 0.0083106849749399556, 0.00059405926370980006, 4.2411486356672479e-06, 1.845513413034599e-09, -5.5557189519521401e-12],
              [1.1343995299654923, 1.0477119283533722, 1.0052643099208489, 1.0000345339411401, 1.0000001764532096, 1.0000000000089937, 1.0, 1.0])

    '''   
    #Incredibly inefficient way of printing out different headings...I know -__-
    if func == cosh:
        print "Parabolic:  x                Cosh(x)"
    elif func == NLL:
        print "Parabolic:  tau              NLL"
    elif func == NLL5000:
        print "Parabolic:  tau              NLL (Only 5000 measurements)"
    elif func == NLL2500:
        print "Parabolic:  tau              NLL (Only 2500 measurements)"
    elif func == NLL1250:
        print "Parabolic:  tau              NLL (Only 1250 measurements)"
    else:
        print "Parabolic:  x       f(x)"
    #Iterate until an x-coordinate near the min-point is reached
    #Take that as x0, and the subsequent two points as x1 and x2
    y = func([x])
    for i in range(0, len(x)):
        if y[i] < y[i+1]:
            x0 = x[i] 
            y0 = y[i]
            x1 = x[i+1]
            y1 = y[i+1]
            x2 = x[i+2]
            y2 = y[i+2]
            break          
            
    X = [x0,x1,x2]
    Y = [y0,y1,y2]
    C = zip(X,Y)
    C.sort(key=operator.itemgetter(1))
    
    min_x = [C[2][0],C[1][0],C[0][0]]
    min_y = [C[2][1],C[1][1],C[0][1]]
    C.sort()

    #Count the number of iterations I've run 
    iteration = 0
    #Checks if the difference in x3 is within the desired error yet
    #Resolution of data (t, sigma) is 0.000001 (picosec)
    while (abs(min_x[len(min_x)-1] - min_x[len(min_x)-2]) > 0.000001): #or miny[len(miny)-1] != miny[len(miny)-2] ):
        
        x3 = 0.5* ((C[2][0]**2-C[1][0]**2)*C[0][1] + (C[0][0]**2-C[2][0]**2)*C[1][1] + (C[1][0]**2-C[0][0]**2)*C[2][1]) /   \
            ((C[2][0]-C[1][0])*C[0][1] + (C[0][0]-C[2][0])*C[1][1] + (C[1][0]-C[0][0])*C[2][1])
        
        y3 = func([x3])
        
        #Add the minima (x,y) into the coordinate list 
        C.append((x3, y3))
        #Order in ascending y 
        C.sort(key=operator.itemgetter(1))
        #Remove the coords with largest y value for next interpolation
        del C[3]
        #Order back in ascending x
        C.sort()
        #Store the new minima x3 into the list of x3s
        min_x.append(x3)
        min_y.append(y3)
        
        #print "Iteration Number: ", iteration
        #print "min tau (x): ",x3
        #print "min NLL (y): ",y3
        iteration += 1 
    print "Minimum: ",x3, y3
    print "Iteration: ", iteration
    
    return min_x, min_y   

#==============================================================================
#Find Standard Deviation via Curvature
#============================================================================== 
def curvature(func, x):
    '''
    Prints the output of the paramin function and also prints the computed
    standard deviation value from the paramin function, computed via the curvature of 
    the last parabolic estimate at the minimum position.
    Args:
        func:       The mathematical function to be minimised. 
        x:          An array of discrete x values for which the argument function is 
                    a function of. 
    Prints:
        see paramin
        sd1:         The standard deviation of the minimised x-coordinate 
    Example: >>> xlist = np.linspace(-4,4,40) 
             >>> curvature(cosh,xlist)
             Parabolic:  x                Cosh(x)
             Minimum:  -5.55571895195e-12 1.0
             Iteration:  5
             Standard Deviation (Curvature):  0.999999993903
             ===============================================

    '''
    min_x, min_y = paramin(func, x)
    startTime = time.time()  #For function execution timing
    
    #Find my Accuracy Levels
    #Keep the coordinates that produce the lagrange interpolant that gives the converging
    #result for y3, x3
    last_x_points = min_x[len(min_x)-4:len(min_x)-1]
    last_y_points = min_y[len(min_y)-4:len(min_y)-1]
    
    #METHOD 1 (CURVATURE)
    #Find the curvature of the quadratic interpolant (The curvature of the last 
    #parabolic estimate can be taken as the error) SEE Y2 STATS LECTURE 8 (Top Page2) 
    D = zip(last_x_points,last_y_points)
    D = sorted(D, key = lambda x: x[0])
    curvature = 2*((D[2][0]-D[1][0])*D[0][1] + (D[0][0]-D[2][0])*D[1][1] + (D[1][0]-D[0][0])*D[2][1])/  \
                    ((D[1][0]-D[0][0])*(D[2][0]-D[0][0])*(D[2][0]-D[1][0]))
    
    #In the stats notes, definition is a postive log likelihood, so no negative sign here               
    
    sd1 = np.sqrt(1./curvature)   
    print "Standard Deviation (Curvature): ", sd1
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="
    
#==============================================================================    
#Find Standard Deviation via Bisection Method (root finding)
#============================================================================== 
def bisect(func, x, x_a):
    '''
    Prints the output of the paramin function and also prints the computed
    standard deviation value from the paramin function, computed via the bisection
    root finding method of the function shifted vertically down by (its minimum value
    + 0.5).
    Args:
        func:       The mathematical function to be minimised. 
        x:          An array of discrete x values for which the argument function is 
                    a function of. 
        x_a:        The (float) guess coordinate of 
                    the coordinate point of the standard deviation/ root             
    Prints:
        see paramin
        sd1:         The standard deviation of the minimised x-coordinate 
    Example: >>> taulist = T = np.linspace(0.05,1,40) 
             >>> bisect(NLL,taulist,0.41)
             Parabolic:  tau              NLL
             Minimum:  0.404545786642 6220.44689279
             Iteration:  6
             Standard Deviation (Bisection):  0.00473848101429
             ===============================================

    '''
    min_x, min_y = paramin(func, x)
    startTime = time.time()  #For function execution timing
    #The tau and NLL values of the minimum
    min_tau = min_x[len(min_x)-1]
    min_NLL = min_y[len(min_y)-1]
    y0 = -0.5
    x0 = min_tau
 
    #Caclculate the midpoint (Left Side)
    old_root = 0
    new_root = 1
    while (abs(new_root-old_root) > 0.000001):
        old_root = new_root
        new_root = (x0+x_a)/2.
        if y0*(func([new_root])- (min_NLL + 0.5)) > 0:
            x0  = new_root
        else:
            x_a = new_root
                
    print "Standard Deviation (Bisection): ", abs(min_tau - new_root)
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="       


#WELCOME TO THE 2D WORLD! (Darn you Background Signal)
#==============================================================================
#Gradient Vector Function
#==============================================================================
def grad_vector(f, initialguess):
    '''
    Returns the gradient array (vector) of an input function (f) at the 
    coordinate position array (initialguess). This is called during the minimisation
    schemes where the iterative step involves the substraction of a certain amount
    the gradient vector of the function at the currently iterated position. 
    Args:
        f:              The function for which the gradient array is evaluated 
        initialguess:   The coordinate list/array for which the gradient array is evaluated at
    Returns:
        np.array(del_f): The gradient array of the function (f) evaluated at coordinates (initialguess)
        
    Example: >>> grad_vector(NLLBG,[0.404,1.0])
             array([ -24.75748591,  266.2870576 ])
    '''
    N = len(initialguess)
    #stepsize
    h = 10**-3
    #Create my array which stores the function gradient
    del_f = []
    #Calculate the first order derivate of f w.r.t to each variable 
    for i in range(0,N):
        step = np.array([0.0 for x in range(0,N)])
        step[i] = h
        #Using central difference scheme
        df_dvar = (f(initialguess+step)-f(initialguess-step))/(2*h)
        del_f.append(df_dvar) 
    return np.array(del_f)

#==============================================================================
#Hessian Function
#==============================================================================
def hessian(f, initialguess):
    '''
    Returns the Hessian array (matrix) of an input function (f) at the 
    coordinate position array (initialguess). This is called during the Newton method
    minimisation scheme when the Hessian is evaluated at each iterative step. Also 
    called when the "stan_dev" function (see below) is called to evaluate the Covariance
    matrix of the function (f) at the coordinate position array (initialguess)
    Args:
        f:              The function for which the Hessian array is evaluated 
        initialguess:   The coordinate list/array for which the Hes array is evaluated at
    Returns:
        H:              The Hessian array of the function (f) evaluated at coordinates (initialguess)
        
    Example: >>> hessian(NLLBG,[0.404, 1.0])
             [[45354.589815360669, 14612.476004685959],
              [14612.476004685959, 27300.253012981557]]
    '''
    #Vector length
    N = len(initialguess)
    #stepsize
    h = 10**-3
    
    #Initialise the H matrix
    H = [[0.0] * N for i in range(0, N)]
    #Iterate through row by row
    for i in range(0,N):
        for j in range(i,N):
            if i == j:
                step = np.array([0.0 for x in range(0,N)])
                step[j] = h
                #Using central difference scheme
                d2f_dvar2 = (f(initialguess+step)-2*f(initialguess)+f(initialguess-step))/(h**2)
                H[i][j] = d2f_dvar2
            else:
                step = np.array([0.0 for x in range(0,N)])
                step[j] = h
                step1 = np.array([0.0 for x in range(0,N)])
                step1[i] = h
                #Calculate the mixed derivatives
                d2f_dxdy = (f(initialguess+step+step1)-f(initialguess+step-step1)-f(initialguess-step+step1)+f(initialguess-step-step1))/(4*h**2)
                H[i][j] = d2f_dxdy
                H[j][i] = d2f_dxdy #symmetry
    return H

#==============================================================================
#Ellipse Plotting, Standard Deviation
#==============================================================================
def stan_dev(f, min_coord):
    '''
    Prints the standard deviation array of an input function (f) at its minimum
    coordinate position (min_coord). This is called at the end of the multidimensional
    minimisation functions: gradient; newton; quasi_newton; sim_anneal; nelder_mead.
    Standard deviation is found via evaluation of the Covariance matrix at the minimum
    (min_coord) position and involves calling of the above "hessian" function.If the 
    minimised function has two free parameters (len(min_coord) = 2), the uncertainty
    ellipse is also plotted
    Args:
        f:              The function for which the standard deviation is evaluated 
        min_coord:      The coordinate list/array for which the standard deviation is evaluated at
    Prints:
        np.array(sd):   The standard deviation array of the function (f) at its minimum
        
    Example: >>> stan_dev(NLLBG, [0.4097,0.983])
            Stand. Dev:  [ 0.00549645  0.00861018]
    '''
    N = len(min_coord)
    #Find the standard deviation
    sd = []
    #Covariance Matrix
    cov_matrix = np.linalg.inv(hessian(f, min_coord))
    for i in range(0,N):
        sd.append(np.sqrt(cov_matrix[i][i]))
    print "Stand. Dev: ", np.array(sd)
    #print "Error Matrix: ", cov_matrix
    #Find the eigenvalues and eigenvectors of the covariance matrix
    e_values, e_vectors = np.linalg.eig(cov_matrix)
    #Find the angle of the SD ellipse
    angle = np.arccos(abs(e_vectors[0,0]))
    #Find the widt/ height of the major/minor axis
    axes = np.sqrt(e_values)
    #print axes
    if N == 2:
        #Plot the standard deviation ellipse
        fig = plt.figure(2)
        ax = fig.add_subplot(111, aspect='equal')
        ell = Ellipse(xy=min_coord, width=axes[0]*2, height=axes[1]*2, angle=np.rad2deg(angle), fill=False,
            edgecolor="red")
        ax.add_artist(ell)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$a$')
        ax.set_title(r"Confidence interval of the minimum value")
        ax.set_xlim(0.4, 0.42)
        ax.set_ylim(0.97, 0.995)

#==============================================================================
#Gradient Method
#==============================================================================
def gradient(f, initialguess, alpha):
    '''
    Prints the minimum position of a minimised input function (f) via the iterative 
    Gradient method, evaluated from an intial guess coordinate position (initialguess)
    with input alpha (alpha) coefficient parameter (alpha << 1). Involves the calling 
    of the "grad_vector" function in each iterative step and the "stan_dev" function at 
    the end.
    
    Args:
        f:              The function to be minimised via the Gradient method 
        initialguess:   The initial guess coordinate list/array for which the 
                        minimisation method starts iterating from
        alpha:          The (float) alpha coefficient that is multiplied to the gradient
                        vector during each iterative step (alpha << 1)
    Prints:
        min_coords[-1]: The coordinate array of the minimum position of the function 
                        f. min_coords is a list storing the minimum coordinate evaluated
                        during each iterative step, with min_coords[-1] giving the last
                        evaluated coordinate before convergence
        it:             The number of iterations taken for the converged minimum coordinate
                        to be generated
        np.array(sd):   see stan_dev
        elapsedTime:    Time taken for the function to execute
        
    Example: >>> gradient(NLLBG,[0.404, 1.0],0.00003)
            Gradient:        tau          a
            Minimum:  [ 0.40968558  0.98367961]
            Iteration:  20
            Stand. Dev:  [ 0.00549268  0.00856228]
            Time taken:  8.51399993896  s
            ===============================================
    '''
    
    startTime = time.time()  #For function execution timing
    print "Gradient:        tau          a"
    #Where tau,a is my initial guess coordinates (in list form) [tau,a]

    #Store every successively generated minimum coordinate into a list
    min_coords = [np.array(initialguess)]
    it = 1
    
    def iterate():
        new_min = 0
        #Create my array which stores the function gradient
        del_f = grad_vector(f,min_coords[-1])
        #Iterate by negative of gradient*some coefficient
        new_min = min_coords[-1] - alpha*del_f
        min_coords.append(new_min)
    #First iteration
    iterate()
    
    #Iterate until convergence 
    while (np.linalg.norm(min_coords[-2]-min_coords[-1]) > 0.000001):
        iterate()
        it += 1    
        
    print "Minimum: ",min_coords[-1] 
    print "Iteration: ", it
    
    stan_dev(f,min_coords[-1])
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="     

#==============================================================================
#Newton's Method
#==============================================================================
def newton(f, initialguess,gamma):
    '''
    Prints the minimum position of a minimised input function (f) via the iterative 
    Newton's method, evaluated from an intial guess coordinate position (initialguess)
    with input gamma (gamma) relaxation parameter (gamma =< 1). Involves the calling 
    of the "grad_vector" and "hessian" functions in each iterative step and the 
    "stan_dev" function at the end.
    
    Args:
        f:              The function to be minimised via Newton's method 
        initialguess:   The initial guess coordinate list/array for which the 
                        minimisation method starts iterating from
        gamma:          The (float) gamma relaxation parameter that is multiplied to the gradient
                        vector during each iterative step (gamma =< 1)
    Prints:
        min_coords[-1]: The coordinate array of the minimum position of the function 
                        f. min_coords is a list storing the minimum coordinate evaluated
                        during each iterative step, with min_coords[-1] giving the last
                        evaluated coordinate before convergence
        it:             The number of iterations taken for the converged minimum coordinate
                        to be generated
        np.array(sd):   see stan_dev
        elapsedTime:    Time taken for the function to execute
        
    Example: >>> newton(NLLBG,[0.404, 1.0],1.0)
            Newton:        tau          a
            Minimum:  [ 0.40968615  0.98367842]
            Iteration:  11
            Stand. Dev:  [ 0.0054927   0.00856236]
            Time taken:  15.2380001545  s
            ===============================================
    '''
    startTime = time.time()
    print "Newton:        tau          a"
    #Where tau,a is my initial guess coordinates (in list form) [tau,a]
   
    #Store every successively generated minimum coordinate into a list
    min_coords = [np.array(initialguess)]
    it = 1
    
    def iterate():
        new_min = 0
        #Create my array which stores the function gradient
        del_f = grad_vector(f,min_coords[-1])
        #Create my Inverse Hessian
        H = hessian(f, initialguess)
        invH = np.linalg.inv(H)
        #Iterative step
        new_min = min_coords[-1] - gamma*np.dot(invH, del_f)
        min_coords.append(new_min)
        
    
    iterate()
    #Iterate until convergence 
    while (np.linalg.norm(min_coords[-2]-min_coords[-1]) > 0.000001):
        iterate()
        it +=1
        
    print "Minimum: ",min_coords[-1] 
    print "Iteration: ", it
    stan_dev(f,min_coords[-1])
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="

#==============================================================================
#Quasi-Newton Method
#==============================================================================
def quasi_newton(f, initialguess,alpha):
    '''
    Prints the minimum position of a minimised input function (f) via the iterative 
    Quasi-Newton method, evaluated from an intial guess coordinate position (initialguess)
    with input alpha (alpha) coefficient parameter (alpha << 1). Involves the calling 
    of the "grad_vector" function in each iterative step and the "stan_dev" function at the end.
    
    Args:
        f:              The function to be minimised via the Quasi-Newton method 
        initialguess:   The initial guess coordinate list/array for which the 
                        minimisation method starts iterating from
        alpha:          The (float) alpha coefficient that is multiplied to the gradient
                        vector during each iterative step (alpha << 1)
    Prints:
        min_coords[-1]: The coordinate array of the minimum position of the function 
                        f. min_coords is a list storing the minimum coordinate evaluated
                        during each iterative step, with min_coords[-1] giving the last
                        evaluated coordinate before convergence
        it:             The number of iterations taken for the converged minimum coordinate
                        to be generated
        np.array(sd):   see stan_dev
        elapsedTime:    Time taken for the function to execute
        
    Example: >>> quasi_newton(NLLBG,[0.404, 1.0],0.00003)
            Quasi-Newton:     tau          a
            Minimum:  [ 0.40968559  0.9836796 ]
            Iteration:  20
            Stand. Dev:  [ 0.00549268  0.00856228]
            Time taken:  8.7349998951  s
            ===============================================
    '''
    startTime = time.time()
    print "Quasi-Newton:     tau          a"
    #Where tau,a is my initial guess coordinates (in list form) [tau,a]
    N = len(initialguess) 
    #Store every successively generated minimum coordinate into a list
    min_coords = [np.array(initialguess)]
    it = 1
    
    #Initialise my inverse Hessian estimate (I matrix)
    global G
    G = [[0.0] * N for i in range(0, N)]
    for i in range(0, N):
        G[i][i] = 1.0
    #Very first iteration
    del_f = grad_vector(f,min_coords[-1])
    gradient_list = [del_f]
    
    new_min = min_coords[-1] - alpha*del_f
    min_coords.append(new_min)
    
    def iterate():
        global G
        new_min = 0
        #Create the delta_n vector
        del_n = min_coords[-1] - min_coords[-2]
        #Create my array which stores the function gradient
        del_f1 = grad_vector(f,min_coords[-1])
        gradient_list.append(del_f1)
        #Create the gamma_n vector
        gamma_n = gradient_list[-1] - gradient_list[-2]
        #DFP update algoritm
        G = G+ np.outer(del_n,del_n)/np.dot(gamma_n,del_n) - np.dot(np.dot(G,np.outer(del_n,del_n)),G)/np.dot(np.dot(gamma_n,G),gamma_n)
        #Iterative step
        new_min = min_coords[-1] - alpha*np.dot(G, del_f1)
        min_coords.append(new_min)
    
    while (np.linalg.norm(min_coords[-2]-min_coords[-1]) > 0.000001):
        iterate()
        it +=1

    print "Minimum: ",min_coords[-1] 
    print "Iteration: ", it
    stan_dev(f,min_coords[-1])
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="

#==============================================================================
#Simulated Annealing (More applicable to more complex functions with 
#multiple minima)
#==============================================================================

def sim_anneal(f, initialguess):
    '''
    Prints the minimum position of a minimised input function (f) via the heuristic 
    Simulated Annealing method; which uses the Metropolis Algorithm, evaluated from 
    an intial guess coordinate position (initialguess). Involves the calling of the "stan_dev" function at the end.
    
    Args:
        f:              The function to be minimised via Simulated Annealing 
        initialguess:   The initial guess coordinate list/array for which the 
                        minimisation method starts iterating from
    Prints:
        initialguess:   The coordinate array of the minimum position of the function 
                        f. The original initialguess array is equated to a new array solution
                        during each step of the minimiser depending on a probability
                        value as determined by the Metropolis Algorithm.
                        This initialguess is this the last evaluated coordinate before the 
                        synthetic temperature of the annealing scheme reaches its minimum value
        it:             The number of iterations taken for the converged minimum coordinate
                        to be generated
        np.array(sd):   see stan_dev
        elapsedTime:    Time taken for the function to execute
        
    Example: >>> sim_anneal(NLLBG,[0.404, 1.0])
            Sim Annealing:    tau          a
            Minimum:  [ 0.40836024  0.98489418]
            Iteration:  688
            Stand. Dev:  [ 0.00544948  0.00848285]
            Time taken:  78.8710000515  s
            ===============================================
    '''
    startTime = time.time()
    print "Sim Annealing:    tau          a"
    old_cost = f(initialguess)
    T = 1
    T_min = 10**-3
    step = 0.99
    N = len(initialguess)
    it = 0
    
    while T > T_min:
        new_sol = initialguess + np.random.uniform(-0.01,0.01,N)
        new_cost = f(new_sol)
        
        if np.exp((old_cost-new_cost)/T) > np.random.uniform(0,1):
            old_cost = new_cost
            initialguess = new_sol
            
        it += 1
        #Using a geometric Temperature Schedule
        T *= step
    print "Minimum: ",initialguess
    print "Iteration: ", it
    stan_dev(f,initialguess)
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="

#==============================================================================
#Nelder-Mead (The algorithm implemented in scipy.optimize.fmin)
#==============================================================================
def nelder_mead(f, initialguess):
    '''
    Prints the minimum position of a minimised input function (f) via the Nelder-Mead
    method (which is also the algorithm used in the scipy.optimize.fmin function which
    will be used for function validation), evaluated from an intial guess coordinate 
    position (initialguess). Involves the calling of the "stan_dev" function at the end.
    
    Args:
        f:              The function to be minimised via the Nelder-Mead method 
        initialguess:   The initial guess coordinate list/array for which the 
                        minimisation method starts iterating from
    Prints:
        zipped[0][1] :  The coordinate array of the minimum position of the function 
                        f. zipped is the list that stores the evaluated function values
                        and the corresponding simplex coordinates in ascending order
                        of the function value. For example, zipped[0] = (f(x,y), array
                        ([x, y])). zipped[0][1] thus represents the minimum coordinate
                        values of the function f at the last iterative/convergence.
        it:             The number of iterations taken for the converged minimum coordinate
                        to be generated
        np.array(sd):   see stan_dev
        elapsedTime:    Time taken for the function to execute
        
    Example: >>> nelder_mead(NLLBG,[0.404, 1.0])
            Nelder Mead:    tau        a
            Minimum:  [ 0.40968399  0.98368242]
            Iteration:  35
            Stand. Dev:  [ 0.00549262  0.00856209]
            Time taken:  7.85400009155  s
            ===============================================
    '''
    startTime = time.time()
    print "Nelder Mead:    tau        a"
    #stepsize
    h = 0.002
    #Create my initial simplex
    N = len(initialguess) 
    simplex_coords = [np.array(initialguess)]
    simplex_fs = [f(initialguess)]
    #Orthonormal conditions
    for i in range(0,N):
        step = np.array([0.0 for x in range(0,N)])
        step[i] = h
        coord = simplex_coords[0] + step
        simplex_coords.append(coord)
        simplex_fs.append(f(coord))
    #Zip the two lists together and sort according to ascending f(x)
    zipped = zip(simplex_fs,simplex_coords)
    zipped.sort()
    it = 0
    #print zipped
    while (np.linalg.norm(zipped[0][1]-zipped[1][1]) > 0.000001):
        #Compute Centroid
        C = sum([i[1] for i in zipped[0:N]])/N
        x_worst = zipped[N][1]
        #Transformation
        xr = 2*C - x_worst     
        f_xr = f(xr)
        
        #Next step depedent on the reflected point
        #Check if the function at xr is lower than the 2nd worst but greater than the best
        if f_xr < zipped[N-1][0] and f_xr >= zipped[0][0]:
            zipped[N] = (f_xr,xr)
            
        #Check if the function at xr is lower than the best    
        elif f_xr < zipped[0][0]:
            #Expand! Greedy Optimisation
            xe = C + 2*(xr-C)
            f_xe = f(xe)
            if f_xe < f_xr:
                zipped[N] = (f_xe,xe)
            else:
                zipped[N] = (f_xr,xr)
                
        #Otherwise if the reflection point was greater than the 2nd highest point
        elif f_xr >= zipped[N-1][0]:
           #Contract!!
           xc = C + 0.5*(x_worst-C)
           f_xc = f(xc)
           if f_xc < zipped[N][0]:
               zipped[N] = (f_xc,xc)
           else:
               #Shrinkage! Redefine the simplex and start over 
               x_best = zipped[0][1]
               x_rank = [i[1] for i in zipped]
               new_x = [x_best + 0.5*(i-x_best) for i in x_rank]
               new_x[0] = x_best
               new_f = [f(i) for i in new_x]
               zipped = zip(new_f,new_x)
               
        else:
               #Exception encountered -->Shrinkage! Redefine the simplex and start over 
               x_best = zipped[0][1]
               x_rank = [i[1] for i in zipped]
               new_x = [x_best + 0.5*(i-x_best) for i in x_rank]
               new_x[0] = x_best
               new_f = [f(i) for i in new_x]
               zipped = zip(new_f,new_x)
        zipped.sort()           
          
        it += 1
        
    print "Minimum: ",zipped[0][1] 
    print "Iteration: ", it
    stan_dev(f,zipped[0][1])
    elapsedTime = time.time() - startTime
    print "Time taken: ",elapsedTime, " s"
    print "==============================================="
        
#==============================================================================
#Function Calls
#==============================================================================
#Discrete x points to compute (cosh)
xlist = np.linspace(-4,4,40)
#Discrete tau points to compute for
taulist = T = np.linspace(0.05,1,40)
print "==============================================="

paramin(cosh,xlist) 
#paramin(NLL,taulist)     #Not called because curvature/bisection functions already call paramin      
curvature(NLL,taulist)
curvature(NLL5000,taulist)   #Test the effect of fewer measurements on uncertainty
curvature(NLL2500,taulist) 
curvature(NLL1250,taulist)
bisect(NLL,taulist,0.41)       #Test the effect of fewer measurements on uncertainty
bisect(NLL5000,taulist,0.41)   
bisect(NLL2500,taulist,0.41)
bisect(NLL1250,taulist,0.41)


#Feel Free to use other initial guess values, alpha/gamma parameters! 

gradient(NLLBG,[0.404, 1.0],0.00003)
newton(NLLBG,[0.404, 1.0],1.0) 
quasi_newton(NLLBG,[0.404, 1.0],0.00003)  
sim_anneal(NLLBG,[0.404, 1.0]) 
nelder_mead(NLLBG,[0.404, 1.0])    

#==============================================================================
#Rosenbrock Test Function for the optimisation algorithms (Just for Fun)
#==============================================================================
def rosenbrock(coord):
    x,y = coord
    return (1.-x)**2 + 100*(y-x**2)**2

#nelder_mead(rosenbrock,[200.,200.])
    
#==============================================================================
#VALIDATION CODES 
#==============================================================================
print "Validation for Parabolic Minimiser"
'''
Validation codes for the above functions:
    See online documentation for the scipy.optimize.fmin function
    The first call, sp.optimize.fmin(NLL,[0.4]) validates the result of the "paramin"
    function on the NLL function with initial guess of [0.4].
    The second call, sp.optimize.fmin(NLLBG,[0.4,0.9]) validates the results of the
    multidimensional minimisers (gradient,newton,quasi_newton,sim_anneal,nelder_mead)
    on the NLLBG function with initial guess of [0.4,0.9].
'''
print sp.optimize.fmin(NLL,[0.4])
print "==============================================="
print "Validation for Multidimensional Minimisers"
print sp.optimize.fmin(NLLBG,[0.4,0.9])