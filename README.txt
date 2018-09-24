README 								Generated: 16122017 18:43 Hrs

==============================================================================================
Table of contents
==============================================================================================
1. Introduction to project files and version information
2. data.py
3. NLL.py
4. Minimiser.py
==============================================================================================
1. Introduction to project files and version information
==============================================================================================
The following files relevant to the project are present in the same foder as this README file.
The order you should open them should be data.py, NLL.py, Minimiser.py, which should follow
the order of the project script. Each of the following sections describes the code and 
functions being used to generate results. Every single code/ function description is also 
present on the respective .py files, if you wish to dive in straight. For your ease, pressing 
the run button on each code on Spyder should generate the relevant results without any extra 
effort from you. Feel free to play around with any parameters after you've understood the 
code/function and arguments.

For this project, the following versions and imports were used.

-Spyder (Python 2.7)
-Numpy (V. 1.12.1)
-Scipy (V. 0.19.0)
-Matplotlib (V. 2.0.2)
-Unicodescv (V. 0.14.1)

==============================================================================================
2. data.py
==============================================================================================
data.py contains code relevant to retrieving the data from lifetime.txt and producing the 
fit functions of the D meson signal and Background signal. The lifetime.txt was converted
to a csv file for ease of use. Thus, the data extraction is via a csvreader code. The following
functions are included here:
==============================================================================================
plot(samp)
'''
    Plots the histogram distribution of an input set of data (samp)
    Args:
        samp:   An array/list containing the data for which the histogram distribution 
                is plotted for 
    Example: plot(t); plot(sigma)
'''
==============================================================================================
fitSig(t, sigma, tau)
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
==============================================================================================
fitBG(t, sigma)
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
==============================================================================================
plotFit(t,sigma, tau)
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
==============================================================================================
3. NLL.py
==============================================================================================
NLL.py contains code that's used for generating the NLL functions, for both the scenario with/
without background effects. Addtional NLL functions also exist that accept a lesser range
of (t,sigma) data from the lifetime.csv file. Lastly, functions exists to produce plots of the 
NLL vs tau/ tau and a; as well as a cosh(x) function which is used to validate the parabolic
minimiser later. 
==============================================================================================
cosh(coord)
'''
    Returns the cosh(x) function for a taken x value .
    Args:
        coord:      A list containing the floating point x value(s) for which cosh is computed
    Returns:
        np.cosh(x): The cosh(x) value for the input x argument
    Example:>>> Cosh(0.0)
            1.0
'''
==============================================================================================
NLL(coord)
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
==============================================================================================
NLL5000(coord)
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
==============================================================================================
NLL2500(coord)
'''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with only the first 2500 data from a fixed set of t,sigma data 
    (see "from data import...") and the arg (coord) value.
'''
==============================================================================================
NLL1250(coord)
'''
    Returns the Negative Log Likelihood (NLL) for a taken tau (coord) value from a PDF 
    evaluated with only the first 1000 data from a fixed set of t,sigma data 
    (see "from data import...") and the arg (coord) value.
'''
==============================================================================================
NLLplot(tau)
'''
    Plots the distribution of the Negative Log Likelihood (NLL) against tau, 
    for a taken list/array of tau values from PDFs evaluated with a fixed set of 
    t,sigma data (see "from data import...") and the arg (tau) values.
    Example: taulist = np.linspace(0.05,1,40)
             >>> NLLplot(taulist) 
'''
==============================================================================================
NLLBG(coord)
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
==============================================================================================
NLLplotBG(coord)
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
==============================================================================================
NLLBGcontour(coord)
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
==============================================================================================
4. Minimiser.py
==============================================================================================
Minimiser.py contains all the minisation codes, as well as the validation codes. 
Function codes that are repeatedly used are also present, for example, the hessian and 
gradient vector function, to avoid repetition of code in different functions (waste space)
==============================================================================================
paramin(func,x)
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
==============================================================================================
curvature(func, x)
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
==============================================================================================
bisect(func, x, x_a)
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
==============================================================================================
grad_vector(f, initialguess)
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
==============================================================================================
hessian(f, initialguess)
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
==============================================================================================
stan_dev(f, min_coord)
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
==============================================================================================
gradient(f, initialguess, alpha)
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
==============================================================================================
newton(f, initialguess,gamma)
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
==============================================================================================
quasi_newton(f, initialguess,alpha)
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
==============================================================================================
sim_anneal(f, initialguess)
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
==============================================================================================
nelder_mead(f, initialguess)
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
==============================================================================================
Validation codes for the above functions:
    See online documentation for the scipy.optimize.fmin function
    The first call, sp.optimize.fmin(NLL,[0.4]) validates the result of the "paramin"
    function on the NLL function with initial guess of [0.4].
    The second call, sp.optimize.fmin(NLLBG,[0.4,0.9]) validates the results of the
    multidimensional minimisers (gradient,newton,quasi_newton,sim_anneal,nelder_mead)
    on the NLLBG function with initial guess of [0.4,0.9].
==============================================================================================
