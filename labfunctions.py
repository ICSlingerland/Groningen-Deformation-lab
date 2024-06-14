from statsmodels.tsa.stattools import adfuller
import pandas as pd
import math
import numpy as np

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

def get_vertical(lof, omega, alpha_d, d_e, d_n):
    """
    Computes the vertical deformation.

    Parameters:
    lof : The line-of-sight deformation.
    omega : The incidence angle (in radians).
    alpha_d : The angle between the north and the range (in radians).
    d_e : The deformation component for east deformation.
    d_n : The deformation component for north deformation.

    Returns:
    float: The vertical deformation.
    """
    return (lof - (d_e* np.sin(omega) * np.sin(alpha_d)) - (d_n*np.sin(omega) * np.cos(alpha_d))) / np.cos(omega)
    
def overall_model_test(e0, Q_yy):
    """
    Overal model test

    Parameters:
    e0= residuals
    Q_yy: observation co variance matrix

    Returns
    float: Test statistic, critical value

    """

    T_q = e0.T @ np.linalg.inv(Q_yy) @ e0
    q= e0.size-4
    c2 = chi2.ppf(1 - 0.05, df=q)

    return T_q, c2


def lineofsight(omega, alpha_d, d_e, d_n, d_u):
    """
    Computes the line-of-sight deformation.

    Parameters:
    omega (float): The incidence angle (in radians).
    alpha_d (float): The angle between the north and the range (in radians).
    d_e (float): The deformation component for east deformation.
    d_n (float): The deformation component for north deformation.
    d_u (float): The deformation component for up deformation.

    Returns:
    float: The dot product of the line-of-sight vector and the deformation vector.
    """
    A = np.array([np.sin(omega) * np.sin(alpha_d), np.sin(omega) * np.cos(alpha_d), np.cos(omega)])
    B = np.array([d_e, d_n, d_u])
    return np.dot(A, B)

def compute_arc_deformation_mm(arc_phase_uw):
    c = 299_792_458
    freq = 5.404*10**9

    lam_meters = c / freq
    
    wavelength = lam_meters * 10**3 # mm


    arc_deformation = wavelength * arc_phase_uw / 4 / math.pi
    
    return arc_deformation

def get_vertical(lof, omega):
    """
    Computes the vertical deformation.

    Parameters:
    lof (float or np.array): The line-of-sight deformation.
    omega (float): The incidence angle (in radians).
    alpha_d (float): The angle between the north and the range (in radians).
    d_e (float): The deformation component for east deformation.
    d_n (float): The deformation component for north deformation.

    Returns:
    float: The vertical deformation.
    """
    # return (lof - (d_e* np.sin(omega) * np.sin(alpha_d)) - (d_n*np.sin(omega) * np.cos(alpha_d))) / np.cos(omega)

    return lof / np.cos(omega)
    
def fit_static_model(deformations, time):
    # time series observations
    #time= np.array(time)
    y = deformations # N or E
    m = len(y)
    # sampling interval
    #dt = diff[1] time series observations

   
    # design matrix based on y(t) = y0 + rt + e(t) 
    A = np.ones((m,4))
    A[:,1] = time
    A[:,2] = np.cos((2*np.pi)*time)
    A[:,3] = np.sin((2*np.pi)*time)
    m, n = A.shape

    # making cofactor matrices of white noise and flicker noise
    Qcf = np.zeros((m, m, 1))
    Qcf[:,:,0] = np.eye(m,m) 
    #Qcf[:,:,1] = pl_cofactor(2, time, dt)
    # initilize variance components
    sig0 = np.array([1])
    # apply LS-VCE to estimate two variance components and Qyy
    Sighat, Qs, Qyy, AllSighat = ls_vce(A, y, Qcf, sig0)
    Qyyi = linalg.inv(Qyy)
    # BLUE estimate of x
    xhat = linalg.inv(A.T @ Qyyi @ A) @ A.T @ Qyyi @ y
    # BLUE estimate of y
    yhat = A @ xhat;                         
    # BLUE estimate of e (residuals)
    ehat = y - yhat;
    return xhat ,yhat,ehat, Qyy