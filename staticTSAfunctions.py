import pandas as pd
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import signal
from scipy.stats.distributions import chi2

def log_likelihood(Qyy, ehat):
    """
    This function computes the Log-Likelihood value for a given covariance matrix and residual vector.
    INPUT:
        Qyy: mxm covariance matrix of observations (time series)
        ehat: least squares estimate of residulas
    OUTPUT:
        log_like: Log-Likelihood value
    """
    m = Qyy.shape[0]
    # compute eigenvalues of Qyy
    lambda_vals = linalg.eigvals(Qyy)
    # compute sum-of-log of eigenvalues
    log_det = np.sum(np.log(lambda_vals))
    # compute Qyy_inv * ehat
    inv_Qyy_ehat = linalg.solve(Qyy, ehat)
    # compute log-likelihood
    log_like = -m/2*np.log(2*np.pi) - 1/2*log_det - 1/2*ehat.T @ inv_Qyy_ehat
    return log_like

def ls_vce(A, y, Qcf, sig0):
    """ 
    This function implements the LS-VCE method presented in:
    Teunissen, P.J.G., A.R. Amiri-Simkooei (2008). Least-squares variance component estimation, Journal of Geodesy, 82 (2): 65-82

    Input:
        A: the initial design matrix
        y: the vector of observations
        Qcf: mxm cofactor matrices Qcf[:,:,i], i=1,...,p. For example Q1 = Qcf[:,:,1].
        sig0: initial values of variance components, usually sig0=[1,...,1]
    Output:
        Sighat: final estimates of variance components
        Qs: covariance matrix of Sighat
        Qyy: final covariance matrix of observation vector y
        AllSighat: All variance components computed in different iterations, finally converged to Sighat 
    """
    # number of variance components
    p = np.size(Qcf,2)
    # size of design mtrix, m: number of rows (observations), and n: number of columns (unknowns)
    m, n = A.shape
    # initialze Sighat as sig0
    Sighat = sig0
    sig0 = Sighat+1
    # iteration counter is set to 0
    IT = 0
    # initialize an empty array AllSighat (all estimated variances in different iterations)
    AllSighat = np.array([]) 
    # a loop, which checks if the conversion of variances has happened
    while np.max(np.abs(Sighat-sig0)) > 1e-10 and IT < 10:  # change the threshold to see the effect (max 10 iterations)
        sig0 = Sighat
        # initialize Qyy
        Qyy = np.zeros((m, m))
        # given sig0, update Qyy
        for i in range(p):
             Qyy += sig0[i]*Qcf[:, :, i]
        # invert Qyy
        Qyinv = linalg.inv(Qyy)
        # compute the projector Pao
        Pao = np.eye(m) - A.dot(linalg.inv(A.T.dot(Qyinv).dot(A))).dot(A.T).dot(Qyinv)
        # compute the product of Qyinv and Pao
        QyinvP = Qyinv.dot(Pao)
        # compute the least squares residuals
        ehat = Pao.dot(y)
        Help = Qyinv.dot(ehat)
        # initialize l amd N
        l = np.zeros((p))
        N = np.zeros((p, p))
        # compute l (px1) and N (pxp)
        for i in range(p):
            l[i] = 0.5*(Help.T.dot(Qcf[:, :, i]).dot(Help))
            for j in range(i, p):
                N[i, j] = 0.5*np.trace(QyinvP.dot(Qcf[:, :, i]).dot(QyinvP).dot(Qcf[:, :, j]))
                N[j, i] = N[i, j]
        # we can estimate Sighat = inv(N)*l, but this has the risk to give negative variances
        # Sighat = linalg.inv(N).dot(l)
        # Qs = linalg.inv(N)
        # alternative is to use non-negative least squares (NNLS) to guarantee positive variances
        Sighat, Qs = nnls_v(N, l)  
        AllSighat = np.concatenate((AllSighat, Sighat)) # concatenate the subvector with AllSighat
        IT += 1
        print('Iteration %d, Estimated variances: %s' % (IT, Sighat)) 
    AllSighat = np.reshape(AllSighat, (p, IT), order='F') # reshape AllSighat into a pxIT matrix (column-wise)
    
    return Sighat, Qs, Qyy, AllSighat

def nnls_v(N, L):
    """ 
    This function implements the Non-Negative LS-VCE (NNLS-VCE) method (not part of LO, we just use it).
    It solves the non-negative least-squares problem: min = 0.5*s^T*N*s + L^t*s subject to s>=0
    Without the non-negativity constraints the least-squares solution is s = inv(N)*L.
    But nnls_v poses the constraints s>=0 to obtain non-negative unknowns s. For further information refer to:
    Non-negative least-squares variance component estimation with application to GPS time series, AR Amiri-Simkooei, Journal of Geodesy 90, 451-466
    Input:
        N: the normal pxp matrix
        L: the px1 vector

    Output:
        Sighat: final estimates of variance components Sighat>=0
        Qs: covariance matrix of Sighat
    """
    p = len(L)
    mu0 = -L.copy()
    s0 = np.zeros((p))
    stest = s0 + 1
    s = s0.copy()
    while linalg.norm(s-stest) > 1e-10:
        for k in range(p):
            s[k] = max(0, s0[k] - mu0[k] / N[k, k])
            mu0 = mu0 + (s[k] - s0[k]) * N[:, k]
        stest = s0.copy()
        s0 = s.copy()
    index = np.where(s == 0)[0]
    Ct = np.zeros((len(index), p))
    k1, k2 = Ct.shape
    for i in range(len(index)):
        Ct[i, index[i]] = 1
    C = Ct.T
    Ni = linalg.inv(N)
    if k1 > 0:
        Pco = np.eye(p) - C @ linalg.inv(Ct @ Ni @ C) @ Ct @ Ni
    else:
        Pco = np.eye(p)
    Qs = Ni @ Pco
    Sighat = s.copy()
    return Sighat, Qs

def pl_cofactor(kappa, time, dt):
    """
    This function computes the power-law cofactor matrix for a given spectral index kappa.
    INPUT:
        kappa: spectral index in the renge of [0-2]
        time: time instances of the time series
        dt: sampling interval
    OUTPUT:
        Q: cofactor matrix Q_pl(kappa)  
    """
    # change time to integer numbers
    time = np.round(time/dt).astype(int)
    # shift it to have t(0) = 1
    t = time - time[0] + 1

    m = t[-1]
  
    H = np.zeros((m,1))
    print(f'H {H.shape}')
    H[0] = 1
    for i in range(1, m):
        # make H[i]
        H[i] = H[i-1] * (kappa/2 + i - 1)/i
    # make a toeplitz matrix and then consider only its upper triangular
    U = np.triu(linalg.toeplitz(H))
   
    # find possible gaps in data (if data is not equally-spaced)
    # index refers to indices where we have data
    index = np.isin(np.arange(1, m+1), t)
  
    # make the final U, removing gaps
    U = U[:, index]
   
    Q = np.power(dt, kappa/2) * U.T @ U
  
    return Q


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