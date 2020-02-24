import numpy as np
import matplotlib.pyplot as plt

def tau(M):
#   autocorr is the UNintegrated autocorrelation curve
    autocorr = auto_corr_fast(M)
#   tau = 1 + 2*sum(G)
    return 1 + 2*np.sum(autocorr), autocorr


def auto_corr(M, kappa = 1000):
    auto_corr = np.zeros(kappa-1)
    mu = np.mean(M)
    for s in range(1,kappa-1):
        auto_corr[s] = np.mean( (M[:-s]-mu) * (M[s:]-mu) ) / np.var(M)
    return auto_corr


def auto_corr_fast(M, kappa = 1000):
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)

    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G

def compare_IAC(Ms,labels):
    for ind,M in enumerate(Ms):
        IAC,G = tau(M)
        plt.plot(np.arange(len(G)),G,label="{}: IAC = {:.2f}".\
                                            format(labels[ind],IAC))
    plt.legend(loc='best',fontsize=14)
    plt.tight_layout()
    #plt.show()
    plt.savefig('project_plot3.png')
    plt.close()