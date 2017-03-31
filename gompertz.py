from GPy.core import Mapping, Param
from paramz.transformations import Logexp
import paramz.transformations
import numpy as np

class Gompertz(Mapping):

    def __init__(self, mu, lam, A,):
        super(Gompertz, self).__init__(1,1, name='gompertz')

        self.mu = Param('mu', mu, Logexp())
        self.lam = Param('lam', lam, Logexp())
        self.A = Param('A', A, Logexp())

        #self.mu = Param('mu', mu)
        #self.lam = Param('lam', lam)
        #self.A = Param('A', A)

        self.link_parameters(self.mu, self.lam, self.A)

    def f1(self, X):
        return self.mu*np.e/self.A*(self.lam-X) + 1

    def f2(self, X):
        return -np.exp(self.f1(X))

    def f(self, X):

        return self.A * np.exp(self.f2(X))

        #return self.A * np.exp(-np.exp(self.mu*np.e/self.A*(self.lam-X) + 1))

    def gradients_X(self, dL_dF, X):
        print 'gradients_X'
        return self.f(X)*self.f2(X)*(-self.mu*np.e/self.A)

    def update_gradients(self, dL_dF, X):

        #plt.plot(dL_dF)

        #print dL_dF * (self.f(X) * self.f2(X) * np.e/self.A*(self.lam-X))
        #print np.sum(dL_dF * (self.f(X) * self.f2(X) * np.e/self.A*(self.lam-X)))

        self.mu.gradient = np.sum(dL_dF * (self.f(X) * self.f2(X) * np.e/self.A*(self.lam-X)))
        self.lam.gradient = np.sum(dL_dF * (self.f(X) * self.f2(X) * self.mu*np.e/self.A))
        self.A.gradient = np.sum(dL_dF * (self.f(X)/self.A  + self.f(X)*self.f2(X)*(-self.mu*np.e/self.A/self.A*(self.lam-X))))

        #raise NotImplementedError
