import numpy as np
import Black76
from  abc import ABCMeta,abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class BaseSABR(metaclass=ABCMeta):

    def __init__(self, f=0.01, shift=0,t=1,v_atm_n = 0.0010,beta=1,rho=0,volvol=0):
        self.f = f
        self.t = t
        self.shift = shift
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.v_atm_n = v_atm_n
        self.params = dict()

    @abstractmethod
    def alpha(self):
        "implied alpha parameter from the ATM normal vol"

    @abstractmethod
    def fit(self,k,v):
        "best fit the model to a discrete vol smile"

    @abstractmethod
    def lognormal_vol(self,k):
        "return the lognormal vol from a given strike"

    @abstractmethod
    def normal_vol(self,k):
        "returns the normal vol from a given strike"

    @abstractmethod
    def call(self,k,cp='Call'):
        "abstract mehod fro call prices"

    def density(self,k):
        std_dev = self.v_atm_n*np.sqrt(self.t)
        dk = 1e-4*std_dev
        d2call = self.call(k+dk)-2*self.call(k)+self.call(k-dk)
        return d2call/dk**2

    def get_params(self):

        return self.__dict__

    def __repr__(self):
        class_name = self.__class__.__name__
        return (class_name, _pprint(self.__dict__))

def _pprint(params):
    params_list=list()
    params_list = list()
    for i, (k, v) in enumerate(params):
        if type(v) is float:
            this_repr = '{}={:.4f}'.format(k, v)
        else:
            this_repr = '{}={}'.format(k, v)
        params_list.append(this_repr)
    return params_list

class BaseLogNormalSABR(BaseSABR):

    def normal_vol(self,k):
        f,s,t = self.f ,self.shift,self.t
        v_sln = self.lognormal_vol(k)

    def call(self,k,cp='call'):

        f,s,t = self.f , self.shift, self.t
        v_sln = self.lognormal_vol(k)
        pv = Black76.shifted_lognormal_call(k,f,s,t,v_sln,0,cp)
        return pv


def alpha(v_atm_ln,f,t,beta,rho,volvol):

    f_ = f**(beta-1)
    p = [t*f_**3*(1-beta)**2/24,
         t*f_**2*rho*beta*volvol/4,
         (1+t*volvol**2*(2-3*rho**2)/24*f_),
         -v_atm_ln]

    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots),np.real(roots))
    alpha_1st_guess = v_atm_ln*f**(1-beta)
    i_min = np.argmin(np.abs(roots_real-alpha_1st_guess))
    return roots_real[i_min]

class BaseNormalSABR(BaseSABR):

    def lognormal_vol(self,k):
        f,s,t = self.f,self.shift,self.t
        v_n = self.normal_vol(k)
        v_sln = Black76.normal_to_shifted_LogNormal(k,f,s,t,v_n)
        return v_sln

    def call(self,k,cp='Call'):

        f,t = self.f , self.t
        v_n = self.lognormal_vol(k)
        pv = Black76.normal_call(k,f,t,v_n,0,cp)
        return pv



def lognormal_vol(k,f,t,alpha,beta,rho,volvol):
    if k <= 0 or f <= 0:
        return 0.
    eps = 1e-07
    logfk = np.log(f / k)
    fkbeta = (f * k) ** (1 - beta)
    a = (1 - beta) ** 2 * alpha ** 2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / fkbeta ** 0.5
    c = (2 - 3 * rho ** 2) * volvol ** 2 / 24
    d = fkbeta ** 0.5
    v = (1 - beta) ** 2 * logfk ** 2 / 24
    w = (1 - beta) ** 4 * logfk ** 4 / 1920
    z = volvol * fkbeta ** 0.5 * logfk / alpha
    # if |z| > eps

    def _x(rho,z):
        a = (1 - 2*rho*z+z**2)**0.5 + z-rho
        b = 1-rho
        return np.log(a/b)

    if abs(z) > eps:
        vz = alpha*z*(1+(a+b+c)*t)/(d*(1+v+w)*_x(rho,z))
        return vz
    else:
        v0 = alpha*(1+(a+b+c)*t)/(d*(1+v+w))
        return v0


