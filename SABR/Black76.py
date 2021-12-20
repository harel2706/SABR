import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def lognormal_call(k,f,t,v,r,cp='call'):

    if k <= 0 or f <= 0 or t <= 0 or v <= 0:
          return 0.
    d1 = (np.log(f/k)+v**2*t/2)/(v*t**0.5)
    d2 = d1 - v*t**0.5
    if cp=='call':
         pv = np.exp(-r*t)*(f*norm.cdf(d1)-k*norm.cdf(d2))
    elif cp=='put':
           pv = np.exp(-r*t)*(-f*norm.cdf(-d1)+k*norm.cdf(-d2))
    else:
           pv = 0
    return pv

def shifted_lognormal_call(k,f,s,t,v,r,cp='call'):
    return lognormal_call(k+s,f+s,t,v,r,cp)

def normal_call(k,f,t,v,r,cp='call'):

    cp_sign ={'call':1,
              'put':-1}[cp]
    d1 = (f-k)/(v*t**0.5)
    pv = np.exp(-r*t) * (cp_sign * (f - k) * norm.cdf(cp_sign * d1) +
                          v * (t / (2 * np.pi))**0.5 * np.exp(-d1**2 / 2))
    return pv

def normal_to_shifted_LogNormal(k,f,s,t,v_n):
    #convert a normal vol for a given strike to a shifted log normal vol

    n=1e12 #rate of optimizer convergence
    eps = 1e-07 #numerical tolerance for K=F

    # if K=f, use simple first guess
    if abs(k-f)<=eps:
        v_sln_0 = v_n/(f+s)
    else:
        v_sln_0 = hagan_normal_to_lognormal(k,f,s,t,v_n)

    target_premium = n * normal_call(k,f,t,v_n,0)

    def premium_square_error(v_sln):
        premium = n*shifted_lognormal_call(k,f,s,t,v_sln,0)
        return (premium-target_premium)**2

    res = minimize(fun=premium_square_error,
                    x0 = v_sln_0,
                    jac=None,
                    options={'gtol': 1e-8,
                             'eps': 1e-9,
                              'maxiter': 10,
                            'disp': False},
                       method='CG')
    return res.x[0]

def hagan_normal_to_lognormal(k,f,s,t,v_n):
    "convert normal vol to lognormal vol using Hagan's 2002 paper"

    k = k+s
    f = f+s

    if abs(np.log(f/k))<= 1e-8:
        factor = k
    else:
        factor = (f-k)/np.log(f/k)
    p = [factor*(-1/24)*t,
         0,
         factor,
         -v_n]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots),np.real(roots))
    v_sln_0 = v_n/f
    i_min = np.argmin(np.abs(roots_real-v_sln_0))
    return roots_real[i_min]

def hagan_lognormal_to_normal(k,f,s,t,v_sln):
    "convert lognormal to normal vol"

    k = k+s
    f = f+s
    logFK = np.log(f/k)
    a = v_sln*np.sqrt(f*k)
    b = (1/24)*logFK**2
    c = (1/1920)*logFK**4
    d = (1/24)*(1-(1/120)*logFK**2)*v_sln**2*t
    e = (1/5760)*v_sln**4*t**2
    v_n = a * (1+b+c)/(1+d+e)
    return v_n

def shifted_LogNormal_to_normal(k,f,s,t,v_sln):
    "convert a normal vol for a given strike to a lognormal vol"

    n= 1e2
    target_premium = n*shifted_lognormal_call(k,f,s,t,v_sln,0)
    v_n_0 = hagan_lognormal_to_normal(k,f,s,t,v_sln)

    def premium_square_error(v_n):
        premium = n * normal_call(k,f,t,v_n,0)
        return (premium-target_premium)**2

    res = minimize(fun=premium_square_error,
                       x0=v_n_0,
                       jac=None,
                       options={'gtol': 1e-8,
                                'eps': 1e-9,
                                'maxiter': 10,
                                'disp': False},
                       method='CG')
    return res.x[0]

def LogNormal_to_LogNormal(k,f,s,t,v_u_sln,u):
    """Convert a LogNormal vol to a  shifted SLN vol."""
    n = 1e2

    #use simple first guess
    v_sln_0 = v_u_sln*(f+u)/(f+s)

    target_premium = n*shifted_lognormal_call(k,f,u,t,v_u_sln,0)

    def premium_square_error(v_sln):
        premium = n*shifted_lognormal_call(k,f,s,t,v_sln,0)
        return (premium-target_premium)**2

    res = minimize(
            fun=premium_square_error,
            x0=v_sln_0,
            jac=None,
            options={'gtol': 1e-8,
                     'eps': 1e-9,
                     'maxiter': 10,
                     'disp': False},
            method='CG'
        )
    return res.x[0]

def bachelier_call(f,k,v,t):
    v_sqrt_t = v *np.sqrt(t)
    f_minus_k = f -k
    return f_minus_k*norm.cdf(f_minus_k/v_sqrt_t)+v_sqrt_t*norm.pdf(f_minus_k/v_sqrt_t)

print(lognormal_call(k=1.5246,f=1.5246,t=1,v=42.88,r=1.5246))
print(bachelier_call(k=1.5246,f=1.5246,t=1,v=0.8341))