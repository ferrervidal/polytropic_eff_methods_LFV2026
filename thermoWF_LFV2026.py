
"""
Minimal thermodynamic property module supporting polytropic efficiency
calculations for gas turbine performance modeling.

This module provides a limited set of thermodynamic property functions
and a minimal thermodynamic state container intended solely to support
the illustrative polytropic efficiency formulations presented in the
accompanying paper.

The implementation prioritizes clarity, transparency, and consistency
with standard gas turbine performance modeling practice. It is not
intended to be a general-purpose thermodynamic library or a high-fidelity
property package.

Thermodynamic properties are based on simplified Walsh and Fletcher–type
polynomial fits and are suitable for instructional use and comparative
method studies.

Ref: 
    
    Walsh, P.P, Fletcher, P. (2004), "Gas Turbine Performance",
    Second Edition, Wiley (Blackwell Science). 
    Chapter 3, Pg 116-117

Author: Ferrer-Vidal, 2026
"""

import numpy as np
import scipy as sp

"""
Polynomial coefficients.
"""
# Air
A=np.array([
    0.992313, 
    0.236688,
    -1.852148,
    6.083152,
    -8.893933,
    7.097112,
    -3.234725,
    0.794571,
    -0.081873,
    0.422178,
    0.001053
])
# Combustion Products
B=np.array([
    -0.718874,
    8.747481,
    -15.863157,
    17.254096,
    -10.233795,
    3.081778,
    -0.361112,
    -0.003919,
    0.055593,
    -0.0016079
])

"""
Thermodynamic property functions.
"""
def cp(T, FAR): 
    """
    Specific heat capacity in [J/(kg-K)].
    """
    Tz=T/1000
    air_cp=A[0]+A[1]*Tz+A[2]*Tz**2+A[3]*Tz**3+A[4]*Tz**4+A[5]*Tz**5+A[6]*Tz**6+A[7]*Tz**7+A[8]*Tz**8 # cp of pure air.
    comb_cp= FAR/(1+FAR)*(B[0]+B[1]*Tz+B[2]*Tz**2+B[3]*Tz**3+B[4]*Tz**4+B[5]*Tz**5+B[6]*Tz**6+B[7]*Tz**7) # contribution of combustion products.
    return (air_cp+comb_cp)*1E3

def h(T, FAR):
    """
    Specific enthalpy in [J/kg].
    """
    Tz=T/1000
    air_h=A[0]*Tz+(A[1]*Tz**2)/2+(A[2]*Tz**3)/3+(A[3]*Tz**4)/4+(A[4]*Tz**5)/5+(A[5]*Tz**6)/6+(A[6]*Tz**7)/7+(A[7]*Tz**8)/8+(A[8]*Tz**9)/9+A[9] # enthalpy of pure air.
    comb_h=FAR/(1+FAR)*(B[0]*Tz+(B[1]*Tz**2)/2+(B[2]*Tz**3)/3+(B[3]*Tz**4)/4+(B[4]*Tz**5)/5+(B[5]*Tz**6)/6+(B[6]*Tz**7)/7+(B[7]*Tz**8)/8+B[8]) # contribution of combustion products.
    return (air_h+comb_h)*1E6


def intCpQT(T1, T2, FAR): 
    """
    F2-F1 in [J/(kg-K)]. This is int(cp/T)*dT between T1 and T2.
    """
    Tz1=max(T1/1000,1e-4)
    Tz2=max(T2/1000,1e-4)

    air_F2=A[0]*np.log(Tz2)+A[1]*Tz2+A[2]/2*Tz2**2+A[3]/3*Tz2**3+A[4]/4*Tz2**4+A[5]/5*Tz2**5+A[6]/6*Tz2**6+A[7]/7*Tz2**7+A[8]/8*Tz2**8+A[10] #  pure air.
    comb_F2= FAR/(1+FAR)*(B[0]*np.log(Tz2)+B[1]*Tz2+B[2]/2*Tz2**2+B[3]/3*Tz2**3+B[4]/4*Tz2**4+B[5]/5*Tz2**5+B[6]/6*Tz2*6+B[7]/7*Tz2**7+B[9]) # contribution of combustion products.
    
    air_F1=A[0]*np.log(Tz1)+A[1]*Tz1+A[2]/2*Tz1**2+A[3]/3*Tz1**3+A[4]/4*Tz1**4+A[5]/5*Tz1**5+A[6]/6*Tz1**6+A[7]/7*Tz1**7+A[8]/8*Tz1**8+A[10] #  pure air.
    comb_F1= FAR/(1+FAR)*(B[0]*np.log(Tz1)+B[1]*Tz1+B[2]/2*Tz1**2+B[3]/3*Tz1**3+B[4]/4*Tz1**4+B[5]/5*Tz1**5+B[6]/6*Tz1*6+B[7]/7*Tz1**7+B[9]) # contribution of combustion products.
    
    return (air_F2+comb_F2-(air_F1+comb_F1))*1E3

def R(FAR=0.0):
    """
    Specific gas constant in [J/(kg-K)].
    """
    return (287.05-0.00990*FAR+1E-07*FAR**2)

def dS(T1,P1,T2,P2,FAR=0.0):
    """
    Specific entropy in [J/(kg-K)].
    """
    P2=max(P2,1)
    P1=max(P1,1)
    return (intCpQT(T1,T2,FAR)-R(FAR)*np.log(P2/P1))

def gamma(T, FAR=0):
    """
    Specific heat capacity ratio.
    """
    cpp=cp(T, FAR)
    RR=R(FAR)
    return cpp/(cpp-RR)




class flowStation:
    """
    Minimal thermodynamic state container.

    This class stores total temperature, total pressure, and fuel–air ratio
    and provides access to derived thermodynamic properties using the
    associated property functions.

    It is intentionally limited in scope and is not a flow or component model.

    This class is intended solely to support the illustrative polytropic
    efficiency methods presented in the accompanying ASME Turbo Expo paper.
    """

    def __init__(self, Tt=None, Pt=None, FAR=0.0):
        self.Tt = Tt      # Total temperature [K]
        self.Pt = Pt      # Total pressure [Pa]
        self.FAR = FAR    # Fuel–air ratio [-]

        self.ht = None    # Total enthalpy [J/kg]
        self.st = None    # Total entropy [J/(kg-K)]

        if Tt is not None and Pt is not None:
            self.update_state()

    def update_state(self):
        """
        Update thermodynamic properties from current Tt, Pt, and FAR.
        """
        self.ht = h(self.Tt, self.FAR)
        # Entropy is relative; store absolute form for consistency
        self.st = intCpQT(1.0, self.Tt, self.FAR) - R(self.FAR)*np.log(self.Pt)

    def setTtPt(self, Tt, Pt):
        """
        Set total temperature and pressure and update state.
        """
        self.Tt = Tt
        self.Pt = Pt
        self.update_state()

    def copy(self, to_copy):
        """
        Return a shallow copy of the flow station.
        """
        self.Tt = to_copy.Tt
        self.Pt = to_copy.Pt
        self.FAR = to_copy.FAR
        self.ht = to_copy.ht
        self.st = to_copy.st

    def dS_to(self, other):
        """
        Entropy change from this state to another flowStation.
        """
        return dS(self.Tt, self.Pt, other.Tt, other.Pt, self.FAR)
