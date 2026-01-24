"""
Reference implementation accompanying:

Ferrer-Vidal España-Heredia, L.E., "REVISITING THE POLYTROPIC EFFICIENCY AND ITS PRACTICAL USE," Proceedings of the ASME Turbo Expo 2026,
Paper No. GT2026-177067.

This script demonstrates several commonly used implementations of
polytropic efficiency for compressor performance calculations, including
perfect-gas, thermally perfect, and real-gas formulations.

The code is intended for instructional and research purposes only.
It prioritizes clarity and traceability over computational efficiency
or general-purpose robustness.

Users are encouraged to consult the associated paper for theoretical
background, assumptions, and appropriate interpretation of results.

This code requires the CoolProp library for real-gas property calculations.
See https://coolprop.org/ for installation and documentation.   

This code also requires the thermoWF_GT2026 module.
"""

import thermoWF_GT2026 as th
from scipy.optimize import fsolve, newton
import numpy as np
from CoolProp.CoolProp import PropsSI as CP


# -------------------
# Constants
# -------------------
N_STEPS = 400                # number of integration steps for GI methods
T_AMBIENT = 288.15           # initial ambient temperature (K)
P_AMBIENT = 101325.0         # initial ambient pressure (Pa)
FAR=0.0                      # fuel-air ratio.  Required by flowStation object.
EFF_RANGE = np.linspace(0.8, 1.0, 20)  # efficiency range
PR_RANGE = np.linspace(3.0, 30, 20)    # pressure ratio range
REF_ENTH=th.h(1.0, 0.0)            # reference enthalpy (J/kg)
DP_STEP=2000                        # Constant dp step in (Pa) for use in methods 5 & 6.

# -------------------
# FlowStation setup
# -------------------
fs1 = th.flowStation()
fs2 = th.flowStation()
fs1.setTtPt(T_AMBIENT, P_AMBIENT)


# -------------------
# Process function
# -------------------
def procPolyEff(fs, init, PR, eff, method, TOL=1e-6):
    """
    Calculate total temperature at target PR for various polytropic models.
    
    Inputs:
        fs: A blank flow-station object to be used as the outlet station. flowStation class defined in the cythermWF module.
        init: The initial flow-station, also a flowStation object from cythermWF.
        PR: Pressure ratio.
        eff: Polytropic efficiency.
        method: What method to use, from "Methods" below.
        TOL: Tolerance for iterative methods.
    
    Methods:
        0: Perfect gas.
        1: Isentropic pressure ratio (TPIP)
        2: Iterative average gamma (TPPT)
        3: Thermally perfect Gibbs integral (TPGI), with number of steps (N_STEP) specified.
        4: Real gas Gibbs integral (RGGI), with number of steps (N_STEP) specified.
        5: Thermally perfect Gibbs integral (TPGI), with pressure step (DP_STEP) specified.
        6: Real gas Gibbs integral (RGGI), with pressure step (DP_STEP) specified.
        7: Thermally perfect entropic efficiency (TPEE)
        8: Iterative average gamma (TPAG)
        9: Real gas entropic efficiency (RGEE)
    """
    PR = max(0, PR)
    fs.copy(init)
    Pt = init.Pt * PR

    if method == 0:
        # Perfect gas.
        if PR > 1:
            TR=PR**((init.gamma-1)/(init.gamma*eff))
        else:
            TR=PR**((init.gamma-1)*eff/(init.gamma))
        
        fs.setTtPt(init.Tt*TR, Pt)

    if method == 1:
        # Thermally perfect isenropic pressure ratio (TPIP).
        if PR > 1:
            PRs =(PR) ** (1.0 / eff)
        else:
            PRs = (PR) ** (eff)
        
        def obj(Tguess):
            return th.intCpQT(init.Tt,Tguess,FAR)-th.R(FAR)*np.log(PRs)
        Tt = newton(obj, init.Tt)
        h=th.h(Tt,FAR)-REF_ENTH
        fs.setHtPt(h, Pt)

    elif method == 2:
        # Thermally perfect process temperature (TPPT).
        TtAve = init.Tt
        TtAveOld = init.Tt + 10.0
        while abs(TtAve - TtAveOld) > TOL:
            gamma = th.gamma(TtAve)
            TtAveOld = TtAve
            if PR > 1:
                TR = PR ** ((gamma - 1) / (gamma * eff))
            else:
                TR = PR ** ((gamma - 1) * eff / (gamma))
            Tt = init.Tt * TR
            TtAve = (init.Tt + Tt) / 2
        fs.setTtPt(Tt, Pt)

    elif method == 3 or method == 5:
    
        if method == 3:
            #  Thermally perfect Gibbs integral (TPGI), step number specified
            dp = (Pt - init.Pt) / N_STEPS
        
        if method == 5:
            #  Thermally perfect Gibbs integral (TPGI), pressure step specified
            dp = DP_STEP
            
        PtNew = init.Pt
        TtNew = init.Tt
        htNew = init.ht
        dT=0

        def obj(TtGuess, PtNew, TtOld, htOld):
            htNew = th.h(TtGuess, FAR)-REF_ENTH
            Tave = 0.5 * (TtGuess + TtOld)
            Pave = PtNew - dp / 2
            vdp = fs.R * (Tave) / (Pave) * dp
            dH = (htNew - htOld) * eff
            return dH - vdp
        
        def deriv(T, PtNew, TtOld, htOld):
            # return eff*th.cp(T,FAR)-fs.R*dp/(2*(PtNew-dp/2))
            return eff*fs.cp-fs.R*dp/(2*(PtNew-dp/2))

        while PtNew < Pt:
            TtOld = TtNew
            htOld = htNew
            PtNew += dp


            TtGuess=TtOld+dT/eff
            # TtNew = float(fsolve(obj, TtGuess, args=(PtNew, TtOld, htOld))[0])
            TtNew = float(newton(obj, TtGuess, deriv, args=(PtNew, TtOld, htOld), tol=1e-4))
            dT=TtNew-TtOld
            htNew = th.h(TtNew, FAR)-REF_ENTH

        fs.setTtPt(TtNew, PtNew)

    elif method == 4 or method == 6:
        
        if method == 4:
            #  Thermally perfect Gibbs integral (TPGI), step number specified
            dp = (Pt - init.Pt) / N_STEPS
        if method == 6:
            #  Real gas Gibbs integral (RGGI), pressure step specified
            dp = DP_STEP
            
        PtNew = init.Pt
        TtNew = init.Tt
        htNew = CP('H', 'T', TtNew, 'P', PtNew, 'Air')
        htInit=htNew

        def obj(TtGuess, PtStep, TtOld, htOld):
            htGuess = CP('H', 'T', TtGuess, 'P', PtStep, 'Air')
            Tmid = 0.5 * (TtGuess + TtOld)
            Pmid = PtStep - 0.5 * dp
            rho = CP('D', 'T', Tmid, 'P', Pmid, 'Air')
            vdp = (1.0 / rho) * dp
            dH = (htGuess - htOld) * eff
            return dH - vdp

        while PtNew < Pt:
            TtOld = TtNew
            htOld = htNew
            PtNew += dp
            
            TtGuess=TtOld+dT
            #TtNew = float(fsolve(obj, TtGuess, args=(PtNew, TtOld, htOld))[0])
            TtNew = float(newton(obj, TtGuess, args=(PtNew, TtOld, htOld), tol=1e-4))
            dT=TtNew-TtOld
       
            htNew = CP('H', 'T', TtNew, 'P', PtNew, 'Air')

        fs.setTtPt(TtNew, PtNew)

    elif method==7:
        # Thermally perfect entropic efficiency (TPEE).
        def obj(TtGuess):
            S1=init.s
            S2p=S1+th.dS(init.Tt,init.Pt,TtGuess,init.Pt,FAR=0.0)
            S2=S1+th.dS(init.Tt,init.Pt,TtGuess,Pt,FAR=0.0)
            return eff*(S2p-S1)-(S2p-S2)
        
        # Perfect gas method for initial guess
        if PR > 1:
            TR=PR**((init.gamma-1)/(init.gamma*eff))
        else:
            TR=PR**((init.gamma-1)*eff/(init.gamma))

        TtGuess = init.Tt * TR
        Tt = newton(obj, TtGuess)
        fs.setTtPt(Tt, Pt)

    elif method == 8:
        # Thermally perfect average gamma (TPAG)
        Tt2Old = init.Tt-2*TOL
        Tt2 = init.Tt
        gamma1 = th.gamma(init.Tt)
        gamma2=gamma1
        while abs(Tt2Old - Tt2) > TOL:
            gammaAve = 0.5 * (gamma1 + gamma2)
            if PR > 1:
                TR = PR ** ((gammaAve - 1) / (gammaAve * eff))
            else:
                TR = PR ** ((gammaAve - 1) * eff / (gammaAve))
            Tt = init.Tt * TR
            Tt2Old = Tt2
            Tt2=Tt
            gamma2=th.gamma(Tt)
        fs.setTtPt(Tt, Pt)
    
    elif method==9:
        # Real gas entropic efficiency (RGEE)

        S1=CP('S', 'T', init.Tt, 'P', init.Pt, 'Air')
        def obj(TtGuess):
            S2p=CP('S', 'T', TtGuess, 'P', init.Pt, 'Air')
            S2=CP('S', 'T', TtGuess, 'P', Pt, 'Air')
            return eff*(S2p-S1)-(S2p-S2)

        
        # Perfect gas method for initial guess
        if PR > 1:
            TR=PR**((init.gamma-1)/(init.gamma*eff))
        else:
            TR=PR**((init.gamma-1)*eff/(init.gamma))

        TtGuess = init.Tt * TR
        Tt = newton(obj, TtGuess)
        fs.setTtPt(Tt, Pt)




# Usage example, with inputs as above:
fs2.setTtPt(T_AMBIENT, P_AMBIENT) # Set flowStation to ambient conditions.
PR=15.0          # Example pressure ratio.
polyEff=0.9     # Example polytropic efficiency.
procPolyEff(fs2, fs1, PR, polyEff, 2) # Call propPolyEff with Method 2.
print("Tt:", fs2.Tt) # Print outlet temperature.