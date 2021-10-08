#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import curve_fit

from pathlib import Path

def Henderson_Hasselbalch(pH, pKa):
    return 1 / (1 + 10**(1.0*(pH - pKa)))

plt.rcParams.update({'font.size': 22})

l = Path('.').rglob('*/*.cphlog')
#l = Path('.').rglob('example/*/*.cphlog')
l = sorted(l)

pHs = []
lambdaValues = []
for i in l:
    pH = float(str(i).split('/')[0])
    pHs.append(pH)
    lambdaValues.append(np.loadtxt(i, usecols=(9,10,11)))

pHs, idx = np.unique(pHs, return_inverse=True)        
nPhs = len(np.unique(idx))

lambdaValues = np.array(lambdaValues)
protFrac = []

plt.figure(figsize=(14, 9))

print('#  pH   frac')
for i in range(nPhs):
    indices = np.where(idx == i)
    lambda_ = np.concatenate(lambdaValues[indices])
    chi = lambda_[:, 0]*(1 - lambda_[:, 1]) + (1 - lambda_[:, 0])*lambda_[:, 1]
    prot_frac = chi.mean()
    protFrac.append(prot_frac)
    prot_frac_err = np.sqrt(prot_frac*(1 - prot_frac) / chi.size)
    print('%5.2f %6.4f %6.4f'%(pHs[i], prot_frac, prot_frac_err))
    
    plt.errorbar(pHs[i], prot_frac, yerr=prot_frac_err, marker='o', markersize=6, color='k', capsize=5, capthick=2)

fitPH = np.linspace(2.8, 6.0, 100)

popt, pcov = curve_fit(Henderson_Hasselbalch, pHs, protFrac, bounds=(0,7))

plt.plot(fitPH, Henderson_Hasselbalch(fitPH, *popt), 'r-')

plt.text(5.25, 0.8, r'f(pH) = $\frac{1}{(1+10^{pH - pka})}$'+'\n'
         +r'pKa = '+str(round(popt[0],5))+'$\pm$'+str(round(pcov[0][0],5)), 
         fontsize=15,  color='red', style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
#plt.text(2.5, 0.15, r'pKa = '+str(round(popt[0],5))+'$\pm$'+str(round(pcov[0][0],5)), fontsize=10,  color='red')

plt.ylabel('protonated fraction, f')
plt.xlabel('pH')
plt.ylim([0,1])
plt.grid(linestyle='--', linewidth='0.5')
plt.show()
