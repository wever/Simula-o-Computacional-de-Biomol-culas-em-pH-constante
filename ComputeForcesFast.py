def ComputeForcesFast():
    rriCut = 1.0/rCutSq
    rri3Cut = rriCut**3
    uCut = 4. * rri3Cut * (rri3Cut - 1.) + 1.

    f = np.zeros_like(pos)
    mol_a = np.zeros_like(pos)
    uSum = 0.0
    for i in range(nMol-1):
        rij       = pos[i,:]-pos[i+1:,:]                        # Separation vectors for j>i
        rij       = np.remainder(rij + box/2., box) - box/2.
        rijSq     = np.sum(rij**2,axis=1)                   # Squared separations for j>1
        inRange   = rijSq < rCutSq                         # Set flags for within cutoff
        rri       = np.where ( inRange, 1.0/rijSq, 0.0 )  # (sigma/rij)**2, only if in range
        rri3      = rri ** 3
        fij       = 48. * rri3 * (rri3 - 0.5) * rri                               # LJ scalar part of forces
        fij       = rij * fij[:,np.newaxis]                 # LJ pair forces
        
        mol_a[i,:]    = mol_a[i,:] + np.sum(fij,axis=0)
        mol_a[i+1:,:] = mol_a[i+1:,:] - fij

        uVal = 4. * rri3 * (rri3 - 1.) + 1.
        uVal = np.where ( inRange, uVal, 0.0 )
        uSum += np.sum(uVal,axis=0)
    return mol_a, (uSum-uCut)/nMol
