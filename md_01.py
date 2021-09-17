import numpy as np
import time
from masses import label2mass
from scipy import constants
from writepdb import writePDB

# Posições iniciais das Partículas
def InitPos():
    p = np.random.rand(nMol, NDIM)
    p = box*p
    return p

# Velocidades iniciais das Partículas
def InitVel():
    v = np.zeros([nMol, NDIM])
    
    rangeV = 1.0             # ESCOLHER UM NÚMERO ALEATÓRIO ENTRE -rangeV e rangeV
    
    for i in range(NDIM):
        # v aleatórias para x, y e z
        v[:,i] = np.random.uniform(-rangeV,rangeV,size=nMol)
 
    v *= velMag             # ESCALAR PARA A TEMPERATURA DE INTERESSE
    
    vSum = v.sum(axis=0)    # REMOVER O CENTRO DE MASSA TRANSLACIONAL DA VELOCIDADE
    v += -vSum/nMol   
    return v

# Acelerações iniciais das Partículas
# A aceleração será estimada através cálculo das forças 
def InitAcel():
    a = np.zeros([nMol, NDIM])
    return a

def ComputeForcesFast():
    rriCut = 1.0/rCutSq
    rri3Cut = rriCut**3
    uCut = 4. * rri3Cut * (rri3Cut - 1.) + 1.

    f = np.zeros_like(pos)
    mol_a = np.zeros_like(pos)
    uSum0 = 0.
    uSum = 0.0
    virSum = 0.0#np.zeros([nMol, NDIM])
    for i in range(nMol-1):
        rij       = pos[i,:]-pos[i+1:,:]                        # Separation vectors for j>i
        rij       = np.remainder(rij + box/2., box) - box/2.
        rijSq     = np.sum(rij**2,axis=1)                   # Squared separations for j>1
        inRange   = rijSq < rCutSq                         # Set flags for within cutoff
        rri       = np.where ( inRange, 1.0/rijSq, 0.0 )  # (sigma/rij)**2, only if in range
        rri3      = rri ** 3
        fij       = 48. * rri3 * (rri3 - 0.5) * rri                               # LJ scalar part of forces
                
        fij       = rij * fij[:,np.newaxis]                 # LJ pair forces
        
        virSum  += np.sum(np.sum(rij * fij, axis=0))

        
        sumFij    = np.sum(fij,axis=0)
                
        mol_a[i,:]    = mol_a[i,:] + sumFij
        mol_a[i+1:,:] = mol_a[i+1:,:] - fij

        uVal = 4. * rri3 * (rri3 - 1.) + 1.
        uVal = np.where ( inRange, uVal, 0.0 )
        uSum += np.sum(uVal,axis=0)
        if i == 0: uSum0 = 1.0* uSum
    return mol_a, (uSum-uCut)/nMol, virSum, uSum0

# Calcular as forças do sistema através do potencial de LJ
def ComputeForces():
    # Condições periódicas de contorno (PBC)
    def ApplyBoundaryCondVec(dxyz):
        if dxyz >= 0.5*box:
            dxyz -= box
        elif dxyz < -0.5*box:
            dxyz += box
        return dxyz

    rriCut = 1.0/rCutSq
    rri3Cut = rriCut**3
    uCut = 4. * rri3Cut * (rri3Cut - 1.) + 1.
    
    ri3Cut = 1.0/(rCutSq*rCut)
    ri9Cut = ri3Cut**3
    virCut = 16./9.*np.pi*dens**2*(2.*ri9Cut - 3.*ri3Cut)
    
    uSum0 = 0.
    uSum = 0.
    virSum = 0.
    
    if not boolThermostat:
        mol_a = np.zeros([nMol,NDIM])
    for j1 in range(nMol-1):
        x1, y1, z1 = pos[j1,0], pos[j1,1], pos[j1,2]
        fora = []
        for j2 in range(j1+1,nMol):
            x2, y2, z2 = pos[j2,0], pos[j2,1], pos[j2,2]
            
            dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
            dx, dy, dz= ApplyBoundaryCondVec(dx), ApplyBoundaryCondVec(dy), ApplyBoundaryCondVec(dz)
            
            rr = dx*dx+dy*dy+dz*dz
            if rr<rCutSq:
            #if rr<box/2.:
                rri = 1.0/rr
                rri3 = rri**3
                fcVal = 48. * rri3 * (rri3 - 0.5) * rri
                uVal = 4. * rri3 * (rri3 - 1.) + 1.
                mol_a[j1] += fcVal*np.array([dx,dy,dz])
                mol_a[j2] += -fcVal*np.array([dx,dy,dz])
                uSum += uVal
                virSum += fcVal * rr
                if j1 == 0: uSum0 = 1.0* uSum
    return mol_a, (uSum - uCut)/nMol, virSum+virCut, uSum0


def AdjustInitTemp(vel):
    vFac = velMag/np.sqrt(2*kinEnergy)
    vel *= vFac
    return vel

def Thermostat(ace):
    s1 = 0.
    s2 = 0.
    
    vt   = vel + 0.5*dt*ace
    
    s1 = np.sum(vt*ace)
    s2 = np.sum(vt*vt)
    
    vFac = - s1 / s2
    
    vt   = vel + 0.5*dt*ace
    ace += vFac * vt

    return ace

def ApplyBoundaryCond(dxyz):
    dxyz = dxyz % box
    return dxyz

# Integração das equações no tempo
def Integration(part, vel, pos=None):
    if part == 1:
        vel += 0.5*dt*ace
        pos += dt*vel
        return vel, pos
    else:
        vel += 0.5*dt*ace
        return vel

# Médias das propriedades por blocos
# energia cinética, energia potencial, pressão...
def EvalProps(temp, kinEnergy, totEnergy, pressure):
    vSum = np.sum(vel,axis=0)
    vv = vel**2
    vvSum = np.sum(vv)
    kinEnergy = 0.5 * vvSum / nMol
    totEnergy = kinEnergy + potEnergy
    temp = 1 / (NDIM * nMol) * vvSum
    pressure = dens * (vvSum + virSum) / NDIM

    vvSum0 = np.sum(np.sum(vel[0]**2),axis=0)
    kinEnergy0 = 0.5 * vvSum
    
    return temp, kinEnergy, totEnergy, kinEnergy0, pressure

def BlockAvarage(data):
    dataSplit = np.array_split(data,nBlock)
    ave = [np.mean(d) for d in dataSplit]
    blockAve = np.mean(ave)
    stdErr   = np.std(ave)
    return blockAve, stdErr
 
# Print Resumo
def PrintSummary():
    #totEnergyMean = np.mean(totEnergyL)/eps
    #totEnergyStd = np.std(totEnergyL)/eps
    #potEnergyMean = np.mean(potEnergyL)/eps
    #potEnergyStd = np.std(potEnergyL)/eps
    #kinEnergyMean = np.mean(kinEnergyL)/eps
    #kinEnergyStd = np.std(kinEnergyL)/eps
    #tempMean = np.mean(tempL)*eps/BOLTZMANN
    #tempStd = np.std(tempL)*eps/BOLTZMANN
    
    totEnergyMean, totEnergyStd = BlockAvarage(np.array(totEnergyL)*eps)
    potEnergyMean, potEnergyStd = BlockAvarage(np.array(potEnergyL)*eps)
    kinEnergyMean, kinEnergyStd = BlockAvarage(np.array(kinEnergyL)*eps)
    tempMean, tempStd = BlockAvarage(np.array(tempL)*eps/BOLTZMANN)
    print("\n\n\n")
    print('%25s %5.2f %5s %5.2f' % ('<Total Energy (kJ/mol)>: ', totEnergyMean, '+/-', totEnergyStd))
    print('%25s %5.2f %5s %5.2f' % ('<Potential Energy (kJ/mol)>: ', potEnergyMean, '+/-', potEnergyStd))
    print('%25s %5.2f %5s %5.2f' % ('<kinetic Energy (kJ/mol)>: ', kinEnergyMean, '+/-', kinEnergyStd))
    print('%25s %5.2f %5s %5.2f' % ('<Temperature (K)>: ', tempMean, '+/-', tempStd))
    print("\n\n\n")
    return


t0 = time.time()

np.random.seed(0)                   

########## CONSTANTES #############
NDIM       = 3
AVOGADRO   = constants.value('Avogadro constant')                       # mol^-1 
BOLTZMANN  = constants.value('Boltzmann constant')                      # J K^-1 
BOLTZMANN  = BOLTZMANN*AVOGADRO/1000                                    # kJ mol^-1 K^1
AM2KG      = constants.value('atomic mass unit-kilogram relationship')  # kg
###################################
####### INFOS DOS SISTEMA #########
title = 'Argon MD'
sigma = 3.401              # AA (LJ)
eps   = 0.979              # kJ mol^-1 (LJ)
mass  = label2mass['Ar']   # a.m
###################################
####### PAR. DOS CÁLCULOS #########
stepLim = 10000             # Número total de passos
stepEq  = 2000
nMol    = 1024              # Número de átomos no sistema
dt_     = 0.10864530912012466            # Passo de integração (ps)
box_    = 4.142*(nMol*4)**(1./3.)        # 46.63616190345         # Tamanho da caixa (AA)
temp_   = 273.              # Temperatura do sistema (K)
dt      = dt_*np.sqrt(eps/(mass*sigma**2))
box     = box_/sigma
tempEx  = temp_*BOLTZMANN/eps
dens    = nMol/box**3
###################################
########### THERMOSTAT ############
boolThermostat = False
stepThermostat = stepEq + 8000
if stepThermostat > stepLim:
    print("stepThermostat > stepLim")
    exit()
elif stepThermostat < stepEq:
    print("stepThermostat < stepEq")
    exit()
tauT    = 200                                # Número de passos para escalar a temperatura (termostato)
###################################
######### Outras Variáveis ########
rCut      = 2.5 #2**(1./6.)           # Unidades reduzidas
rCutSq    = rCut*rCut
velMag = np.sqrt(NDIM*(1 - 1/nMol)*tempEx)  # VELOCITIES MAGNITUDE BASED ON TEMPERATURE. THE VELOCITIES DIRECTIONS ARE CHOSEN RANDOMLY.
totEnergy = 0.0
potEnergy = 0.0
kinEnergy = 0.0
virSum    = 0.0
pressure  = 0.0
temp      = 0.0
nBlock    = 10
totEnergyL= []
potEnergyL= []
kinEnergyL= []
tempL     = []
###################################
######### INICIALIZAÇÃO ###########
pos = InitPos()            # Posição inicial
vel = InitVel()            # Velocidade inicial
ace = InitAcel()           # Aceleração inicial
###################################

writePDB('input.pdb',pos*sigma, box*sigma, np.array(['Ar']*nMol), w='w')

temp, kinEnergy, totEnergy, kinEnergy0, pressure = EvalProps(temp, kinEnergy, totEnergy, pressure)

for step in range(1,stepLim):
    # printar o cabeçalho
    if step == 1:
        print('%5s %5s' % ("passo", "tempo (ps)"))
            
    vel, pos = Integration(1, vel, pos)
    pos = ApplyBoundaryCond(pos)
    ace, potEnergy, virSum, potEnergy0 = ComputeForcesFast()
    vel = Integration(2, vel)
    
    temp, kinEnergy, totEnergy, kinEnergy0, pressure = EvalProps(temp, kinEnergy, totEnergy, pressure)
                
    if step < stepEq:
        vel = AdjustInitTemp(vel)
        #continue
    
    if boolThermostat and step % tauT  == 0 and step < stepThermostat:
        ace = Thermostat(ace)
    
    if step == stepEq:
        boolThermostat= True
     
    totEnergyL.append(totEnergy)
    potEnergyL.append(potEnergy)
    kinEnergyL.append(kinEnergy)
    tempL.append(temp)
    
    print('%5d %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' % (step, step*dt_, temp*eps/BOLTZMANN, kinEnergy*eps, potEnergy*eps, totEnergy*eps, pressure*eps*1660/sigma**3, dens*mass*AM2KG/(sigma*1e-10)**3))
    
    
    writePDB('input.pdb',pos*sigma, box*sigma, np.array(['Ar']*nMol), w='a')


np.savetxt("vel.txt", vel)

PrintSummary()

tF = time.time()
print("CPU time = ", tF - t0, 'seconds')


















