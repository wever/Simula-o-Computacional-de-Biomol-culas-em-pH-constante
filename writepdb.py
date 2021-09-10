import os

def writePDB(outName, coord, box, labels, w='w'):
    fmt = {
        'ATOM': (
            "ATOM  {serial:5d} {name:<4s}{altLoc:<1s}{resName:<4s}"
            "{chainID:1s}{resSeq:4d}{iCode:1s}"
            "   {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}{occupancy:6.2f}"
            "{tempFactor:6.2f}      {segID:<4s}{element:>2s}\n"),
        'REMARK': "REMARK     {0}\n",
        'HEADER': "HEADER    {0}\n",
        'END': "END\n",
        'CRYST1': ("CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}"
                   "{ang[0]:7.2f}{ang[1]:7.2f}{ang[2]:7.2f} "
                   "{spacegroup:<11s}{zvalue:4d}\n")
    }
    
    if os.path.isfile(outName) and w == 'w':
        boolExit = True
        for name in range(10):
            bakName = '#'+outName+'_'+str(name)+'#'
            if not os.path.isfile(bakName):
                boolExit = False
                break
    
        if not boolExit:
            os.rename(outName, bakName)
        else:
            print('Many backup files (#*)')
            exit()

    nAtoms = len(labels)
    with open(outName, w+'t') as outputPDB:
        # Header
        outputPDB.write(fmt['HEADER'].format('We Welcome...'))
        outputPDB.write(fmt['REMARK'].format('“My brain hurts!”'))
        outputPDB.write(fmt['CRYST1'].format(box=[box]*3,
                                             ang=[90.,90.,90.],
                                             spacegroup='',
                                             zvalue=1))
        for i, l in zip(range(nAtoms), labels):
            outputPDB.write(fmt['ATOM'].format(
                serial=i,
                name=l,
                altLoc=' ',
                resName=l,
                chainID='',
                resSeq=i,
                iCode=' ',
                pos=coord[i],
                occupancy=1,
                tempFactor=1,
                segID=' ',
                element=l
            ))
        outputPDB.write(fmt['END'])
