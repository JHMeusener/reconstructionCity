import torch
import numpy as np
def linetracer(startvoxel, endvoxel):
    '''input: numpy array of 1,3 array'''
    diffFastestIdx = np.argmax(abs(endvoxel - startvoxel))
    start = startvoxel[0,diffFastestIdx]
    end = endvoxel[0,diffFastestIdx]
    forward = True
    nr1up = True
    nr2up = True
    if start > end:
        forward = False
    nr1Direction = endvoxel[0,diffFastestIdx-1]-startvoxel[0,diffFastestIdx-1]
    if nr1Direction < 0:
        nr1up = False
        nr1Direction = -nr1Direction
    nr2Direction = endvoxel[0,diffFastestIdx-2]-startvoxel[0,diffFastestIdx-2]
    if nr2Direction < 0:
        nr2up = False
        nr2Direction = -nr2Direction
    fastError = abs(end-start)
    nr1Error = abs(end-start)//2
    nr2Error = abs(end-start)//2
    currentPosition = startvoxel
    for i in range(abs(end-start)):
        nr1Error -= nr1Direction
        nr2Error -= nr2Direction
        if nr1Error < 0:
            #one step in nr1 direction
            nr1Error += fastError
            currentPosition[0,diffFastestIdx-1] += 1 if nr1up else -1
        if nr2Error < 0:
            #one step in nr2 direction
            nr2Error += fastError
            currentPosition[0,diffFastestIdx-2] += 1 if nr2up else -1
        currentPosition[0,diffFastestIdx] += 1 if forward else -1
        yield currentPosition

if __name__ == '__main__':
    startvoxel = np.array([[10,10,10]])
    endvoxel = np.array([[18,13,15]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
    endvoxel = np.array([[10,10,10]])
    startvoxel = np.array([[18,13,15]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
    startvoxel = np.array([[10,10,10]])
    endvoxel = np.array([[13,18,15]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
    endvoxel = np.array([[10,10,10]])
    startvoxel = np.array([[13,18,15]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
    startvoxel = np.array([[10,10,10]])
    endvoxel = np.array([[13,18,19]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
    endvoxel = np.array([[10,10,10]])
    startvoxel = np.array([[13,18,19]])
    print(startvoxel,endvoxel)
    for i in linetracer(startvoxel, endvoxel):
        print(i)
        
