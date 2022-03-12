import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from linAlgHelper import asSpherical, asCartesian, p2NormLastDim, getPlaneNormalFrom3Points,\
                       projectVectorOnVector,  getPointDistances2PlaneNormal, getLineIntersection,\
                        linePlaneIntersection, getObjectBinIdx, createPlaneExample, getPerObjectIdx,\
                        maskEvenTensorOutput





device = torch.device(0)

#vertices (copied from blender)
dodecahedronVertices = torch.tensor([[-0.9504, -0.3090, -1.3098],
[-0.5871, 0.8091, -1.3098],
[-1.5371, -0.5003, -0.3092],
[-0.9504, 1.3079, -0.3092],
[1.5378, 0.4995, 0.3092],
[0.9517, -0.3090, -1.3098],
[0.0000, -1.6171, -0.3092],
[0.0007, -1.0000, -1.3098],
[0.5885, 0.8091, -1.3098],
[-0.0003, 1.6171, 0.3092],
[0.9504, 1.3075, -0.3092],
[-0.9507, 0.3090, 1.3098],
[-1.5378, 0.4990, 0.3092],
[0.0004, 1.0000, 1.3098],
[-0.5874, -0.8091, 1.3098],
[-0.9507, -1.3079, 0.3092],
[1.5375, -0.4995, -0.3092],
[0.9504, -1.3087, 0.3092],
[0.9515, 0.3090, 1.3098],
[0.5882, -0.8091, 1.3098]], device=device)

#vertices on a plane
planeVertices = torch.tensor([[1,0,2,12,3],
[10,8,1,3,9],
[8,10,4,16,5],
[16,17,6,7,5],
[15,2,0,7,6],
[0,1,8,5,7],
[13,18,4,10,9],
[12,11,13,9,3],
[11,12,2,15,14],
[15,6,17,19,14],
[16,4,18,19,17],
[18,13,11,14,19]], device=device)

#vertices (copied from blender)
dodecahedronVertices = torch.tensor([[1.,1.,-1.],
[1.,-1.,-1.],
[1.,1.,1.,],
[1.,-1.,1.],
[-1.,1.,1.],
[-1.,-1.,1.],
[-1.,-1.,-1.],
[-1.,1.,-1.]], device=device)

#vertices on a plane
planeVertices = torch.tensor([[0,1,2,3],
[0,2,7,4],
[0,1,7,6],
[7,4,5,6],
[2,3,5,4],
[1,3,5,6]], device=device)

#get planeNeighbours for each plane and Vertex-Neighbours for each Vertex
vertexPlaneNeighbours = []
VertexVertexNeighbours = []
for i in range(len(dodecahedronVertices)):
    vertexPlaneNeighbours.append([])
    VertexVertexNeighbours.append([])
    same = []
    for p in range(len(planeVertices)):
        if i in planeVertices[p]:
            vertexPlaneNeighbours[-1].append(p)
            for u in planeVertices[p]:
                same.append(u)
    for s in range(len(dodecahedronVertices)):
        if same.count(s) == 2:
            VertexVertexNeighbours[-1].append(s)
vertexPlaneNeighbours = torch.tensor(vertexPlaneNeighbours, device=device)
VertexVertexNeighbours = torch.tensor(VertexVertexNeighbours, device=device)

#create planenormals
startPlanes = getPlaneNormalFrom3Points(dodecahedronVertices[planeVertices[:,:3],:][None,:,:,:])
epsilon = 0.001




def step(startPlanes, objectPoints):
    #zero points
    maxima, maxIdx = objectPoints.max(dim=1)
    minima, minIdx = objectPoints.min(dim=1)
    offset = (maxima+minima)/2.
    zeroed_input = objectPoints-offset[:,None,:]
    objectsNr, planeNr, coordinates = objectPoints.shape
    startDistances = getPointDistances2PlaneNormal(zeroed_input,startPlanes)
    startDistancesMin = startDistances.min(dim=1)
    startExtremaIdx = startDistancesMin.indices
    startDistances -= startDistancesMin.values[:,None,:]
    startExtrema = getPerObjectIdx(startExtremaIdx,zeroed_input, device=device)
    startPlanes = projectVectorOnVector(startPlanes[:,:,None,:].repeat(objectsNr,1,1,1),startExtrema[:,:,None,:], epsilon=epsilon)
    planeNeighboursIdx = vertexPlaneNeighbours.repeat(objectsNr,1,1)
    vertices = planeNeighboursIdx.shape[1]
    #create new plane-vectors and norm them to the found extrema
    newPlanes = getPlaneNormalFrom3Points(getPerObjectIdx(planeNeighboursIdx.reshape(-1,vertices*3),
                                   startExtrema, device=device).reshape(objectsNr,vertices,3,3))
    distances = getPointDistances2PlaneNormal(zeroed_input,newPlanes)
    distancesMin = distances.min(dim=1)
    newExtremaIdx = distancesMin.indices #minimal distance is the "outermost" distance
    distances -= distancesMin.values[:,None,:]
    newExtrema = getPerObjectIdx(newExtremaIdx,zeroed_input, device=device)
    planeExtremaIdx = torch.cat([startExtremaIdx, newExtremaIdx],dim=1)
    newPlanes = projectVectorOnVector(newPlanes[:,:,None,:],newExtrema[:,:,None,:],epsilon=0.0)
    #some planes will be nan (planes that only had 2 distinct extrema 
    #because an extremum was shared between planes)
    #some will also have a normal length of 0 (2d objects) and no direction, which will give errors
    #There are a bunch of new Vertices calculate every possible vertex and then determine if it is inside the old planes

    ###calculate the new planeNeighboursIdxs###
    #every vertex will be a new plane
    #permute all old neighbourplanes with the vertexneighbourplanes with the new plane
    #newPlaneNeighboursIdx_t = torch.cat([planeNeighboursIdx,vertexNeighboursIdx + planeNr],dim=2)
    #newPlaneNeighboursIdx= newPlaneNeighboursIdx_t[:,:,[[0, 1],[0, 2],[0, 3],[0, 4],[0, 5],[1, 2],
    #                                                        [1, 3],[1, 4],[1, 5],[2, 3],[2, 4],[2, 5],
    #                                                        [3, 4],[3, 5],[4, 5]]]
    #newPlaneNeighboursIdx = torch.cat([torch.arange(planeNr,planeNr+newPlanes.shape[1], device=device)[None,:,None,None].repeat(newPlaneNeighboursIdx_t.shape[0],
    #                                                                        1,
    #                                                                        15,1), newPlaneNeighboursIdx],dim=-1)
    newPlanes = torch.cat([startPlanes,newPlanes],dim=1)
    newExtrema = torch.cat([startExtrema,newExtrema],dim=1)
    return newPlanes,newExtrema, planeExtremaIdx

def step_withoutNaN(startPlanes, objectPoints):
    newPlanes,newExtrema, planeExtremaIdx = step(startPlanes, objectPoints)
    invalidNormals = torch.isnan(newPlanes).sum(axis=2) > 0
    newPlanes[invalidNormals] = newPlanes[:,:6,:].repeat(1,3,1)[:,:14,:][invalidNormals]
    #add variability
    newPlanes[invalidNormals] = newPlanes[invalidNormals]+torch.rand_like(newPlanes[invalidNormals])*newPlanes[invalidNormals]*0.1
    return newPlanes


if __name__ == '__main__':
    points = torch.rand(2,12,3).cuda()
    #startplanes = torch.tensor([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])[None,:,:].cuda()
    planeHrep = step_withoutNaN(startPlanes, points.cuda())
    print(planeHrep)

    #torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])[None,:,:].cuda()