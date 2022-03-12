import pyredner
import torch
import linAlgHelper
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import ConvexHull, HalfspaceIntersection
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull
from neuralVolumeHelper import randCam, matrixLookat, createInputVector_planeHitModel,HiddenPrints, SIREN, getView, circular2sinCosC,bound2Mesh, compare2CenteredModels, bound2Pointcloud, meshIt, modelCenterCorrection, getPredictionPoints,compare2CenteredModels, bound2bounds, meshBoundsTM, mesh2pointcloud, array2Pointcloud

class NeuralBound:
    pointDoubleOccupationVector = torch.Tensor([[]])
    neuralBoundList = []
    unoccupiedRegions = torch.Tensor([[]])
    
    def createOccupationVector(points):
        NeuralBound.pointDoubleOccupationVector = torch.zeros_like(points[:,0])
        for volume in NeuralBound.neuralBoundList:
            volume.insideOccupationCheck(points)
        NeuralBound.unoccupiedRegions = NeuralBound.pointDoubleOccupationVector == 0
        NeuralBound.pointDoubleOccupationVector = NeuralBound.pointDoubleOccupationVector > 1
        
    def __init__(self, 
                 additionalBounds = None,
                 boundsize = 1.,
                 center=torch.Tensor([[0.,0.,0.]]), 
                 verbose=True,
                centerLR = 0.001,
                boundsLR = 0.01):
        self.center = center.cuda()
        self.bounds = torch.Tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])*boundsize
        if additionalBounds is not None:
            self.bounds = torch.cat((self.bounds,additionalBounds),0)
        self.bounds = self.bounds.cuda()
        self.name = "newBound"
        self.verbose = verbose
        self.center.requires_grad = True
        self.bounds.requires_grad = True
        self.centerOptim = torch.optim.Adam([self.center], lr=centerLR)
        self.boundsOptim = torch.optim.Adam([self.bounds], lr=boundsLR)
        NeuralBound.neuralBoundList.append(self)
        if self.verbose:
            print("{} was created at {}".format(self.name, self.center.cpu().detach()))
    
    def getVolume(self):
        return (self.bounds[0]-self.bounds[3])[0].detach()*(self.bounds[1]-self.bounds[4])[1].detach()*(self.bounds[2]-self.bounds[5])[2].detach()
    
    def clampBounds(self):
        with torch.no_grad():
            self.bounds[0,0].clamp(-99999,-0.01)
            self.bounds[0,1] = 0.
            self.bounds[0,2] = 0.
            self.bounds[3,0].clamp(0.01,99999)
            self.bounds[3,1] = 0.
            self.bounds[3,2] = 0.
            self.bounds[1,1].clamp(-99999,-0.01)
            self.bounds[1,0] = 0.
            self.bounds[1,2] = 0.
            self.bounds[4,1].clamp(0.01,99999)
            self.bounds[4,0] = 0.
            self.bounds[4,2] = 0.
            self.bounds[2,2].clamp(-99999,-0.01)
            self.bounds[2,0] = 0.
            self.bounds[2,1] = 0.
            self.bounds[5,2].clamp(0.01,99999)
            self.bounds[5,0] = 0.
            self.bounds[5,1] = 0.
        
    def insideOccupationCheck(self, points):
        '''Takes only surfacepoints (n,3)'''
        with torch.no_grad():
            centeredPoints = points-self.center
            boundsTest = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints[None,:,:], self.bounds[None,:,:])[0]
            inside = boundsTest>0
            completeInner = inside.sum(dim=1)==inside.shape[1]
            NeuralBound.pointDoubleOccupationVector += completeInner*1
    
    def boundsAdjustmentStep(self, surfacepoints, emptyVectors):
        '''gets tensor(n,3) surfacespoints (surface ) with (n,1) values (1 for surface, -1 for empty)'''
        size = self.bounds[:6].detach().abs().max()
        #create empty points
        emptypoints = torch.cat((surfacepoints+emptyVectors*0.1,
                                 surfacepoints+emptyVectors*0.15,
                                 surfacepoints+emptyVectors*0.6,
                                 surfacepoints+emptyVectors*size*0.3),0)        
        self.clampBounds()
        self.centerOptim.zero_grad()
        self.boundsOptim.zero_grad()
        centeredPoints_surface = surfacepoints-self.center
        centeredPoints_empty = emptypoints-self.center
        boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
        boundsTest_empty = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_empty[None,:,:], self.bounds[None,:,:])[0]
        with torch.no_grad():
            inside_surface = boundsTest_surface>0
            inside_empty = boundsTest_empty>0
            near_surface = boundsTest_surface>-size
            importance_surface = 0.1*size/(0.1*size+abs(boundsTest_surface)) 
            importance_empty = (0.1*size/(0.1*size+abs(boundsTest_empty)) )
            completeInner_surface = inside_surface.sum(dim=1)==inside_surface.shape[1]
            completeInner_empty = inside_empty.sum(dim=1)==inside_empty.shape[1]
            completeNear_surface = near_surface.sum(dim=1)==near_surface.shape[1]
        insideGradient_surfaceOverlap = torch.sigmoid(boundsTest_surface*4./(size))   #höherers loss weiter innen -nach aussen nimmt es ab
        insideGradient_empty = torch.sigmoid(boundsTest_empty*4./(size))   #höherers loss weiter innen -nach aussen nimmt es ab
        outsideGradient_surface = 1. - torch.sigmoid(boundsTest_surface*2./(size)) #höherers loss weiter aussen -nach innen nimmt es ab
        #recalculation of value (overlap and reproducing accuracy/certainty)
        #overlapLoss
        overLapFactor =(completeInner_surface*NeuralBound.pointDoubleOccupationVector)[:,None]  #smaller negative value to encourage seamless reconstructions
        overlapLoss = insideGradient_surfaceOverlap* overLapFactor *importance_surface**3
        missedPointFactor = (completeNear_surface*NeuralBound.unoccupiedRegions)[:,None]
        missedPointsLoss = outsideGradient_surface*missedPointFactor*importance_surface*3
        # for the inner loss: errors near a plane are important, but the gradient of the error should shrink, the nearer it gets
        innerEmptyFactor = completeInner_empty[:,None]
        innerEmptyLoss= insideGradient_empty*innerEmptyFactor*importance_empty**2
        error = overlapLoss.sum() +missedPointsLoss.sum()+ innerEmptyLoss.sum() 
        #print("overlap: ",overlapLoss.detach().sum().item(), " missedPoints: ", missedPointsLoss.detach().sum().item(), " innerEmpty: ", innerEmptyLoss.detach().sum().item())
        error.backward()
        self.centerOptim.step()
        self.boundsOptim.step()
        self.clampBounds()
        self.centerOptim.zero_grad()
        self.boundsOptim.zero_grad()
        return {"overlap":overLapFactor.sum().item(),
                "missedPoints": missedPointFactor.sum().item(),
                "inside Empty": innerEmptyFactor.sum().item(),
                "insideGradient_surfaceOverlap": insideGradient_surfaceOverlap.detach().sum().item(),
                "outsideGradient_surface": outsideGradient_surface.detach().sum().item(),
                "insideGradient_empty": insideGradient_empty.detach().sum().item()
                
                }
        

    def show(self):
        size = self.bounds[:6].detach().abs().max()
        data = torch.rand(10000,3).cuda()*2.*size-size
        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(data[None,:,:], self.bounds.detach()[None,:,:])[0]
        inside = boundsTest>0
        inside = inside.sum(dim=1)==inside.shape[1]
        filtered = torch.cat((data[inside],self.bounds.detach()),0)
        filteredIdx = torch.arange(len(filtered))
        hull = ConvexHull(filtered.cpu())
        verts_ = torch.tensor(hull.vertices)
        vertIdx = torch.arange(len(verts_))
        filteredIdx[verts_.long()] = vertIdx
        faces_ = torch.tensor(hull.simplices)
        vertices, faces =  filtered[verts_.long()], filteredIdx[faces_.long()]
        mesh = tm.Trimesh(vertices=vertices.cpu()+self.center.detach().cpu(), faces=faces.cpu())
        pointcloudPoints = mesh.sample(2000)
        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
        colors = np.ones_like(pointcloudPoints).astype(np.float64)
        colors[:,2] = colors[:,2]*np.random.rand()
        colors[:,1] = colors[:,1]*np.random.rand()
        colors[:,0] = colors[:,0]*np.random.rand()
        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
        return(pointcloudMesh)
    
    def train(points, cameraPosition):
        '''surface points in (n,3) and cameraposition in (3)'''
        points = points.cuda()
        loss = {"overlap":0.,
                "missedPoints": 0.,
                "inside Empty": 0.}
        NeuralBound.createOccupationVector(points)
        emptyVectors = (cameraPosition[None,None,:].cuda()-points).reshape(-1,3)
        emptyVectors = emptyVectors/((emptyVectors**2).sum(dim=-1))[:,None]**0.5
        for volume in NeuralBound.neuralBoundList:
            tempLoss = volume.boundsAdjustmentStep(points,emptyVectors)
            loss["overlap"] += tempLoss["overlap"]
            loss["missedPoints"] += tempLoss["missedPoints"]
            loss["inside Empty"] += tempLoss["inside Empty"]

        return loss

        
            
    

objects = pyredner.load_obj('/home/jhm/Documents/redner-master/tutorials/teapot.obj', return_objects=True)
objectsTM = tm.load('/home/jhm/Documents/redner-master/tutorials/teapot.obj')
pointcloudTarget = (objectsTM.sample(1500)/100.)
pointcloudT = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudTarget))
colors = np.ones_like(pointcloudTarget).astype(np.float64)
colors[:,2] = colors[:,0]*0.
colors[:,0] = colors[:,1]*0.
pointcloudT.colors = o3d.utility.Vector3dVector(colors)
loss = {"overlap":100.,
                "missedPoints": 100.,
                "inside Empty": 100.}
with torch.no_grad():
    img, pos, mask, cam = getView(objects)
#training des neural Volumes
bound1 = NeuralBound(additionalBounds = torch.rand((16,3))-0.5,
                 boundsize = 0.05,
                 center=pos[mask][-5,:][None,:], 
                 verbose=True,
                centerLR = 0.01,
                boundsLR = 0.001)

bound2 = NeuralBound(additionalBounds = torch.rand((16,3))-0.5,
                 boundsize = 0.03,
                 center=pos[mask][5,:][None,:], 
                 verbose=True,
                centerLR = 0.01,
                boundsLR = 0.001)

bound3 = NeuralBound(additionalBounds = torch.rand((16,3))-0.5,
                 boundsize = 0.03,
                 center=pos[mask][-150,:][None,:], 
                 verbose=True,
                centerLR = 0.01,
                boundsLR = 0.001)

trainingPointClouds = [pointcloudT]


print("beginning Training")
for iter_idx in range(500):
        #create data
        with torch.no_grad():
            img, pos, mask, cam = getView(objects)
            pos = pos
        #learn
        newLoss = NeuralBound.train(pos[mask].reshape(-1,3), cam.position)
        loss["overlap"] = loss["overlap"]*0.9 + 0.1*newLoss["overlap"]
        loss["missedPoints"] = loss["missedPoints"]*0.9 + 0.1*newLoss["missedPoints"]
        loss["inside Empty"] = loss["inside Empty"]*0.9 + 0.1*newLoss["inside Empty"]
        if iter_idx % 100 == 0:
            print("run ",iter_idx, " loss overlap: ", loss["overlap"], " loss missedPoints: ", loss["missedPoints"], " loss inside: ", loss["inside Empty"] )
            print("volume center ",bound1.center.detach(), " bound size ",bound1.bounds[:6].detach().abs().max())
            print()
            trainingPointClouds.append(bound1.show())
            trainingPointClouds.append(bound2.show())
            trainingPointClouds.append(bound3.show())
       
print("")
print("Bounds:")
print(bound1.bounds)
print(bound2.bounds)
print(bound3.bounds)
        
        
o3d.visualization.draw_geometries(trainingPointClouds)

o3d.visualization.draw_geometries([trainingPointClouds[0],trainingPointClouds[-1],trainingPointClouds[-2],trainingPointClouds[-3]])

