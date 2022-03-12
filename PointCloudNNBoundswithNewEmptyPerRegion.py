# %%
import torch
import linAlgHelper
from scipy.spatial import ConvexHull, HalfspaceIntersection
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
from neuralVolumeHelper import matrixLookat, createInputVector_planeHitModel,HiddenPrints, SIREN, circular2sinCosC,bound2Mesh, compare2CenteredModels, bound2Pointcloud, meshIt, modelCenterCorrection, getPredictionPoints,compare2CenteredModels, bound2bounds, meshBoundsTM, mesh2pointcloud, array2Pointcloud
import open3d as o3d
import trimesh as tm
from PointCloudNeuralNetDist import NeuralConvexReconstruction
from pathlib import Path
import os
import gc

pfad = "/files/Code/convexNeuralVolumeData"
pfad = "/home/jhm/Desktop/Arbeit/ConvexNeuralVolume"


ball = tm.primitives.Capsule(radius=1., height=0.,sections=128)

#areasize = 1

debug = False

debugSingle = True
debug_all_iters = 0
debug_iter = 0

def loadPoints(x_block,y_block,z_block):
    my_file = Path(pfad+"/blocks/{}x{}y{}z.npy".format(x_block,y_block,z_block))
    if my_file.is_file():
        points = np.load(my_file)
        return points

def loadEmptyPoints(x_block,y_block,z_block):
    my_file = Path(pfad+"/emptyBlocks/e_{}x{}y{}z.npy".format(x_block,y_block,z_block))
    if my_file.is_file():
        points = np.load(my_file)
        return points


def createBounds(size = 10.):
        bounds = torch.zeros(12,3).cuda()
        bounds[0,0] = -(np.random.rand()+0.3)
        bounds[3,0] = (np.random.rand()+0.3)
        bounds[1,1] = -(np.random.rand()+0.3)
        bounds[4,1] = (np.random.rand()+0.3)
        bounds[2,2] = -(np.random.rand()+0.3)
        bounds[5,2] = (np.random.rand()+0.3)
        bounds[6:12] = bounds[0:6] + (torch.rand((6,3)).cuda()*0.1-0.05)
        bounds = bounds*size
        return bounds


class NeuralBound:
    activeSet = set()
    pointDoubleOccupationVector = torch.Tensor([[]])
    neuralBoundList = []
    unoccupiedRegions = torch.Tensor([[]])
    pointsVolumeOverlapVector = torch.Tensor([[0.,0.,0.]]).cuda()
    pointsVolumeOverlapVectorDoubleOccupation = torch.Tensor([[0.,0.,0.]]).cuda()

    def deactivateAll():
        for i in list(NeuralBound.activeSet):
            NeuralBound.activeSet.remove(i)
            vol = NeuralBound.neuralBoundList[i]
            NeuralBound.neuralBoundList[i] = None
            del vol
    
    def __init__(self, 
                 bounds,
                 center=torch.Tensor([[0.,0.,0.]]), 
                centerLR = 0.001,
                boundsLR = 0.01,
                variableFaktoren = [2,2,2,3,3,3,5.,1.,2.],
                maxPointsInput = 150000,
                maxPointsChunk = 30000):
        '''bounds should start with: [1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]'''
        self.center = center.cuda()
        self.noOverlap = True
        self.centerBackup = center
        self.boundsBackup = bounds
        self.bounds = bounds
        self.bounds = self.bounds.cuda()
        self.center.requires_grad = True
        self.bounds.requires_grad = True
        self.centerOptim = torch.optim.Adam([self.center], lr=centerLR)
        self.boundsOptim = torch.optim.Adam([self.bounds], lr=boundsLR)
        self.variableFaktoren = variableFaktoren
        self.neuralReconstruction = NeuralConvexReconstruction()
        self.maxpointsInput = maxPointsInput
        self.maxpointsChunk = maxPointsChunk
        self.id = NeuralBound.neuralBoundList.__len__()
        NeuralBound.neuralBoundList.append(self)
        self.neuralNetTrained = True
        self.adjustedForEpochs = 0
        NeuralBound.activeSet.add(self.id)
        self.debug = False

    def deactivate(self):
        NeuralBound.activeSet.remove(self.id)
        NeuralBound.neuralBoundList[self.id] = None
    
    def getVolume(self):
        return ((self.bounds[3]-self.bounds[0])[0].detach()*(self.bounds[4]-self.bounds[1])[1].detach()*(self.bounds[5]-self.bounds[2])[2].detach()).item()

    def getBounds(self, addDistance=0.0):
        minX = (self.center+self.bounds[0])[0].detach()-addDistance
        maxX = (self.center+self.bounds[3])[0].detach()+addDistance
        minY = (self.center+self.bounds[1])[1].detach()-addDistance
        maxY = (self.center+self.bounds[4])[1].detach()+addDistance
        minZ = (self.center+self.bounds[2])[2].detach()-addDistance
        maxZ = (self.center+self.bounds[5])[2].detach()+addDistance
        return [minX,maxX,minY,maxY,minZ,maxZ]
    
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

    def checkNan(self):
       if ((self.center >= 0.)*1 + (self.center < 0.1)*1).sum() < 3:
               print("center was NaN")
               self.center = self.centerBackup.cuda().clone()
               self.center.requires_grad = True
               self.centerOptim = torch.optim.Adam([self.center], lr=self.variableFaktoren[8])
       if ((self.bounds >= 0.)*1 + ((self.bounds < 0.)*1)).sum() < (len(self.bounds)*3):
               print("bounds were NaN")
               self.bounds = self.boundsBackup.cuda().clone()
               self.bounds.requires_grad=True
               self.boundsOptim = torch.optim.Adam([self.bounds], lr=self.variableFaktoren[9])
    
    def boundsAdjustmentStep(self, surfacepoints, overlappoints, emptyPointPrototypes, emptyPointCellsize, color):
        '''gets tensor(n,3) surfacespoints (surface ) with (n,1) values (1 for surface, -1 for empty)'''
        self.centerBackup = self.center.detach().clone()
        self.boundsBackup = self.bounds.detach().clone()
        size = self.bounds[:6].detach().abs().max()
        try:
            if surfacepoints is None:
                return {"overlap":0.,
                "missedPoints": 0.,
                "flippotential": 0.,
                "innersurfaceLoss": 0.,
                "inside Empty": 0.}, False
        except: pass
        try: 
            if overlappoints is None:
                overlappoints = []
        except: pass
        try:
            if emptyPointPrototypes is None:
                emptyPointPrototypes = torch.tensor([[0.,0.,0.,]])
        except: pass
        if len(surfacepoints) > self.maxpointsInput:
            surfacepoints = surfacepoints.cpu()
        if len(overlappoints) > self.maxpointsInput:
            overlappoints = overlappoints.cpu()
        if len(emptyPointPrototypes) > self.maxpointsInput:
            emptyPointPrototypes = emptyPointPrototypes.cpu()
        if len(color) > self.maxpointsInput:
            color = color.cpu()
        missedPointsLoss_all = torch.tensor(0.)
        innerSurfaceLoss_all = torch.tensor(0.)
        innerEmptyLoss_all  = torch.tensor(0.)
        doubleOccLoss_all = torch.tensor(0.)
        flipPotential_all = torch.ones((len(self.bounds)))
        flipPotential = torch.ones((len(self.bounds)))
        killme = True
        allSurfaceWeight = 0
        #split inputdata
        for endFaktor in range(0,len(surfacepoints)//self.maxpointsInput+1):
            surfacepoints_ = (surfacepoints[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]).detach().float().cuda()
            color_ = (color[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]).detach().float().cuda()
            #test that there are not more valid points than the chunksize
            with torch.no_grad():
                centeredPoints_surface = surfacepoints_-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
                near_surface = boundsTest_surface>-size*self.variableFaktoren[0]
                completeNear_surface = near_surface.sum(dim=1)==near_surface.shape[1]
                in_surface = boundsTest_surface>0
                completeIn_surface = in_surface.sum(dim=1)==in_surface.shape[1]
                
                
                del boundsTest_surface,near_surface, in_surface
            surfacepoints_ = surfacepoints_[completeNear_surface]
            color_ = color_[completeNear_surface]
            inMask = completeIn_surface[completeNear_surface]
            if self.debug:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(surfacepoints_.detach().cpu().numpy())
                pcd.colors = o3d.utility.Vector3dVector(color_.detach().cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
            #split surface-point learning
            for chunkFactor in range(len(surfacepoints_)//self.maxpointsChunk+1):
                surfacepointsChunk = surfacepoints_[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                colorChunk = color_[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                inMaskChunk = inMask[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                centeredPoints_surface = surfacepointsChunk-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]/size
                value = ((torch.clamp(torch.sigmoid(boundsTest_surface.detach()*10),0.,0.5)*2.)).prod(dim=1)
                if self.debug:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(surfacepoints_.detach().cpu().numpy())
                    pcd.colors = o3d.utility.Vector3dVector((color_*value[:,None]).detach().cpu().numpy())
                    o3d.visualization.draw_geometries([pcd])
                innerSurfaceLoss = torch.tensor(0.)
                if len(boundsTest_surface > 0):
                    flipPotential = torch.zeros_like(boundsTest_surface[0,:])
                    if inMaskChunk.sum() > 0:
                        killme = False
                    if inMaskChunk.sum() > 10:
                        innerSurfaceLoss = (boundsTest_surface[inMaskChunk]).min(axis=1)[0].mean()*self.variableFaktoren[10]
                    if inMaskChunk.sum() > 80:
                        flipPotential = boundsTest_surface[inMaskChunk].min(axis=0)[0]
                #train neural reconstruction -bounds influence training value
                points = centeredPoints_surface.detach()
                del centeredPoints_surface
                if len(points) > 0:
                    lastNNLoss, NNDifference = self.neuralReconstruction.train(points.float(), value.float())
                    self.neuralReconstruction.trainColor(points.float(), value.float(), colorChunk.float())
                    del value, colorChunk, points
                    if abs(NNDifference).mean() < size*0.1:
                        self.neuralNetTrained = True
                    else:
                        self.neuralNetTrained = False
                    NNValue = (torch.relu(size*0.7-abs(NNDifference))/size*0.7).detach()
                    if self.debug:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(surfacepoints_.detach().cpu().numpy())
                        pcd.colors = o3d.utility.Vector3dVector((color_*NNValue).detach().cpu().numpy())
                        o3d.visualization.draw_geometries([pcd])
                    del NNDifference
                    # adjust bounds
                    outsideGradient_surface = torch.nn.functional.relu(torch.tanh(-boundsTest_surface*self.variableFaktoren[3]))*(NNValue*0.5+0.5)
                    del NNValue, boundsTest_surface
                    torch.cuda.empty_cache()
                    #Regularize Model center to Prediction center
                    with torch.no_grad():
                        self.centerBackup = self.center.detach().clone()
                    centerCorrection = modelCenterCorrection(self.neuralReconstruction.learnModel,self.neuralReconstruction.learnModelLastLayer)
                    centerError = torch.nn.functional.l1_loss(self.center,centerCorrection+self.center.detach())
                    del centerCorrection
                    innerSurfaceLoss_all = (innerSurfaceLoss_all +innerSurfaceLoss.detach()).detach()
                    missedPointsLoss = (outsideGradient_surface).sum()/(outsideGradient_surface>0).sum()*self.variableFaktoren[6] +((flipPotential*self.variableFaktoren[11]).mean()+innerSurfaceLoss)
                    del outsideGradient_surface
                    #add centerCorrection to loss
                    missedPointsLoss = missedPointsLoss+centerError
                    missedPointsLoss_all += missedPointsLoss.detach().cpu()
                    missedPointsLoss.backward()
                    self.centerOptim.step()
                    self.boundsOptim.step()
                    self.clampBounds()
                    self.centerOptim.zero_grad()
                    self.boundsOptim.zero_grad()
                    self.checkNan()
                    del centerError, completeIn_surface
                    
                    #del missedPointFactor, value, difference
            flipPotential_all = torch.stack((flipPotential.cpu().detach(),flipPotential_all),0).min(axis=0)[0]
        #adjust for empty points 
        #adjust empty and surface-point sizes
        repeatfactor = len(surfacepoints)//max(1,len(emptyPointPrototypes))
        emptyPointPrototypes = emptyPointPrototypes.repeat(repeatfactor,1)
        #chunk to max size
        for endFaktor in range(0,len(emptyPointPrototypes)//self.maxpointsInput+1):
                if len(emptyPointPrototypes) > 0:
                    emptyPointPrototypes_ = torch.tensor(emptyPointPrototypes[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]).float().cuda()
                    #test that there are not more valid points than the chunksize
                    with torch.no_grad():
                        centeredPoints_empty = emptyPointPrototypes_-self.center
                        boundsTest_empty = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_empty[None,:,:], self.bounds[None,:,:])[0]
                        near_empty = boundsTest_empty>-emptyPointCellsize
                        completeNear_empty = near_empty.sum(dim=1)==near_empty.shape[1]
                        del boundsTest_empty,near_empty
                    if self.debug:
                                cloud = self.show()
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(surfacepoints_.detach().cpu().numpy())
                                pcd.colors = o3d.utility.Vector3dVector((color_).detach().cpu().numpy())
                                pcd2 = o3d.geometry.PointCloud()
                                pcd2.points = o3d.utility.Vector3dVector(emptyPointPrototypes_.detach().cpu().numpy())
                                c2 = torch.ones_like(emptyPointPrototypes_) 
                                c2[:,2] = c2[:,2]*0. 
                                c2[:,0] = c2[:,0]*0.
                                pcd2.colors = o3d.utility.Vector3dVector((c2).detach().cpu().numpy())
                                o3d.visualization.draw_geometries([pcd,pcd2,cloud])
                    centeredPoints_empty = centeredPoints_empty[completeNear_empty]
                    
                    #split empty-point learning
                    for chunkFactor in range(len(centeredPoints_empty)//self.maxpointsChunk+1):
                            emptyPrototypes_ = centeredPoints_empty[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk].cuda()
                            #create empty points
                            emptypoints = emptyPrototypes_+((torch.rand_like(emptyPrototypes_)-0.5).cuda() *2*2*emptyPointCellsize)   
                            if self.debug:
                                cloud = self.show()
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(surfacepoints.detach().cpu().numpy())
                                pcd.colors = o3d.utility.Vector3dVector((color).detach().cpu().numpy())
                                pcd2 = o3d.geometry.PointCloud()
                                pcd2.points = o3d.utility.Vector3dVector((emptypoints+self.center).detach().cpu().numpy())
                                o3d.visualization.draw_geometries([pcd,pcd2,cloud])                           
                            boundsTest_empty = linAlgHelper.getPointDistances2PlaneNormal(emptypoints[None,:,:].float(), self.bounds[None,:,:].float())[0]
                            inner_empty = boundsTest_empty>0.
                            completeInner_empty = inner_empty.sum(dim=1)==inner_empty.shape[1]
                            emptypoints = emptypoints[completeInner_empty]
                            if len(emptypoints) >0:
                                #insideGradient_empty = boundsTest_empty[completeInner_empty].min(axis=1)[0]/size*self.variableFaktoren[2]
                                insideGradient_empty = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_empty[completeInner_empty]*self.variableFaktoren[2]/size),0.001)
                                innerEmptyLoss= insideGradient_empty.mean()*self.variableFaktoren[7]
                                innerEmptyLoss.backward()
                                innerEmptyLoss_all += innerEmptyLoss.detach().cpu()
                                del insideGradient_empty, boundsTest_empty
                                self.centerOptim.step()
                                self.boundsOptim.step()
                                self.clampBounds()
                                self.centerOptim.zero_grad()
                                self.boundsOptim.zero_grad()
                                self.checkNan()
        
        for endFaktor in range(0,len(overlappoints)//self.maxpointsInput+1):   
            if len(overlappoints) == 0: 
                self.noOverlap = True 
                continue
            #test that there are not more valid points than the chunksize
            with torch.no_grad():
                centeredPoints_overlap = torch.tensor(overlappoints[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]).cuda()-self.center
                boundsTest_doubleOcc = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_overlap[None,:,:], self.bounds[None,:,:])[0]
                near_doubleOcc = boundsTest_doubleOcc>-size*self.variableFaktoren[1]   #self.variableFaktoren[2] = 1.0
                completeNear_doubleOcc = near_doubleOcc.sum(dim=1)==near_doubleOcc.shape[1]
            #chunk the relevant points
             #split surface-point learning
            inputPoints = centeredPoints_overlap[completeNear_doubleOcc]
            for chunkFactor in range(len(inputPoints)//self.maxpointsChunk+1):
                centeredPoints_overlap_chunk = inputPoints[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                boundsTest_doubleOcc = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_overlap_chunk[None,:,:], self.bounds[None,:,:])[0]
                with torch.no_grad():
                    in_surface = boundsTest_doubleOcc>0
                    completeIn_surface = in_surface.sum(dim=1)==in_surface.shape[1]
                    if completeIn_surface.sum() > 0:
                        self.noOverlap = False 
                    else:
                        self.noOverlap = True 
                insideGradient_doubleOcc = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_doubleOcc*self.variableFaktoren[4]/size),0.001)
                if len(insideGradient_doubleOcc) > 0:
                    doubleOccLoss = (self.variableFaktoren[5]*insideGradient_doubleOcc.mean())
                    doubleOccLoss_all += doubleOccLoss.detach().cpu()
                    doubleOccLoss.backward()
                    self.centerOptim.step()
                    self.boundsOptim.step()
                self.clampBounds()
                self.centerOptim.zero_grad()
                self.boundsOptim.zero_grad()
                self.checkNan()
        # flip if necessary
        maxFlip = flipPotential_all.argmax()
        if flipPotential_all[maxFlip] > 1.0:
            print("moved from ",self.center)
            self.center = self.center - flipPotential_all[maxFlip]
            print(" to ",self.center)
            self.checkNan()
        self.debug = False
        return {"overlap":doubleOccLoss_all.item(),
                "missedPoints": missedPointsLoss_all.item(),
                "flippotential": flipPotential_all.max(),
                "innersurfaceLoss": innerSurfaceLoss_all.item(),
                "inside Empty": innerEmptyLoss_all.item()}, killme

    def getCellBlocks(self, distance=0.0):
        minX,maxX,minY,maxY,minZ,maxZ = self.getBounds(addDistance=distance)
        cells = set()
        for x in range(int(minX//35)-1,int(maxX//35)+1):
            for y in range(int(minY//35)-1,int(maxY//35)+1):
                for z in range(int(minZ//35)-1,int(maxZ//35)+1):
                    cells.add((x,y,z))
        return cells    

    def filterPoints(self, points, mask):
        size = self.bounds[:6].detach().abs().max()
        with torch.no_grad():
                centeredPoints = points-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints[None,:,:], self.bounds[None,:,:])[0]
                in_surface = boundsTest_surface>-size*0.1
                completeIn_surface = in_surface.sum(dim=1)==in_surface.shape[1]
                mask[completeIn_surface] = True
        return mask

    
    def train(self, neuralVolumeCellRegister, loss):
        self.adjustNeuralRecon2Bounds(iters=2)
        # get the relevant pointcells
        selfCells = self.getCellBlocks()
        #delete own cells from CellRegister
        try:
           otherVolumePoints = neuralVolumeCellRegister.popNearbyVolumePoints(selfCells, self.id)
        except:
           otherVolumePoints = None
        points = None 
        emptyPoints = None
        size = self.bounds[:6].detach().abs().max()
        selfCells = self.getCellBlocks(distance=size)
        for cell in list(selfCells):
            if emptyPoints is None:
                try:
                    emptyPoints = torch.tensor(loadEmptyPoints(cell[0],cell[1],cell[2]))
                except: pass
            else:
                try:
                    emptyPoints = torch.cat((emptyPoints,torch.tensor(loadEmptyPoints(cell[0],cell[1],cell[2]))),0)
                except: pass
            try:
                if points == None:
                    points = torch.tensor(loadPoints(cell[0],cell[1],cell[2]))
                else:
                    points = torch.cat((points,torch.tensor(loadPoints(cell[0],cell[1],cell[2]))),0)
            except:
                try:
                    points = torch.cat((points,torch.tensor(loadPoints(cell[0],cell[1],cell[2]))),0)
                except: pass
        
        killMe = False
        try:
            if points is None:
                print("No points inside")
                killMe = True
                return loss, killMe
        except:
            pass
        if size < 0.04:
            print("Size to small")
            killMe = True
        if debug:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:,:3])
            pcd.colors = o3d.utility.Vector3dVector((points[:,3:]))
            try: 
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(emptyPoints)
                c2 = np.ones_like(emptyPoints) 
                c2[:,2] = c2[:,2]*0. 
                c2[:,0] = c2[:,0]*0.
                pcd2.colors = o3d.utility.Vector3dVector((c2))
                o3d.visualization.draw_geometries([pcd,pcd2])
            except: o3d.visualization.draw_geometries([pcd])
        color = points[:,3:6]
        points = points[:,:3]
        if debug:
            self.debug = True
        if self.noOverlap:
            for subiter in range(50):
                vLoss, killMe = self.boundsAdjustmentStep(points, otherVolumePoints,  emptyPoints, 4., color)
                if self.noOverlap == False:
                    break
        else:
            vLoss, killMe = self.boundsAdjustmentStep(points, otherVolumePoints,  emptyPoints, 4., color)
        selfCells = self.getCellBlocks()
        neuralVolumeCellRegister.registerId(selfCells, self.id)
        loss["overlap"] += vLoss["overlap"]
        loss["missedPoints"] += vLoss["missedPoints"]
        loss["innersurfaceLoss"] += vLoss["innersurfaceLoss"]
        loss["flippotential"] += vLoss["flippotential"].item()
        loss["inside Empty"] += vLoss["inside Empty"]
        loss["volumes"] +=  1
        return loss, killMe
    
    def getInsidePoints(self, pointNr = 10000):
        minX,maxX,minY,maxY,minZ,maxZ = self.getBounds()
        data = (torch.rand(int(abs(pointNr)),3).cuda()-0.5) * 2.*torch.tensor([[maxX-minX, maxY-minY, maxZ-minZ]]).cuda()# + torch.tensor([[minX, minY, minZ]]).cuda()
        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(data[None,:,:], self.bounds.detach()[None,:,:])[0]
        inside = boundsTest>0
        inside = inside.sum(dim=1)==inside.shape[1]
        filtered = torch.cat([data[inside].cpu(),self.bounds.detach().cpu()],0) + self.center.detach().cpu()
        return filtered

    def adjustNeuralRecon2Bounds(self, iters = 100):
        filtered = self.getInsidePoints()
        filteredIdx = torch.arange(len(filtered))
        hull = ConvexHull(filtered)
        verts_ = torch.tensor(hull.vertices)
        vertIdx = torch.arange(len(verts_))
        filteredIdx[verts_.long()] = vertIdx
        faces_ = torch.tensor(hull.simplices)
        vertices, faces =  filtered[verts_.long()], filteredIdx[faces_.long()]
        mesh = tm.Trimesh(vertices=vertices-self.center.detach().cpu().numpy(), faces=faces)
        for i in range(iters):
            pointcloudPoints = torch.tensor(mesh.sample(20000)).float().cuda()
            self.neuralReconstruction.train(pointcloudPoints, torch.ones_like(pointcloudPoints[:,0]).float().cuda())

    def show(self):
        filtered = self.getInsidePoints()
        filteredIdx = torch.arange(len(filtered))
        hull = ConvexHull(filtered)
        verts_ = torch.tensor(hull.vertices)
        vertIdx = torch.arange(len(verts_))
        filteredIdx[verts_.long()] = vertIdx
        faces_ = torch.tensor(hull.simplices)
        vertices, faces =  filtered[verts_.long()], filteredIdx[faces_.long()]
        mesh = tm.Trimesh(vertices=vertices, faces=faces)
        pointcloudPoints = mesh.sample(2000)
        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
        colors = np.ones_like(pointcloudPoints).astype(np.float64)
        colors[:,2] = colors[:,2]*np.random.rand()
        colors[:,1] = colors[:,1]*np.random.rand()
        colors[:,0] = colors[:,0]*np.random.rand()
        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
        return(pointcloudMesh)

heightsKeys = np.load(pfad+"/heights.npy")
heigthsValues = np.load(pfad+"/heightsValues.npy")
heights = {}
for i in range(len(heightsKeys)):
    heights[(heightsKeys[i][0],
            heightsKeys[i][1])] = heigthsValues[i]


class NeuralVolumeCellRegister:
    def __init__(self):
        self.registeredCellBlocks = {}
        self.minXBlock = 999999
        self.maxXBlock = -999999
        self.minYBlock = 999999
        self.maxYBlock = -999999
        self.minZBlock = 999999
        self.maxZBlock = -999999

    def popNearbyVolumePoints(self, selfCells, id):
        points = None
        idList = []
        for cell in selfCells:
            if cell in self.registeredCellBlocks:
                self.registeredCellBlocks[cell].remove(id)
                idList +=(self.registeredCellBlocks[cell])
                if len(self.registeredCellBlocks[cell]) == 0:
                    del self.registeredCellBlocks[cell]
        try:
            idList = np.unique(np.array(idList)).tolist()
        except: pass
        for vid in idList:
            v = NeuralBound.neuralBoundList[vid]
            if v is None:
                continue
            if points is None:
                points = v.getInsidePoints(pointNr = min(v.getVolume()*1000,10000))
            else:
                points = torch.cat([points,v.getInsidePoints(pointNr = min(v.getVolume()*1000,10000))],0)
        return points

    def showNearbyVolumes(self, xmin,xmax,ymin,ymax):
        idList = []
        for x in range(xmin,xmax):
            for y in range(ymin,ymax):
                if (x,y) in heights: 
                    for z in range(heights[(x,y)][0],heights[(x,y)][1]):
                        if (x,y,z) in self.registeredCellBlocks:
                            idList +=(self.registeredCellBlocks[(x,y,z)])
        try:
            idList = np.unique(np.array(idList)).tolist()
        except: pass
        return idList

    def registerId(self, selfCells, id):
        for cell in selfCells:
            if cell in self.registeredCellBlocks:
                self.registeredCellBlocks[cell].append(id)
            else:
                self.registeredCellBlocks[cell] = [id]
    
    '''def createNewNeuralBounds(self, xBlock,yBlock,zBlock):
        points = loadPoints(xBlock,yBlock,zBlock)
        if points is None:
            return
        if (xBlock,yBlock,zBlock) not in self.registeredCellBlocks:
            vNew = '''


register = NeuralVolumeCellRegister()

gridparameter = []   # 0: nearSurfaceFaktor,  
                    # 1: nearDoubleOccFaktor, 
                    # 2: insideEmptySigmoidFaktor, 
                    # 3: missedPointLossSigmoidFaktor,
                    # 4: volumeOverlapSigmoidFaktor, 
                    # 5: overLapFactor
                    # 6: missedPointFaktor,
                    # 7:innerEmptyFaktor,
                    # 8: LR center
                    # 9: LR bounds
                    # 10: Surfacelossfactor
                    # 11: FlippotentialFactor
nearSurfaceFaktor = [0.4]
nearDoubleOccFaktor = [0.]
insideEmptySigmoidFaktor = [2.]
missedPointLossSigmoidFaktor = [2.]
volumeOverlapSigmoidFaktor = [2.]
overLapFactorChoice = [10.]
missedPointFaktor = [4.]
innerEmptyFaktor = [10.]
LR_center = [0.001]
LR_bounds = [0.003]
Surfacelossfactor = [1.5]
FlippotentialFactor = [1.]
gridparameter=torch.tensor([np.random.choice(nearSurfaceFaktor,1),
                    np.random.choice(nearDoubleOccFaktor,1),
                    np.random.choice(insideEmptySigmoidFaktor,1),
                    np.random.choice(missedPointLossSigmoidFaktor,1),
                    np.random.choice(volumeOverlapSigmoidFaktor,1),
                    np.random.choice(overLapFactorChoice,1),
                    np.random.choice(missedPointFaktor,1),
                    np.random.choice(innerEmptyFaktor,1),
                    np.random.choice(LR_center,1),
                    np.random.choice(LR_bounds,1),
                    np.random.choice(Surfacelossfactor,1),
                    np.random.choice(FlippotentialFactor,1)])[:,0]

#only for the first time!
if len(list(os.listdir(pfad+"/saveNeuralNetwork/")))==0:
    #load and register all neural volumes
    volList = os.listdir(pfad+"/neuralVolumes")
    for i,v in enumerate(volList):
        if i%100==0:
            print("initialize volume ",i," of ",len(volList))
        array = np.load(pfad+"/neuralVolumes/"+v)
        center = array[0]
        Hrep = array[1:]  
        newVolume = NeuralBound(torch.tensor(Hrep).cuda() ,center =torch.tensor(center).cuda(),centerLR = gridparameter[-2],
                            boundsLR = gridparameter[-1],
                            variableFaktoren= torch.tensor(gridparameter[:-2]).cuda())
        np.save(pfad+"/saveNeuralNetwork/{}".format(newVolume.id), torch.cat((newVolume.bounds,newVolume.center[None,:]),0).detach().cpu().numpy())
        torch.save(newVolume.neuralReconstruction,pfad+"/saveNeuralNetworkReconstruction/{}.pt".format(newVolume.id))
        #deregister Neural Volume
        NeuralBound.neuralBoundList[-1] = None
        del newVolume

#extend neuralBoundList to max ID of saved neuralVolumes
maxid = 0
for name in os.listdir(pfad+"/saveNeuralNetwork/"):
    tempid = int(name.split(".")[0])
    if tempid > maxid:
        maxid = tempid
NeuralBound.neuralBoundList = [None]*maxid 

def reloadNeuralVolume(volumeId, gridparameter):
    if os.path.isfile(pfad+"/saveNeuralNetworkReconstruction/{}.pt".format(volumeId)):
        newVolData = np.load(pfad+"/saveNeuralNetwork/"+"{}.npy".format(volumeId))
        newVolume = NeuralBound(torch.tensor(newVolData[:-1]).cuda(), center = torch.tensor(newVolData[-1]).cuda(), centerLR = gridparameter[8],
                        boundsLR = gridparameter[9],
                        variableFaktoren= torch.tensor(gridparameter).cuda())
        newVolume.deactivate()
        newVolume.id = volumeId
        NeuralBound.neuralBoundList.pop(-1)
        NeuralBound.neuralBoundList[volumeId] = newVolume
        NeuralBound.activeSet.add(volumeId)
        neuralReconstruction = torch.load(pfad+"/saveNeuralNetworkReconstruction/{}.pt".format(volumeId))
        newVolume.neuralReconstruction = neuralReconstruction
        neuralReconstruction.optimLearn = torch.optim.Adam(neuralReconstruction.learnModel.parameters(), lr=0.0001)
        neuralReconstruction.optimLearnReg = torch.optim.Adam(neuralReconstruction.learnModel.parameters(), lr=0.0001)
        neuralReconstruction.optimLastReg= torch.optim.Adam(neuralReconstruction.learnModelLastLayer.parameters(), lr=0.0001)
        neuralReconstruction.optimLast = torch.optim.Adam(neuralReconstruction.learnModelLastLayer.parameters(), lr=0.0001)
        neuralReconstruction.lastLoss = 1.
        neuralReconstruction.optimColor = torch.optim.Adam(neuralReconstruction.colorModel.parameters(), lr=0.0001)
        cells = newVolume.getCellBlocks()
        register.registerId(cells, newVolume.id)

#register all cells but deactivate them afterwards
for i in range(len(NeuralBound.neuralBoundList)):
    if i % 100 == 0:
        print("register cells ",i," of ",len(NeuralBound.neuralBoundList))
    reloadNeuralVolume(i,gridparameter)
    b = NeuralBound.neuralBoundList[i]
    if b is not None:
        cells = b.getCellBlocks()
        register.registerId(cells, b.id)
        b.deactivate()
        del b



def spawnNewVolumes(x,y,z):
    try:
        points = torch.tensor(loadPoints(x,y,z))[:,:3].float().cuda()
    except:
        return
    mask = torch.zeros_like(points[:,0],dtype=bool).cuda()
    if points is None:
        return
    ids = []
    if (x,y,z) in register.registeredCellBlocks:
        ids = register.registeredCellBlocks[(x,y,z)]
        for id in ids:
            v = NeuralBound.neuralBoundList[id]
            try:
               mask = v.filterPoints(points[:,:3],mask)
            except: pass
    try: 
         if len(mask) > 0: 
            pass
    except:
        return
    startPoints = points[~mask]
    startPoints[:,2] = startPoints[:,2] - 0.3
    if len(startPoints)>0:
        for i in range(int(len(startPoints)/5000)+1):
            pointNr = np.random.randint(len(startPoints))
            newVolume = NeuralBound(createBounds(size=4.).float().cuda() ,center =startPoints[pointNr].float().cuda(),centerLR = gridparameter[8],
                        boundsLR = gridparameter[9],
                        variableFaktoren= torch.tensor(gridparameter).cuda())
            newVolume.adjustNeuralRecon2Bounds(iters=300)
            cells = newVolume.getCellBlocks()
            register.registerId(cells, newVolume.id)

def trainActiveVolumes(iterations, register, tryNr, region, anchorPoints):
    global debug_all_iters, debug_iter, debug
    
    percentFree = 100.
    for iteration in range(iterations):
        debug_iter += 1
        if debug_all_iters > 0:
            if debug_iter%debug_all_iters==0:
                debug = True 
            else:
                debug = False
        percentFree = 100.0
        with torch.no_grad():
            mask = torch.zeros_like(anchorPoints[:,0],dtype=bool).cuda()
            for ida in list(NeuralBound.activeSet):
                v = NeuralBound.neuralBoundList[ida]
                if v is None: 
                    continue
                mask = v.filterPoints(anchorPoints[:,:3],mask)
            percentFree = "{} of {} = {}%".format((~mask).sum().item(), len(anchorPoints),(~mask).sum().item()/len(anchorPoints) * 100.)
        killList = []
        loss = {}
        loss["overlap"] = torch.tensor(0.)
        loss["missedPoints"] = torch.tensor(0.)
        loss["innersurfaceLoss"] = torch.tensor(0.)
        loss["flippotential"] = torch.tensor(0.)
        loss["volumes"] = torch.tensor(0)
        loss["inside Empty"] = torch.tensor(0.)
        for i, volume in enumerate(list(NeuralBound.activeSet)):
            volume = NeuralBound.neuralBoundList[volume]
            if volume is None:
                continue
            print("training volume ",i," of ",len(NeuralBound.activeSet), "iteration: ", iteration, " try: ", tryNr, " region: ", region, " percentFree: ",percentFree)
            loss, killme = volume.train(register, loss)
            torch.cuda.empty_cache()
            if killme:
                killList.append(volume.id)
        for i in killList:
            cells = NeuralBound.neuralBoundList[i].getCellBlocks()
            try:
                register.popNearbyVolumePoints(cells, i) 
                os.remove(pfad+"/saveNeuralNetwork/{}.npy".format(i))
                os.remove(pfad+"/saveNeuralNetworkReconstruction/{}.pt".format(i))
                NeuralBound.neuralBoundList[i] = None
                NeuralBound.activeSet.remove(i)
            except: print("could not kill volume")
    if len(NeuralBound.activeSet)>0:
        print(loss)
    for i, volume in enumerate(list(NeuralBound.activeSet)):
        volume = NeuralBound.neuralBoundList[volume]
        np.save(pfad+"/saveNeuralNetwork/{}".format(volume.id), torch.cat((volume.bounds,volume.center[None,:]),0).detach().cpu().numpy())
        torch.save(volume.neuralReconstruction,pfad+"/saveNeuralNetworkReconstruction/{}.pt".format(volume.id))

#train a small region until there are less than 5% free points

#for anchor in heights.keys():
for anchor in range(1):
    anchor = (123, 134)
    print("training anchor Region",anchor)
    anchorX = anchor[0]
    anchorY = anchor[1]
    anchorPoints = None
    ids = []
    #are there points?
    for z in range(heights[anchor][0],heights[anchor][1]):
        points_ = loadPoints(anchorX,anchorY,z)
        if (anchorX,anchorY,z) in register.registeredCellBlocks:
            ids+=register.registeredCellBlocks[(anchorX,anchorY,z)]
        if points_ is not None:
            try:
                anchorPoints = np.concatenate([anchorPoints,points_],0)
            except:
                anchorPoints = points_
    try: 
        if anchorPoints is None: 
            print("no points",anchor)
            continue
    except: pass
    anchorPoints = torch.tensor(anchorPoints).float().cuda()
    if len(ids) > 0:
        ids = np.unique(np.array(ids)).tolist()
        for i in ids:
            reloadNeuralVolume(i, gridparameter)
    else:
        print("no previous volumes")
        for z in range(heights[anchor][0],heights[anchor][1]):
                spawnNewVolumes(anchorX,anchorY,z)
    print("first training")
    if debugSingle:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(anchorPoints.cpu().numpy()[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(anchorPoints.cpu().numpy()[:,3:])
        volcloud = []
        reccloud = []
        for v in NeuralBound.activeSet:
            v = NeuralBound.neuralBoundList[v]
            volcloud.append(v.show())
            reccloud.append(v.neuralReconstruction.show(v.center.detach()))
        o3d.visualization.draw_geometries(volcloud+[pcd])
        o3d.visualization.draw_geometries(reccloud+[pcd])
        o3d.visualization.draw_geometries([volcloud[0],reccloud[0]])
    #train the volumes in this region for a bit
    trainActiveVolumes(100, register, 0, anchor, anchorPoints) #active ids can change
    #try to estimate if the region is trained fully
    regionShouldTrain = True
    #training loop
    it = 0
    
    while regionShouldTrain:
        it +=1
        mask = torch.zeros_like(anchorPoints[:,0],dtype=bool).cuda()
        for ida in list(NeuralBound.activeSet):
            v = NeuralBound.neuralBoundList[ida]
            if v is None: 
                continue
            mask = v.filterPoints(anchorPoints[:,:3],mask)
        #see how many points are free in this anchorregion
        if (~mask).sum() < 0.05*len(anchorPoints): #this region is trained
            regionShouldTrain = False
            NeuralBound.deactivateAll()
        else:
            for z in range(heights[anchor][0],heights[anchor][1]):
                spawnNewVolumes(anchorX,anchorY,z)
                #print("eh")
            #train the volumes in this region for a bit
            trainActiveVolumes(50, register, it, anchor, anchorPoints) #active ids can change
            if debugSingle:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(anchorPoints.cpu().numpy()[:,:3])
                pcd.colors = o3d.utility.Vector3dVector(anchorPoints.cpu().numpy()[:,3:])
                volcloud = []
                reccloud = []
                for v in NeuralBound.activeSet:
                    v = NeuralBound.neuralBoundList[v]
                    volcloud.append(v.show())
                    reccloud.append(v.neuralReconstruction.show(v.center.detach()))
                o3d.visualization.draw_geometries(volcloud+[pcd])
                o3d.visualization.draw_geometries(reccloud+[pcd])
                o3d.visualization.draw_geometries([volcloud[0],reccloud[0]])
        if it > 15:
           regionShouldTrain = False
           NeuralBound.deactivateAll()
    
