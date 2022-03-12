import torch
import linAlgHelper
from scipy.spatial import ConvexHull, HalfspaceIntersection
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
from neuralVolumeHelper import randCam, matrixLookat, createInputVector_planeHitModel,HiddenPrints, SIREN, getView, circular2sinCosC,bound2Mesh, compare2CenteredModels, bound2Pointcloud, meshIt, modelCenterCorrection, getPredictionPoints,compare2CenteredModels, bound2bounds, meshBoundsTM, mesh2pointcloud, array2Pointcloud
import open3d as o3d
import trimesh as tm



ball = tm.primitives.Capsule(radius=1., height=0.,sections=128)

        
pointsData = torch.load("points.pt")
camPositions = torch.load("camPositions.pt")
pointcloudTarget = torch.cat(pointsData[:16],0).cpu()
mask = np.random.choice(torch.arange(len(pointcloudTarget)),10000)
pointcloudT = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudTarget[mask]))
colors = np.ones_like(pointcloudTarget[mask]).astype(np.float64)
colors[:,2] = colors[:,2]*0.
colors[:,0] = colors[:,0]*0.
pointcloudT.colors = o3d.utility.Vector3dVector(colors)



class NeuralBound:
    pointDoubleOccupationVector = torch.Tensor([[]])
    neuralBoundList = []
    unoccupiedRegions = torch.Tensor([[]])
    pointsVolumeOverlapVector = torch.Tensor([[0.,0.,0.]]).cuda()
    pointsVolumeOverlapVectorDoubleOccupation = torch.Tensor([[0.,0.,0.]]).cuda()
    
    def createOccupationVector(points):
        NeuralBound.pointDoubleOccupationVector = torch.zeros_like(points[:,0]).cuda()
        NeuralBound.pointsVolumeOverlapVector = torch.Tensor([[0.,0.,0.]]).cuda()
        NeuralBound.pointsVolumeOverlapVectorDoubleOccupation = torch.Tensor([[0.,0.,0.]]).cuda()
        for volume in NeuralBound.neuralBoundList:
            volume.insideOccupationCheck1(points)
        NeuralBound.pointsVolumeOverlapVectorDoubleOccupation = torch.zeros_like(NeuralBound.pointsVolumeOverlapVector[:,0])
        for volume in NeuralBound.neuralBoundList:
            volume.insideOccupationCheck2()
        NeuralBound.unoccupiedRegions = NeuralBound.pointDoubleOccupationVector == 0
        NeuralBound.pointsVolumeOverlapVector = NeuralBound.pointsVolumeOverlapVector[NeuralBound.pointsVolumeOverlapVectorDoubleOccupation > 1]
        
    def __init__(self, 
                 additionalBounds = torch.Tensor([[0.75,0.75,0.75],
                                                    [-0.75,0.75,0.75],
                                                    [0.75,-0.75,0.75],
                                                    [0.75,0.75,-0.75],
                                                    [-0.75,-0.75,0.75],
                                                    [0.75,-0.75,-0.75],
                                                    [-0.75,0.75,-0.75],
                                                    [-0.75,-0.75,-0.75]]).cuda(),
                 boundsize = 0.3,
                 center=torch.Tensor([[0.,0.,0.]]), 
                 verbose=True,
                centerLR = 0.001,
                boundsLR = 0.01,
                variableFaktoren = [2,2,2,3,3,3,5.,1.,2.],
                maxPointsInput = 500000,
                maxPointsChunk = 10000):
        self.center = center.cuda()
        self.bounds = torch.Tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]).cuda()
        if additionalBounds is not None:
            self.bounds = torch.cat((self.bounds,additionalBounds),0)*boundsize
        self.bounds = self.bounds.cuda()
        self.name = "newBound"
        self.verbose = verbose
        self.center.requires_grad = True
        self.bounds.requires_grad = True
        self.centerOptim = torch.optim.Adam([self.center], lr=centerLR)
        self.boundsOptim = torch.optim.Adam([self.bounds], lr=boundsLR)
        self.variableFaktoren = variableFaktoren
        #self.neuralReconstruction = neuralConvexReconstruction(self.center)
        self.maxpointsInput = maxPointsInput
        self.maxpointsChunk = maxPointsChunk
        NeuralBound.neuralBoundList.append(self)
        if self.verbose:
            print("{} was created at {}".format(self.name, self.center.cpu().detach()))
            tmesh = meshBoundsTM(self.bounds.detach().cpu())
            self.pCloud = mesh2pointcloud(tmesh, self.center.detach().cpu(), color=[1,0,0])
            cCloud = array2Pointcloud(self.center.detach().cpu(), center = torch.tensor([[0,0,0]]), color=[1,0.0,1.0])
            o3d.visualization.draw_geometries( [pointcloudT]+[self.pCloud,cCloud])
    
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
        
    def insideOccupationCheck1(self, points):
        '''Takes only surfacepoints (n,3)'''
        with torch.no_grad():
            centeredPoints = points-self.center
            boundsTest = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints[None,:,:], self.bounds[None,:,:])[0]
            inside = boundsTest>0
            completeInner = inside.sum(dim=1)==inside.shape[1]
            NeuralBound.pointDoubleOccupationVector += completeInner*1
            size = self.bounds[:6].detach().abs().max()
            NeuralBound.pointsVolumeOverlapVector = torch.cat([NeuralBound.pointsVolumeOverlapVector,
                                                              (torch.rand(3000+3000*int(size**2),3).cuda()-0.5)*2*size+self.center],0)
    
    def insideOccupationCheck2(self):
        with torch.no_grad():
            centeredPoints = NeuralBound.pointsVolumeOverlapVector-self.center
            boundsTest = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints[None,:,:], self.bounds[None,:,:])[0]
            inside = boundsTest>0
            completeInner = inside.sum(dim=1)==inside.shape[1]
            NeuralBound.pointsVolumeOverlapVectorDoubleOccupation += completeInner*1

    
    def boundsAdjustmentStep(self, surfacepoints, emptyVectors):
        '''gets tensor(n,3) surfacespoints (surface ) with (n,1) values (1 for surface, -1 for empty)'''
        size = self.bounds[:6].detach().abs().max()
        if len(surfacepoints) > self.maxpointsInput:
            surfacepoints = surfacepoints.cpu()
            emptyVectors = emptyVectors.cpu()
        missedPointsLoss = torch.tensor(0)
        innerEmptyLoss = torch.tensor(0)
        if self.verbose:
            newtmesh = meshBoundsTM(self.bounds.detach().cpu())
            newpCloud = mesh2pointcloud(newtmesh, self.center.detach().cpu(), color=[1.,0,0])
            pPCloud = array2Pointcloud(surfacepoints.cpu(), center = torch.tensor([[0,0,0]]), color=[0,0.5,0.5])
            cCloud = array2Pointcloud(self.center.detach().cpu(), center = torch.tensor([[0,0,0]]), color=[1,0.0,1.0])
            colors = np.asarray(self.pCloud.colors)
            colors[:,0] = colors[:,0]+0.3
            colors[:,1] = colors[:,1]+0.3
            self.pCloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries( [pointcloudT]+[self.pCloud,pPCloud,newpCloud,cCloud])
            self.pCloud = newpCloud
        #split inputdata
        for endFaktor in range(0,len(surfacepoints)//self.maxpointsInput+1):
            surfacepoints_ = surfacepoints[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput].cuda()
            #test that there are not more valid points than the chunksize
            with torch.no_grad():
                centeredPoints_surface = surfacepoints_-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
                near_surface = boundsTest_surface>-size*self.variableFaktoren[0]
                completeNear_surface = near_surface.sum(dim=1)==near_surface.shape[1]
                del boundsTest_surface,near_surface
            surfacepoints_ = surfacepoints_[completeNear_surface]
            #split surface-point learning
            for chunkFactor in range(len(surfacepoints_)//self.maxpointsChunk+1):
                surfacepointsChunk = surfacepoints_[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                centeredPoints_surface = surfacepointsChunk-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
                outsideGradient_surface = torch.nn.functional.leaky_relu(torch.tanh(-boundsTest_surface*self.variableFaktoren[3]/size),0.001)
                value = (torch.sigmoid(boundsTest_surface.detach()*3/(size))+0.3).prod(dim=1)
                #train neural reconstruction -bounds influence training value
                points = centeredPoints_surface.detach()
                del centeredPoints_surface
                if len(points) > 0:
                    #difference = self.neuralReconstruction.train(points, value)
                    del points
                    #value = 0.1 + 1./(difference.abs()+0.01*size)[:,0]
                    missedPointFactor = (value*NeuralBound.unoccupiedRegions[completeNear_surface][chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk])[:,None]
                    missedPointsLoss = (outsideGradient_surface*missedPointFactor).mean()*self.variableFaktoren[6]
                    del outsideGradient_surface, boundsTest_surface
                    missedPointsLoss.backward()
                    self.centerOptim.step()
                    self.boundsOptim.step()
                    self.clampBounds()
                    self.centerOptim.zero_grad()
                    self.boundsOptim.zero_grad()
                    #del missedPointFactor, value, difference
                    #adjust for empty points      
                    emptyVectors_ = emptyVectors[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput][completeNear_surface][chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk].cuda()
                    #create empty points
                    emptypoints = torch.cat((surfacepointsChunk.detach()+emptyVectors_*size*0.5,
                                            surfacepointsChunk.detach()+emptyVectors_*size*0.9,
                                            surfacepointsChunk.detach()+emptyVectors_*size*1.5,
                                            surfacepointsChunk.detach()+emptyVectors_*size*2.),0)     
                    del surfacepointsChunk
                    if self.verbose:
                        newtmesh = meshBoundsTM(self.bounds.detach().cpu())
                        newpCloud = mesh2pointcloud(newtmesh, self.center.detach().cpu(), color=[1.,0,0])
                        pPCloud = array2Pointcloud(surfacepoints.cpu(), center = torch.tensor([[0,0,0]]), color=[0,0.5,0.5])
                        colors = np.asarray(self.pCloud.colors)
                        colors[:,0] = colors[:,0]+0.3
                        colors[:,1] = colors[:,1]+0.3
                        self.pCloud.colors = o3d.utility.Vector3dVector(colors)
                        cCloud = array2Pointcloud(self.center.detach().cpu(), center = torch.tensor([[0,0,0]]), color=[1,0.0,1.0])
                        pECloud = array2Pointcloud(emptypoints.cpu(), center = torch.tensor([[0,0,0]]), color=[0.5,0.5,0.0])
                        o3d.visualization.draw_geometries( [pointcloudT]+[self.pCloud,pPCloud,pECloud,newpCloud,cCloud])
                        self.pCloud = newpCloud
                        
                    centeredPoints_empty = emptypoints-self.center
                    boundsTest_empty = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_empty[None,:,:], self.bounds[None,:,:])[0]
                    insideGradient_empty = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_empty*self.variableFaktoren[2]/size),0.001)
                    innerEmptyLoss= insideGradient_empty.mean()*self.variableFaktoren[7]
                    innerEmptyLoss.backward()
                    del insideGradient_empty, boundsTest_empty
                    self.centerOptim.step()
                    self.boundsOptim.step()
                    self.clampBounds()
                    self.centerOptim.zero_grad()
                    self.boundsOptim.zero_grad()
                    #train on empty
                    #self.neuralReconstruction.trainEmpty(centeredPoints_empty.detach(), size)
                    del centeredPoints_empty
        try:
            del emptyVectors, emptyVectors_, emptypoints
        except: pass
        del completeNear_surface, surfacepoints, surfacepoints_
        doubleOccLoss = torch.tensor(0)
        for endFaktor in range(0,len(NeuralBound.pointsVolumeOverlapVector)//self.maxpointsInput+1):   
            #test that there are not more valid points than the chunksize
            with torch.no_grad():
                centeredPoints_overlap = NeuralBound.pointsVolumeOverlapVector[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]-self.center
                boundsTest_doubleOcc = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_overlap[None,:,:], self.bounds[None,:,:])[0]
                near_doubleOcc = boundsTest_doubleOcc>-size*self.variableFaktoren[1]   #self.variableFaktoren[2] = 1.0
                completeNear_doubleOcc = near_doubleOcc.sum(dim=1)==near_doubleOcc.shape[1]
            #chunk the relevant points
             #split surface-point learning
            inputPoints = NeuralBound.pointsVolumeOverlapVector[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput][completeNear_doubleOcc]
            for chunkFactor in range(len(inputPoints)//self.maxpointsChunk+1):
                centeredPoints_overlap_chunk = inputPoints[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]-self.center
                boundsTest_doubleOcc = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_overlap_chunk[None,:,:], self.bounds[None,:,:])[0]
                insideGradient_doubleOcc = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_doubleOcc*self.variableFaktoren[4]/size),0.001)
                if len(insideGradient_doubleOcc) > 0:
                    doubleOccLoss = (self.variableFaktoren[5]*insideGradient_doubleOcc.mean())
                    doubleOccLoss.backward()
                    self.centerOptim.step()
                    self.boundsOptim.step()
                self.clampBounds()
                self.centerOptim.zero_grad()
                self.boundsOptim.zero_grad()
        bound2nnLoss = 0#self.adjustBounds2NeuralNet()
        nn2boundLoss = 0#self.adjustNeuralNet2Bound()
        return {"overlap":doubleOccLoss.detach().item(),
                "missedPoints": missedPointsLoss.detach().item(),
                "inside Empty": innerEmptyLoss.detach().item(),
                "bound2nnLoss": bound2nnLoss,
                "nn2boundLoss":nn2boundLoss}
    
    
    def train(points, cameraPosition):
        '''surface points in (n,3) and cameraposition in (3)'''
        points = points.cuda()
        loss = {"overlap":0.,
                "missedPoints": 0.,
                "inside Empty": 0.,
                "bound2nnLoss": 0.,
                "nn2boundLoss":0.}
        NeuralBound.createOccupationVector(points)
        emptyVectors = (cameraPosition[None,:].cuda()-points).reshape(-1,3)
        emptyVectors = emptyVectors/((emptyVectors**2).sum(dim=-1))[:,None]**0.5
        for volume in NeuralBound.neuralBoundList:
            tempLoss = volume.boundsAdjustmentStep(points,emptyVectors)
            loss["overlap"] += tempLoss["overlap"]
            loss["missedPoints"] += tempLoss["missedPoints"]
            loss["inside Empty"] += tempLoss["inside Empty"]
            loss["bound2nnLoss"] +=  tempLoss["bound2nnLoss"]
            loss["nn2boundLoss"] +=  tempLoss["nn2boundLoss"]
        return loss
    
    def show(self):
        size = self.bounds[:6].detach().abs().max()
        data = (torch.rand(10000,3).cuda()-0.5)*2.*size
        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(data[None,:,:], self.bounds.detach()[None,:,:])[0]
        inside = boundsTest>0
        inside = inside.sum(dim=1)==inside.shape[1]
        filtered = torch.cat([data[inside].cpu(),self.bounds.detach().cpu()],0)
        filteredIdx = torch.arange(len(filtered))
        hull = ConvexHull(filtered)
        verts_ = torch.tensor(hull.vertices)
        vertIdx = torch.arange(len(verts_))
        filteredIdx[verts_.long()] = vertIdx
        faces_ = torch.tensor(hull.simplices)
        vertices, faces =  filtered[verts_.long()]+self.center.detach().cpu(), filteredIdx[faces_.long()]
        mesh = tm.Trimesh(vertices=vertices, faces=faces)
        pointcloudPoints = mesh.sample(2000)
        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
        colors = np.ones_like(pointcloudPoints).astype(np.float64)
        colors[:,2] = colors[:,2]*np.random.rand()
        colors[:,1] = colors[:,1]*np.random.rand()
        colors[:,0] = colors[:,0]*np.random.rand()
        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
        return(pointcloudMesh)




volumesNr = 1

volumeStarts = pointsData[0][:volumesNr]
emptyvec = camPositions[0][None,:]-volumeStarts
emptyvec = emptyvec/(emptyvec**2).sum(dim=-1)[:,None]**0.5

trainNr = 0

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
nearSurfaceFaktor = [0.8]
nearDoubleOccFaktor = [0.]
insideEmptySigmoidFaktor = [3.]
missedPointLossSigmoidFaktor = [0.5]
volumeOverlapSigmoidFaktor = [3.]
overLapFactorChoice = [1]
missedPointFaktor = [15]
innerEmptyFaktor = [1]
LR_center = [0.0003]
LR_bounds = [0.0006]

#create search:
for i in range(1):
    gridparameter=torch.tensor([np.random.choice(nearSurfaceFaktor,1),
                    np.random.choice(nearDoubleOccFaktor,1),
                    np.random.choice(insideEmptySigmoidFaktor,1),
                    np.random.choice(missedPointLossSigmoidFaktor,1),
                    np.random.choice(volumeOverlapSigmoidFaktor,1),
                    np.random.choice(overLapFactorChoice,1),
                    np.random.choice(missedPointFaktor,1),
                    np.random.choice(innerEmptyFaktor,1),
                    np.random.choice(LR_center,1),
                    np.random.choice(LR_bounds,1)])[:,0]
    NeuralBound.neuralBoundList = []
    bounds = []
    for i in range(volumesNr):
        #training des neural Volumes
        bounds.append(NeuralBound(boundsize = 0.05,
                        center=torch.tensor(volumeStarts[i]-0.05*emptyvec[i]).cuda(), 
                        verbose=True,
                        centerLR = gridparameter[-2],
                        boundsLR = gridparameter[-1],
                        variableFaktoren= torch.tensor(gridparameter[:-2]).cuda()))


path = []

import gc 
gc.collect()

loss = {"overlap":0.,
                "missedPoints": 0.,
                "inside Empty": 0.,
                "bound2nnLoss": 0.,
                "nn2boundLoss":0.}
first = True

if True:
    trainmax = 1500
    print("beginning Training Nr: ",trainNr)
    for iter_idx in range(trainmax):
        points = pointsData[iter_idx].cuda()
        camPosition = camPositions[iter_idx].cuda()
        newLoss = NeuralBound.train(points, camPosition)
        if first:
            loss["overlap"] = newLoss["overlap"]
            loss["missedPoints"] = newLoss["missedPoints"]
            loss["inside Empty"] = newLoss["inside Empty"]
            loss["bound2nnLoss"] =  newLoss["bound2nnLoss"]
            loss["nn2boundLoss"] =  newLoss["nn2boundLoss"]
            first = False
        else:
            loss["overlap"] = 0.05*newLoss["overlap"]+0.95*loss["overlap"]
            loss["missedPoints"] = 0.05*newLoss["missedPoints"]+0.95*loss["missedPoints"]
            loss["inside Empty"] = 0.05*newLoss["inside Empty"]+0.95*loss["inside Empty"]
            loss["bound2nnLoss"] =  0.05*newLoss["bound2nnLoss"]+0.95*loss["bound2nnLoss"]
            loss["nn2boundLoss"] =  0.05*newLoss["nn2boundLoss"]+0.95*loss["nn2boundLoss"]
        if iter_idx %50 == 0:
            path.append(NeuralBound.neuralBoundList[0].show())
            print("Dies ist Iteration ",iter_idx," of ",trainmax)
            print(newLoss)


pointClouds = []
for v in NeuralBound.neuralBoundList:
    try:
        pointClouds.append(v.show())
    except:
        pass

o3d.visualization.draw_geometries( [pointcloudT]+pointClouds)#+path)
o3d.visualization.draw_geometries( [pointcloudT]+path)