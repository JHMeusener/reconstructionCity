# %%
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
from PointCloudNeuralNetDist import NeuralConvexReconstruction
from pathlib import Path
import os

pfad = "/home/jhm/Desktop/Arbeit/ConvexNeuralVolume"

ball = tm.primitives.Capsule(radius=1., height=0.,sections=128)

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
    pointDoubleOccupationVector = torch.Tensor([[]])
    neuralBoundList = []
    unoccupiedRegions = torch.Tensor([[]])
    pointsVolumeOverlapVector = torch.Tensor([[0.,0.,0.]]).cuda()
    pointsVolumeOverlapVectorDoubleOccupation = torch.Tensor([[0.,0.,0.]]).cuda()
    
    def __init__(self, 
                 bounds,
                 center=torch.Tensor([[0.,0.,0.]]), 
                centerLR = 0.001,
                boundsLR = 0.01,
                variableFaktoren = [2,2,2,3,3,3,5.,1.,2.],
                maxPointsInput = 500000,
                maxPointsChunk = 10000):
        '''bounds should start with: [1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]'''
        self.center = center.cuda()
        self.noOverlap = True
        self.centerBackup = center
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
    
    def boundsAdjustmentStep(self, surfacepoints, overlappoints):
        '''gets tensor(n,3) surfacespoints (surface ) with (n,1) values (1 for surface, -1 for empty)'''
        size = self.bounds[:6].detach().abs().max()
        try:
            if surfacepoints is None:
                return {"overlap":0.,
                "missedPoints": 0.,
                "flippotential": 0.,
                "innersurfaceLoss": 0.}, False
        except: pass
        try: 
            if overlappoints is None:
                overlappoints = []
        except: pass
        if len(surfacepoints) > self.maxpointsInput:
            surfacepoints = surfacepoints.cpu()
        if len(overlappoints) > self.maxpointsInput:
            overlappoints = overlappoints.cpu()
        missedPointsLoss_all = torch.tensor(0.)
        innerSurfaceLoss_all = torch.tensor(0.)
        doubleOccLoss_all = torch.tensor(0.)
        flipPotential_all = torch.ones((len(self.bounds)))
        flipPotential = torch.ones((len(self.bounds)))
        allPointsNr = 1
        killme = True
        allSurfaceWeight = 0
        #split inputdata
        for endFaktor in range(0,len(surfacepoints)//self.maxpointsInput+1):
            surfacepoints_ = torch.tensor(surfacepoints[endFaktor*self.maxpointsInput:(endFaktor+1)*self.maxpointsInput]).float().cuda()
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
            inMask = completeIn_surface[completeNear_surface]
            #split surface-point learning
            for chunkFactor in range(len(surfacepoints_)//self.maxpointsChunk+1):
                surfacepointsChunk = surfacepoints_[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                inMaskChunk = inMask[chunkFactor*self.maxpointsChunk:(chunkFactor+1)*self.maxpointsChunk]
                centeredPoints_surface = surfacepointsChunk-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
                value = (torch.sigmoid(boundsTest_surface.detach()*3/(size))+0.3).prod(dim=1)
                innerSurfaceLoss = torch.tensor(0.)
                if len(boundsTest_surface > 0):
                    flipPotential = torch.zeros_like(boundsTest_surface[0,:])
                    if inMaskChunk.sum() > 0:
                        killme = False
                    if inMaskChunk.sum() > 10:
                        innerSurfaceLoss = (boundsTest_surface[inMaskChunk]/torch.linalg.norm(self.bounds,ord=2,dim=1)).min(axis=1)[0].mean()*self.variableFaktoren[2]
                    if inMaskChunk.sum() > 80:
                        flipPotential = boundsTest_surface[inMaskChunk].min(axis=0)[0]/torch.linalg.norm(self.bounds,ord=2,dim=1)
                #train neural reconstruction -bounds influence training value
                points = centeredPoints_surface.detach()
                del centeredPoints_surface
                if len(points) > 0:
                    lastNNLoss, NNDifference = self.neuralReconstruction.train(points.float(), value.float())
                    if lastNNLoss < size*0.1:
                        self.neuralNetTrained = True
                    NNValue = torch.relu(size*0.1-abs(NNDifference))/size*0.1
                    # adjust bounds
                    if self.neuralNetTrained:
                        outsideGradient_surface = torch.nn.functional.leaky_relu(torch.tanh(-boundsTest_surface*self.variableFaktoren[3]/size),0.001)*NNValue
                    else:
                        outsideGradient_surface = torch.nn.functional.leaky_relu(torch.tanh(-boundsTest_surface*self.variableFaktoren[3]/size),0.001)
                    #Regularize Model center to Prediction center
                    with torch.no_grad():
                        self.centerBackup = self.center.detach().clone()
                    centerCorrection = modelCenterCorrection(self.neuralReconstruction.learnModel,self.neuralReconstruction.learnModelLastLayer)
                    centerError = torch.nn.functional.l1_loss(self.center,centerCorrection+self.center.detach())
                    
                    with torch.no_grad():
                        if self.center.isnan().sum() > 0:
                            print("resetting center to ", self.centerBackup)
                            self.center = self.centerBackup.clone()
                            self.center.requires_grad=True
                            self.optimCenter = torch.optim.Adam([self.center], lr=0.005)
                    innerSurfaceLoss_all = (innerSurfaceLoss_all +innerSurfaceLoss.detach()).detach()
                    del points
                    missedPointsLoss = (outsideGradient_surface).mean()*self.variableFaktoren[6] +((flipPotential*self.variableFaktoren[7]).mean()+innerSurfaceLoss)*0.02
                    
                    del outsideGradient_surface
                    #add centerCorrection to loss
                    if self.neuralNetTrained:
                        missedPointsLoss = missedPointsLoss+centerError+innerSurfaceLoss
                    missedPointsLoss.backward()
                    missedPointsLoss_all += missedPointsLoss.detach().cpu()
                    self.centerOptim.step()
                    self.boundsOptim.step()
                    self.clampBounds()
                    self.centerOptim.zero_grad()
                    self.boundsOptim.zero_grad()
                    
                    #del missedPointFactor, value, difference
            flipPotential_all = torch.stack((flipPotential.cpu().detach(),flipPotential_all),0).min(axis=0)[0]
        del completeNear_surface, surfacepoints, surfacepoints_    
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
        # flip if necessary
        maxFlip = flipPotential_all.argmax()
        if flipPotential_all[maxFlip] > 1.0:
            print("moved from ",self.center)
            self.center = self.center - flipPotential_all[maxFlip]
            print(" to ",self.center)
        return {"overlap":doubleOccLoss_all.item(),
                "missedPoints": missedPointsLoss_all.item(),
                "flippotential": flipPotential_all.max(),
                "innersurfaceLoss": innerSurfaceLoss_all.item()/allPointsNr}, killme

    def getCellBlocks(self, distance=0.0):
        minX,maxX,minY,maxY,minZ,maxZ = self.getBounds(addDistance=distance)
        cells = set()
        for x in range(int(minX//35),int(maxX//35)+1):
            for y in range(int(minY//35),int(maxY//35)+1):
                for z in range(int(minZ//35),int(maxZ//35)+1):
                    cells.add((x,y,z))
        return cells    

    def filterPoints(self, points, mask):
        with torch.no_grad():
                centeredPoints = points-self.center
                boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints[None,:,:], self.bounds[None,:,:])[0]
                in_surface = boundsTest_surface>0
                completeIn_surface = in_surface.sum(dim=1)==in_surface.shape[1]
                mask[completeIn_surface] = True
        return mask

    
    def train(self, neuralVolumeCellRegister, loss):
        # get the relevant pointcells
        selfCells = self.getCellBlocks()
        #delete own cells from CellRegister
        otherVolumePoints = neuralVolumeCellRegister.popNearbyVolumePoints(selfCells, self.id)
        points = None 
        size = self.bounds[:6].detach().abs().max()
        selfCells = self.getCellBlocks(distance=size)
        for cell in list(selfCells):
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
                killMe = True
                return loss, killMe
        except:
            pass
        if size < 0.04:
            killMe = True
        color = points[:,3:6]
        points = points[:,:3]
        if self.noOverlap:
            for subiter in range(50):
                vLoss, killMe = self.boundsAdjustmentStep(points, otherVolumePoints)
                if self.noOverlap == False:
                    break
        else:
            vLoss, killMe = self.boundsAdjustmentStep(points, otherVolumePoints)
        selfCells = self.getCellBlocks()
        neuralVolumeCellRegister.registerId(selfCells, self.id)
        loss["overlap"] += vLoss["overlap"]
        loss["missedPoints"] += vLoss["missedPoints"]
        loss["innersurfaceLoss"] += vLoss["innersurfaceLoss"]
        loss["flippotential"] += vLoss["flippotential"]
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
nearSurfaceFaktor = [1.2]
nearDoubleOccFaktor = [0.]
insideEmptySigmoidFaktor = [1.]
missedPointLossSigmoidFaktor = [1.]
volumeOverlapSigmoidFaktor = [3.]
overLapFactorChoice = [20.]
missedPointFaktor = [30]
innerEmptyFaktor = [1.]
LR_center = [0.001]
LR_bounds = [0.002]
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

#load and register all neural volumes
'''
volList = os.listdir(pfad+"/neuralVolumes")
for v in volList:
    array = np.load(pfad+"/neuralVolumes/"+v)
    center = array[0]
    Hrep = array[1:]  
    newVolume = NeuralBound(torch.tensor(Hrep).cuda() ,center =torch.tensor(center).cuda(),centerLR = gridparameter[-2],
                        boundsLR = gridparameter[-1],
                        variableFaktoren= torch.tensor(gridparameter[:-2]).cuda())
    cells = newVolume.getCellBlocks()
    register.registerId(cells, newVolume.id)
'''
#for every neural Volume load the points and train the network/bounds
#if the bounds are to small delete the neural volume
#if the near volume contains no points delete the neural volume

x_mid = int(25*5/35)
y_mid = int(341*5/35)
z_mid = int(121*5/35)
ids = []
points = loadPoints(x_mid,y_mid,z_mid)
for x in range(-6+x_mid,6+x_mid):
    for y in range(-6+y_mid,6+y_mid):
        if (x,y) in heights: 
            for z in range(heights[(x,y)][0],heights[(x,y)][1]): #load all z
                points_ = loadPoints(x,y,z)
                if (x,y,z) in register.registeredCellBlocks:
                    ids+=register.registeredCellBlocks[(x,y,z)]
                if points_ is not None:
                    try:
                        points = np.concatenate([points,points_],0)
                    except:
                        points = points_

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(points[:,3:])


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
            mask = v.filterPoints(points,mask)
    startPoints = points[~mask]
    if len(startPoints)>0:
        for i in range(int(len(startPoints)/5000)+1):
            pointNr = np.random.randint(len(startPoints))
            newVolume = NeuralBound(torch.tensor(createBounds(size=4.)).float().cuda() ,center =startPoints[pointNr].float().cuda(),centerLR = gridparameter[-2],
                        boundsLR = gridparameter[-1],
                        variableFaktoren= torch.tensor(gridparameter[:-2]).cuda())
            cells = newVolume.getCellBlocks()
            register.registerId(cells, newVolume.id)

#load random cells
for x in range(-6+x_mid,6+x_mid):
    for y in range(-6+y_mid,6+y_mid):
        if (x,y) in heights: 
            for z in range(heights[(x,y)][0],heights[(x,y)][1]):
                if torch.rand(1).item()<0.07:
                    spawnNewVolumes(x,y,z)

volcloudOld = []
for v in NeuralBound.neuralBoundList:
    volcloudOld.append(v.show())

for iteration in range(1000):
    killList = []
    loss = {}
    loss["overlap"] = torch.tensor(0.)
    loss["missedPoints"] = torch.tensor(0.)
    loss["innersurfaceLoss"] = torch.tensor(0.)
    loss["flippotential"] = torch.tensor(0.)
    loss["volumes"] = torch.tensor(0)

    for i, volume in enumerate(NeuralBound.neuralBoundList):
        if volume is None:
            continue
        '''volcloud = []
        for v in NeuralBound.neuralBoundList:
            if v is None:
                continue
            volcloud.append(v.show())
        o3d.visualization.draw_geometries(volcloud+[pcd])
        if volume is None:
            continue
        o3d.visualization.draw_geometries([volume.show()]+[pcd])'''
        print("training volume ",i," of ",len(NeuralBound.neuralBoundList), "iteration: ", iteration)
        loss, killme = volume.train(register, loss)
        if killme:
            killList.append(i)

    for i in killList:
        '''volcloud = []
        for v in NeuralBound.neuralBoundList:
            if v is None:
                continue
            volcloud.append(v.show())
        o3d.visualization.draw_geometries([NeuralBound.neuralBoundList[i].show()]+[pcd])
        o3d.visualization.draw_geometries(volcloud+[pcd])'''
        cells = NeuralBound.neuralBoundList[i].getCellBlocks()
        try:
            register.popNearbyVolumePoints(cells, i) 
        except: pass
        NeuralBound.neuralBoundList[i] = None
    if iteration%50 == 49:
        for x in range(-6+x_mid,6+x_mid):
            for y in range(-6+y_mid,6+y_mid):
                if (x,y) in heights: 
                    for z in range(heights[(x,y)][0],heights[(x,y)][1]):
                        if torch.rand(1).item()<0.07:
                            spawnNewVolumes(x,y,z)
    print(loss)
    for i, volume in enumerate(NeuralBound.neuralBoundList):
        if volume is None:
            try:
                os.remove(pfad+"/saveNeuralNetwork/{}.npz".format(i))
            except: pass
            continue
        else:
            np.save(pfad+"/saveNeuralNetwork/{}.npz".format(i), torch.cat((volume.bounds,volume.center[None,:]),0).detach().cpu().numpy())
        



volcloud = []
for v in NeuralBound.neuralBoundList:
    if v is None:
        continue
    volcloud.append(v.show())
o3d.visualization.draw_geometries(volcloud+[pcd])
# %%

#for a region: if there are points which are not covered by a neural volume 
#   create a new neural volume
#(do this every few iterations)

# %%
