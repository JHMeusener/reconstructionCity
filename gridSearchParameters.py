import torch
import linAlgHelper
from scipy.spatial import ConvexHull, HalfspaceIntersection
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np


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
                variableFaktoren = [2,2,2,3,3,3,5.,1.,2.]):
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
                                                              (torch.rand(100+100*int(size**2),3).cuda()-0.5)*2*size+self.center],0)
    
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
        #create empty points
        emptypoints = torch.cat((surfacepoints+emptyVectors*0.1,
                                 surfacepoints+emptyVectors*0.2,
                                 surfacepoints+emptyVectors*0.4,
                                 surfacepoints+emptyVectors*size*0.3),0)        
        self.clampBounds()
        self.centerOptim.zero_grad()
        self.boundsOptim.zero_grad()
        centeredPoints_surface = surfacepoints-self.center
        centeredPoints_empty = emptypoints-self.center
        if len(NeuralBound.pointsVolumeOverlapVector) > 0:
            centeredPoints_doubleOcc = NeuralBound.pointsVolumeOverlapVector - self.center
        boundsTest_surface = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_surface[None,:,:], self.bounds[None,:,:])[0]
        boundsTest_empty = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_empty[None,:,:], self.bounds[None,:,:])[0]
        if len(NeuralBound.pointsVolumeOverlapVector) > 0:
            boundsTest_doubleOcc = linAlgHelper.getPointDistances2PlaneNormal(centeredPoints_doubleOcc[None,:,:], self.bounds[None,:,:])[0]
        with torch.no_grad():
            near_surface = boundsTest_surface>-size*self.variableFaktoren[0]   #self.variableFaktoren[2] = 1.0
            completeNear_surface = near_surface.sum(dim=1)==near_surface.shape[1]
            near_empty = boundsTest_empty>-size*self.variableFaktoren[1]   #self.variableFaktoren[2] = 1.0
            completeNear_empty = near_empty.sum(dim=1)==near_empty.shape[1]
            if len(NeuralBound.pointsVolumeOverlapVector) > 0:
                near_doubleOcc = boundsTest_doubleOcc>-size*self.variableFaktoren[2]   #self.variableFaktoren[2] = 1.0
                completeNear_doubleOcc = near_doubleOcc.sum(dim=1)==near_doubleOcc.shape[1]
        insideGradient_empty = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_empty/(size*self.variableFaktoren[3])),0.001)
        outsideGradient_surface = torch.nn.functional.leaky_relu(torch.tanh(-boundsTest_surface/(size*self.variableFaktoren[4])),0.001)
        if len(NeuralBound.pointsVolumeOverlapVector) > 0:
            insideGradient_doubleOcc = torch.nn.functional.leaky_relu(torch.tanh(boundsTest_doubleOcc/(size*self.variableFaktoren[5])),0.001)
            overLapFactor =completeNear_doubleOcc[:,None]  #smaller negative value to encourage seamless reconstructions
            overlapLoss = insideGradient_doubleOcc* overLapFactor
        missedPointFactor = (completeNear_surface*NeuralBound.unoccupiedRegions)[:,None]
        missedPointsLoss = outsideGradient_surface*missedPointFactor#*importance_surface**self.variableFaktoren[9]
        # for the inner loss: errors near a plane are important, but the gradient of the error should shrink, the nearer it gets
        innerEmptyFactor = completeNear_empty[:,None]
        innerEmptyLoss= insideGradient_empty*innerEmptyFactor
        if len(NeuralBound.pointsVolumeOverlapVector) > 0:
            error = self.variableFaktoren[6]*overlapLoss.sum() + self.variableFaktoren[7]*missedPointsLoss.sum()+ self.variableFaktoren[8]*innerEmptyLoss.sum() 
        else:
            error = self.variableFaktoren[7]*missedPointsLoss.sum()+ self.variableFaktoren[8]*innerEmptyLoss.sum() 
        #print("overlap: ",overlapLoss.detach().sum().item(), " missedPoints: ", missedPointsLoss.detach().sum().item(), " innerEmpty: ", innerEmptyLoss.detach().sum().item())
        error.backward()
        self.centerOptim.step()
        self.boundsOptim.step()
        self.clampBounds()
        self.centerOptim.zero_grad()
        self.boundsOptim.zero_grad()
        return {"overlap":overlapLoss.detach().sum().item() if len(NeuralBound.pointsVolumeOverlapVector) > 0  else 0.,
                "missedPoints": missedPointsLoss.detach().sum().item(),
                "inside Empty": innerEmptyLoss.detach().sum().item()               
                }
    
    
    def train(points, cameraPosition):
        '''surface points in (n,3) and cameraposition in (3)'''
        points = points.cuda()
        loss = {"overlap":0.,
                "missedPoints": 0.,
                "inside Empty": 0.}
        NeuralBound.createOccupationVector(points)
        emptyVectors = (cameraPosition[None,:].cuda()-points).reshape(-1,3)
        emptyVectors = emptyVectors/((emptyVectors**2).sum(dim=-1))[:,None]**0.5
        for volume in NeuralBound.neuralBoundList:
            tempLoss = volume.boundsAdjustmentStep(points,emptyVectors)
            loss["overlap"] += tempLoss["overlap"]
            loss["missedPoints"] += tempLoss["missedPoints"]
            loss["inside Empty"] += tempLoss["inside Empty"]
        return loss
    
    def show(self):
        size = self.bounds[:6].detach().abs().max()
        data = (torch.rand(10000,3).cuda()-0.5)*2.*size
        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(data[None,:,:], self.bounds.detach()[None,:,:])[0]
        inside = boundsTest>0
        inside = inside.sum(dim=1)==inside.shape[1]
        filtered = data[inside].cpu()
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

        

        
pointsData = torch.load("/files/Dataset/convexNeuralVolumeData/points.pt")
camPositions = torch.load("/files/Dataset/convexNeuralVolumeData/camPositions.pt")


volumesNr = 6

volumeStarts = pointsData[0][:volumesNr]
emptyvec = camPositions[0][None,:]-volumeStarts
emptyvec = emptyvec/(emptyvec**2).sum(dim=-1)[:,None]**0.5

trainNr = 0

gridparameter = []   # 0: nearSurfaceFaktor, 
                    # 1: nearEmptyFaktor, 
                    # 2: nearDoubleOccFaktor, 
                    # 3: insideEmptySigmoidFaktor, 
                    # 4: missedPointLossSigmoidFaktor,
                    # 5: volumeOverlapSigmoidFaktor, 
                    # 6: overLapFactor
                    # 7: missedPointFaktor,
                    # 8:innerEmptyFaktor,
                    # 9: LR center
                    # 10: LR bounds
nearSurfaceFaktor = [0.3,0.75,1.5,3.]
nearEmptyFaktor = [-0.1,0.,0.1]
nearDoubleOccFaktor = [-0.1,0.]
insideEmptySigmoidFaktor = [0.5,1.,2.]
missedPointLossSigmoidFaktor = [0.5,1.,2.]
volumeOverlapSigmoidFaktor = [0.5,1.,2.]
overLapFactorChoice = [1,5]
missedPointFaktor = [1,5,15,30]
innerEmptyFaktor = [1,5,15,30,100]
LR_center = [0.001,0.01,0.1]
LR_bounds = [0.001,0.01,0.1]

#create search:
for i in range(1000):
    gridparameter=torch.tensor([np.random.choice(nearSurfaceFaktor,1),
                    np.random.choice(nearEmptyFaktor,1),
                    np.random.choice(nearDoubleOccFaktor,1),
                    np.random.choice(insideEmptySigmoidFaktor,1),
                    np.random.choice(missedPointLossSigmoidFaktor,1),
                    np.random.choice(volumeOverlapSigmoidFaktor,1),
                    np.random.choice(overLapFactorChoice,1),
                    np.random.choice(missedPointFaktor,1),
                    np.random.choice(innerEmptyFaktor,1),
                    np.random.choice(LR_center,1),
                    np.random.choice(LR_bounds,1)])[:,0]
    bounds = []
    for i in range(volumesNr):
        #training des neural Volumes
        bounds.append(NeuralBound(boundsize = 0.15,
                        center=torch.tensor(volumeStarts[i]-0.05*emptyvec[i]).cuda(), 
                        verbose=False,
                        centerLR = gridparameter[-2],
                        boundsLR = gridparameter[-1],
                        variableFaktoren= torch.tensor(gridparameter[:-2]).cuda()))

    print("beginning Training Nr: ",trainNr)
    for iter_idx in range(1000):
        points = pointsData[iter_idx].cuda()
        camPosition = camPositions[iter_idx].cuda()
        newLoss = NeuralBound.train(points, camPosition)
        if iter_idx %250 == 249:
            print("     ",iter_idx," of 1000")
            boundEnd = []
            centerEnd = []
            for v in bounds:
                boundEnd.append(v.bounds.detach().cpu())
                centerEnd.append(v.center.detach().cpu())
            torch.save(boundEnd, "/files/Code/convexNeuralVolumeResults/bounds_run_{}_iter{}.pt".format(trainNr,iter_idx))
            torch.save(centerEnd, "/files/Code/convexNeuralVolumeResults/centers_run_{}_iter{}.pt".format(trainNr,iter_idx))

    print("training done")
    torch.save(gridparameter,"/files/Code/convexNeuralVolumeResults/gridParameter_{}.pt".format(trainNr))
    trainNr +=1
