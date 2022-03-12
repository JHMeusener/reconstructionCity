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

class NeuralConvexReconstruction:
    def __init__(self, center):
        self.learnModel = SIREN([12,64, 64], lastlayer=False).cuda()
        self.learnModelLastLayer = SIREN([64], lastlayer=True).cuda()
        self.uncertaintyModel = torch.nn.Sequential(nn.Linear(12,256),nn.ReLU(),nn.Linear(256,67)).cuda()
        self.optimLearn = torch.optim.Adam(self.learnModel.parameters(), lr=0.001)
        self.optimLast = torch.optim.Adam(self.learnModelLastLayer.parameters(), lr=0.001)
        self.center = center
        self.center.requires_grad=True
        self.optimCenter = torch.optim.Adam([self.center], lr=0.005)
        self.lastLoss = 1.
    
    def train(self, centered_points, value):
        '''points are the n,3 karthesian coordinate points.  value is the certainty, that the point belongs to the convex part. It will have an effekt on the loss of the network.
            The value is initially derived from the differenciation inside-the bounds, outside (near) the bounds'''
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimCenter.zero_grad()
        prediction, difference, prediction1 = self.predict(centered_points)
        loss = torch.nn.functional.leaky_relu(difference*value, negative_slope=0.3).abs().mean()
        #Regularize Model center to Prediction center
        centerCorrection = modelCenterCorrection(self.learnModel,self.learnModelLastLayer)
        centerError = torch.nn.functional.l1_loss(self.center,centerCorrection+self.center.detach())
        loss += centerError
        lossFactor = self.lastLoss
        self.lastLoss = loss.detach()
        #regularisation is only possible if there are points from all "sections" - so we have to imagine a few for this
        l2_lambda = 0.001
        l2_regularizer = sum(p.pow(2.0).sum()
                        for p in self.learnModel.parameters())
        l2_regularizer += sum(p.pow(2.0).sum()
                        for p in self.learnModelLastLayer.parameters())
        loss = (loss + l2_lambda * l2_regularizer)*lossFactor
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        self.optimCenter.step()
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimCenter.zero_grad()
        return self.lastLoss
    
    def trainEmpty(self, empty_centered_points, size):
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        prediction, difference, prediction1 = self.predict(empty_centered_points)
        loss = torch.nn.functional.relu(-difference/size).abs().sum()
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        return difference.detach()
        
    def predict(self,centered_points):
            sphericalInput = linAlgHelper.asSpherical(centered_points)
            circularIn = circular2sinCosC(sphericalInput[:,:2].float())
            prediction1 = self.learnModel(circularIn)
            prediction = self.learnModelLastLayer(prediction1)
            difference = prediction-sphericalInput[:,2][:,None]
            return prediction, difference, prediction1

    def predictSphere(self):
        inputPoints = ball.sample(10000)
        sphericalInput = linAlgHelper.asSpherical(torch.tensor(inputPoints).float())[:,:2].cuda()
        with torch.no_grad():
            prediction, difference, prediction1 = self.predict(torch.tensor(inputPoints).float().cuda())
            predictedSpherical = torch.cat((sphericalInput,abs(prediction)),dim=1)
            points = (linAlgHelper.asCartesian(predictedSpherical))
        return points, predictedSpherical
        
    def show(self, color = None):
        if color is None:
            color = np.random.rand(3)
        with torch.no_grad():
            points,_ = self.predictSphere()
            points = (points+self.center.detach()).cpu()
            pointcloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
            colors = np.ones_like(points).astype(np.float64)
            colors[:,2] = colors[:,2]*color[2]
            colors[:,1] = colors[:,1]*color[1]
            colors[:,0] = colors[:,1]*color[0]
            pointcloud.colors = o3d.utility.Vector3dVector(colors)
            return pointcloud

neuRec = NeuralConvexReconstruction(torch.tensor([0.,0.,0.]).cuda())
target = tm.load("hotuce.OBJ")

path = []

#train
smoothLoss = 0.
first = True

for i in range(200):
    tpoints = torch.tensor(target.sample(5000))
    loss = abs(neuRec.train(tpoints.float().cuda()-neuRec.center, torch.ones(5000).cuda()))
    if first:
        smoothLoss = loss
        first = False
    smoothLoss = smoothLoss*0.95 + loss.item()*0.05
    if i%10 == 0:
        path.append(neuRec.show())
        print(smoothLoss)
        print(neuRec.center.detach())



pointcloudTarget = target.sample(5000)
pointcloudT = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudTarget))
colors = np.ones_like(pointcloudTarget).astype(np.float64)
colors[:,0] = colors[:,0]*0.
colors[:,1] = colors[:,1]*0.
pointcloudT.colors = o3d.utility.Vector3dVector(colors)



o3d.visualization.draw_geometries( [pointcloudT]+path)#+path)

o3d.visualization.draw_geometries( [pointcloudT]+[path[-1]])#+path)
o3d.visualization.draw_geometries( [path[-1]])#+path)