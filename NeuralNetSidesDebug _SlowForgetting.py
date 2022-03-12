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
referenceUnusedList = torch.stack((torch.arange(20).repeat_interleave(10),torch.arange(10).repeat(20)),1).cuda()

class NeuralConvexReconstruction:
    def __init__(self, center):
        self.learnModel = SIREN([12,64, 64], lastlayer=False).cuda()
        self.learnModelLastLayer = SIREN([64], lastlayer=True).cuda()
        self.learnModel2 = SIREN([12,64, 64], lastlayer=False).cuda()
        self.learnModelLastLayer2 = SIREN([64], lastlayer=True).cuda()
        self.optimLearn = torch.optim.Adam(self.learnModel.parameters(), lr=0.00001)
        self.optimLearnReg = torch.optim.Adam(self.learnModel.parameters(), lr=0.0001)
        self.optimLearnReg2 = torch.optim.Adam(self.learnModel.parameters(), lr=0.0001)
        self.optimLastReg= torch.optim.Adam(self.learnModelLastLayer.parameters(), lr=0.0001)
        self.optimLastReg2 = torch.optim.Adam(self.learnModelLastLayer2.parameters(), lr=0.00002)
        self.optimLearn2 = torch.optim.Adam(self.learnModel2.parameters(), lr=0.000002)
        self.optimLast = torch.optim.Adam(self.learnModelLastLayer.parameters(), lr=0.0001)
        self.optimLast2 = torch.optim.Adam(self.learnModelLastLayer2.parameters(), lr=0.00002)
        self.center = center
        self.centerBackup = center
        self.center.requires_grad=True
        self.optimCenter = torch.optim.Adam([self.center], lr=0.005)
        self.lastLoss = 1.

    def createUnusedSectorLoss(self, centered_points, size):
        sphericalInput = linAlgHelper.asSpherical(centered_points)
        xBinIdx = ((sphericalInput[:,0].clamp_(-np.pi,np.pi)+np.pi)*3.183).long() #pi*3.183... = 10
        yBinIdx = ((sphericalInput[:,1].clamp_(-np.pi/2,np.pi/2)+np.pi/2)*3.183).long()
        idx = torch.stack([xBinIdx,yBinIdx],1)
        usedSectors = torch.unique(idx,dim=0)   
        _,counts = torch.unique(torch.cat((usedSectors,referenceUnusedList),0),return_counts=True, dim=0)
        unusedSectors = referenceUnusedList[counts==1]
        if len(unusedSectors) == 0:
            return 0.
        unusedPoints = (unusedSectors.repeat(len(centered_points)//len(unusedSectors),1)+torch.rand((len(unusedSectors)*(len(centered_points)//len(unusedSectors)),2)).cuda())/3.183
        unusedPoints[:,0] = unusedPoints[:,0]-np.pi
        unusedPoints[:,1] = unusedPoints[:,1]-np.pi/2.
        #create prediction
        circularIn = circular2sinCosC(unusedPoints.float())
        prediction1 = self.learnModel(circularIn)
        prediction = self.learnModelLastLayer(prediction1)
        with torch.no_grad():
            prediction2a = self.learnModel2(circularIn)
            prediction2 = self.learnModelLastLayer2(prediction2a)
            midPredict = (prediction.detach()+prediction2)/2.
        loss = torch.sigmoid((prediction- midPredict).abs()*30.).mean()
        return loss

    def regularizeWithoutInformationLoss(self, centered_points, size):
        #step 1: create difference between predictions 
        sphericalInput = linAlgHelper.asSpherical(centered_points)
        xBinIdx = ((sphericalInput[:,0].clamp_(-np.pi,np.pi)+np.pi)*3.183).long() #pi*3.183... = 10
        yBinIdx = ((sphericalInput[:,1].clamp_(-np.pi/2,np.pi/2)+np.pi/2)*3.183).long()
        idx = torch.stack([xBinIdx,yBinIdx],1)
        usedSectors = torch.unique(idx,dim=0)   
        _,counts = torch.unique(torch.cat((usedSectors,referenceUnusedList),0),return_counts=True, dim=0)
        unusedSectors = referenceUnusedList[counts==1]
        if len(unusedSectors) == 0:
            return 0.
        unusedPoints = (unusedSectors.repeat(len(centered_points)//len(unusedSectors),1)+torch.rand((len(unusedSectors)*(len(centered_points)//len(unusedSectors)),2)).cuda())/3.183
        unusedPoints[:,0] = unusedPoints[:,0]-np.pi
        unusedPoints[:,1] = unusedPoints[:,1]-np.pi/2.
        #create prediction
        with torch.no_grad():
            circularIn = circular2sinCosC(unusedPoints.float())
            prediction1 = self.learnModel(circularIn)
            prediction = self.learnModelLastLayer(prediction1)
            prediction2a = self.learnModel2(circularIn)
            prediction2 = self.learnModelLastLayer2(prediction2a)
            predictionValue = 0.01*size/((prediction-prediction2).abs()+0.01*size)
            pointMask = predictionValue > 0.3
            predictionError = ((prediction-prediction2)[pointMask]).abs().mean()
        #step 2: regularize net 1
        l2_lambda = min(len(centered_points),5000)/5000*0.01
        l2_regularizer = sum(p.pow(4.0).sum()
                            for p in self.learnModel.parameters())
        l2_regularizer += sum(p.pow(4.0).sum()
                            for p in self.learnModelLastLayer.parameters())
        loss = l2_lambda * l2_regularizer
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimLastReg.zero_grad()
        self.optimLearnReg.zero_grad()
        self.optimLast2.zero_grad()
        self.optimLearn2.zero_grad()
        self.optimLastReg2.zero_grad()
        self.optimLearnReg2.zero_grad()
        if len(pointMask) > 0:
            circularIn = circularIn[pointMask[:,0]]
            prediction2 = prediction2[pointMask[:,0]] 
            #step 3: train net 1 on targets from net 2 untill the difference is equal to 1
            predictionErrorNotReached = True
            trys = 0
            while(predictionErrorNotReached):
                prediction1 = self.learnModel(circularIn)
                prediction = self.learnModelLastLayer(prediction1)
                error = ((prediction-prediction2).abs()).mean()
                if error < predictionError:
                    predictionErrorNotReached = False
                (error*0.01).backward()
                self.optimLastReg.step()
                self.optimLearnReg.step()
                self.optimLastReg.zero_grad()
                self.optimLearnReg.zero_grad()
                self.optimLast.zero_grad()
                self.optimLearn.zero_grad()
                self.optimLast2.zero_grad()
                self.optimLearn2.zero_grad()
                self.optimLastReg2.zero_grad()
                self.optimLearnReg2.zero_grad()
                trys += 1
                if trys > 10:
                    break

    
    def train(self, centered_points, value, size):
        '''points are the n,3 karthesian coordinate points.  value is the certainty, that the point belongs to the convex part. It will have an effekt on the loss of the network.
            The value is initially derived from the differenciation inside-the bounds, outside (near) the bounds'''
        self.regularizeWithoutInformationLoss(centered_points, size)
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimCenter.zero_grad()
        self.optimLastReg.zero_grad()
        self.optimLearnReg.zero_grad()
        self.optimLast2.zero_grad()
        self.optimLearn2.zero_grad()
        self.optimLastReg2.zero_grad()
        self.optimLearnReg2.zero_grad()
        prediction, difference, prediction1 = self.predict(centered_points)
        loss = torch.nn.functional.leaky_relu(difference*value, negative_slope=0.3).abs().mean()
        if loss.isnan():
            print('loss is nan')
            return torch.tensor(0.)
        #Regularize Model center to Prediction center
        centerCorrection = modelCenterCorrection(self.learnModel,self.learnModelLastLayer)
        if centerCorrection.isnan().sum() > 0:
            print('centerCorrection is nan')
            centerCorrection = torch.zeros(3).cuda()
        centerError = torch.nn.functional.l1_loss(self.center,centerCorrection+self.center.detach())
        if centerError.isnan():
            print('centerError is nan')
            return torch.tensor(0.)
        loss += centerError
        
        lossFactor = self.lastLoss
        self.lastLoss = loss.detach()
        #regularisation is only possible if there are points from all "sections" - so we have to imagine a few for this
        l2_lambda = min(len(centered_points),5000)/5000*0.01
        l2_regularizer = sum(p.pow(4.0).sum()
                        for p in self.learnModel.parameters())
        l2_regularizer += sum(p.pow(4.0).sum()
                        for p in self.learnModelLastLayer.parameters())
        loss = (loss + l2_lambda * l2_regularizer)*lossFactor
        if loss.isnan():
            print('reg or lossFactor is nan')
            return torch.tensor(0.)
        with torch.no_grad():
            self.centerBackup = self.center.detach().clone()
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        self.optimCenter.step()
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimCenter.zero_grad()
        self.optimLastReg.zero_grad()
        self.optimLearnReg.zero_grad()
        self.optimLast2.zero_grad()
        self.optimLearn2.zero_grad()
        self.optimLastReg2.zero_grad()
        self.optimLearnReg2.zero_grad()
        with torch.no_grad():
            if self.center.isnan().sum() > 0:
                print("resetting center to ", self.centerBackup)
                self.center = self.centerBackup.clone()
                self.center.requires_grad=True
                self.optimCenter = torch.optim.Adam([self.center], lr=0.005)
        #switch model 1 and 2
        self.learnModel, self.learnModelLastLayer, self.optimLast, self.optimLearn, self.learnModel2, self.learnModelLastLayer2, self.optimLast2, self.optimLearn2 = self.learnModel2, self.learnModelLastLayer2, self.optimLast2, self.optimLearn2, self.learnModel, self.learnModelLastLayer, self.optimLast, self.optimLearn
        self.optimLastReg, self.optimLearnReg, self.optimLastReg2, self.optimLearnReg2 = self.optimLastReg2, self.optimLearnReg2, self.optimLastReg, self.optimLearnReg
        return self.lastLoss
    
    def trainEmpty(self, empty_centered_points, size):
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        prediction, difference, prediction1 = self.predict(empty_centered_points)
        loss = torch.nn.functional.relu(-difference/size).abs().sum()
        if loss.isnan():
            print('emptyloss is nan')
            return torch.tensor(0.)
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        return difference.detach()

    def keepShape(self,sphericalInput):
        sphericalInput = self.createUnusedPoints(sphericalInput)
        circularIn = circular2sinCosC(sphericalInput[:,:2].float())
        prediction1 = self.learnModel(circularIn)
        prediction = self.learnModelLastLayer(prediction1)
        difference = prediction-sphericalInput[:,2][:,None]
        return prediction, difference, prediction1
        
    def predict(self,centered_points):
            sphericalInput = linAlgHelper.asSpherical(centered_points)
            circularIn = circular2sinCosC(sphericalInput[:,:2].float())
            prediction1 = self.learnModel(circularIn)
            prediction = self.learnModelLastLayer(prediction1)
            difference = prediction-sphericalInput[:,2][:,None]
            return prediction, difference, prediction1

    def predict2(self,centered_points):
            sphericalInput = linAlgHelper.asSpherical(centered_points)
            circularIn = circular2sinCosC(sphericalInput[:,:2].float())
            prediction1 = self.learnModel2(circularIn)
            prediction = self.learnModelLastLayer2(prediction1)
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

neuRec = NeuralConvexReconstruction(torch.tensor([73.09600081,  17.79199924, -53.68591931]).cuda())
target = tm.load("Einstein_bust.obj")

path = []

#train
smoothLoss = 0.
smoothLoss2 = 0.
first = True

for i in range(1600):
    tpoints = torch.tensor(target.sample(5000))
    mask = tpoints[:,2] > -50.
    tpoints = tpoints[mask]
    if torch.isnan(neuRec.center).sum() > 0:
        print("error")
    loss = abs(neuRec.train(tpoints.float().cuda()-neuRec.center, torch.ones(5000).cuda(), 1.))
    loss2 = abs(neuRec.train(tpoints.float().cuda()-neuRec.center, torch.ones(5000).cuda(), 1.))
    if first:
        smoothLoss = loss
        smoothLoss2 = loss2
        first = False
    smoothLoss = smoothLoss*0.95 + loss.item()*0.05
    smoothLoss2 = smoothLoss2*0.95 + loss2.item()*0.05
    if i%100 == 0:
        path.append(neuRec.show())
        print(smoothLoss, smoothLoss2)
        print(neuRec.center.detach())

print("Switching Pointtargets")
for i in range(1600):
    tpoints = torch.tensor(target.sample(5000))
    mask = tpoints[:,2] < -50.
    tpoints = tpoints[mask]
    if torch.isnan(neuRec.center).sum() > 0:
        print("error")
    loss = abs(neuRec.train(tpoints.float().cuda()-neuRec.center, torch.ones(5000).cuda(), 1.))
    loss2 = abs(neuRec.train(tpoints.float().cuda()-neuRec.center, torch.ones(5000).cuda(), 1.))
    smoothLoss = smoothLoss*0.95 + loss.item()*0.05
    smoothLoss2 = smoothLoss2*0.95 + loss2.item()*0.05
    if i%100 == 0:
        path.append(neuRec.show())
        print(smoothLoss, smoothLoss2)
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