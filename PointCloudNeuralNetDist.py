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

ball = tm.primitives.Capsule(radius=1., height=0.,sections=128)
referenceUnusedList = torch.stack((torch.arange(20).repeat_interleave(10),torch.arange(10).repeat(20)),1).cuda()

class NeuralConvexReconstruction:
    def __init__(self):
        self.learnModel = SIREN([12,64, 64], lastlayer=False).cuda() #torch.nn.Sequential(torch.nn.Linear(2,256),
        #                                        torch.nn.ReLU(),
        #                                        torch.nn.Linear(256,256),
        #                                        torch.nn.ReLU(),
        #                                        torch.nn.Linear(256,256),
        #                                        torch.nn.ReLU()  ).cuda() #
        self.learnModelLastLayer = SIREN([64], lastlayer=True).cuda()#torch.nn.Sequential(torch.nn.Linear(256,1)).cuda() #
        self.optimLearn = torch.optim.Adam(self.learnModel.parameters(), lr=0.0004)
        self.optimLearnReg = torch.optim.Adam(self.learnModel.parameters(), lr=0.00004)
        self.optimLastReg= torch.optim.Adam(self.learnModelLastLayer.parameters(), lr=0.00004)
        self.optimLast = torch.optim.Adam(self.learnModelLastLayer.parameters(), lr=0.00004)
        self.lastLoss = 1.
        self.colorModel = SIREN([12,64, 64, 3], lastlayer=False).cuda()#torch.nn.Sequential(torch.nn.Linear(12,256),
        #                                        torch.nn.ReLU(),
        #                                        torch.nn.Linear(256,256),
        #                                        torch.nn.ReLU(),
        #                                        torch.nn.Linear(256,3),
        #                                        torch.nn.Sigmoid()  ).cuda()#
        self.optimColor = torch.optim.Adam(self.colorModel.parameters(), lr=0.0001)
    
    def train(self, centered_points, value):
        '''points are the n,3 karthesian coordinate points.  value is the certainty, that the point belongs to the convex part. It will have an effekt on the loss of the network.
            The value is initially derived from the differenciation inside-the bounds, outside (near) the bounds'''
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimLastReg.zero_grad()
        self.optimLearnReg.zero_grad()
        prediction, difference, prediction1 = self.predict(centered_points)
        loss = torch.nn.functional.leaky_relu(difference*value[:,None], negative_slope=0.1).abs().mean()*100.
        lossFactor = self.lastLoss
        self.lastLoss = loss.detach()
        l2_lambda = min(len(centered_points),5000)/5000*0.005
        l2_regularizer = sum(p.pow(2.0).sum()
                        for p in self.learnModel.parameters())
        l2_regularizer += sum(p.pow(2.0).sum()
                        for p in self.learnModelLastLayer.parameters())
        loss = (loss + l2_lambda * l2_regularizer)
        loss.backward()
        self.optimLast.step()
        self.optimLearn.step()
        self.optimLast.zero_grad()
        self.optimLearn.zero_grad()
        self.optimLastReg.zero_grad()
        self.optimLearnReg.zero_grad()
        return loss.detach(), difference.detach()

    def trainColor(self, centered_points, value, color):
        self.optimColor.zero_grad()
        prediction1 = self.predictColor(centered_points)
        loss = ((color-(prediction1)).abs()*value[:,None]).mean()*100.
        l2_lambda = min(len(centered_points),5000)/5000*0.005
        l2_regularizer = sum(p.pow(2.0).sum()
                        for p in self.colorModel.parameters())
        loss = (loss + l2_lambda * l2_regularizer)
        loss.backward()
        self.optimColor.step()
        self.optimColor.zero_grad()
        return loss.detach()

    def predict(self,centered_points):
            sphericalInput = linAlgHelper.asSpherical(centered_points)
            circularIn = circular2sinCosC(sphericalInput[:,:2].float())
            prediction1 = self.learnModel(circularIn)
            prediction = self.learnModelLastLayer(prediction1)
            difference = prediction-sphericalInput[:,2][:,None]
            return prediction, difference, prediction1

    def predictColor(self,centered_points):
        sphericalInput = linAlgHelper.asSpherical(centered_points)
        circularIn = circular2sinCosC(sphericalInput[:,:2].float())
        prediction1 = self.colorModel(circularIn)
        return torch.tanh(prediction1)/2.+0.5

    def predictSphere(self):
        inputPoints = ball.sample(100000)
        sphericalInput = linAlgHelper.asSpherical(torch.tensor(inputPoints).float())[:,:2].cuda()
        with torch.no_grad():
            prediction, difference, prediction1 = self.predict(torch.tensor(inputPoints).float().cuda())
            predictedSpherical = torch.cat((sphericalInput,abs(prediction)),dim=1)
            points = (linAlgHelper.asCartesian(predictedSpherical))
        return points, predictedSpherical
        
    def show(self,center, color = None):
        with torch.no_grad():
            points,_ = self.predictSphere()
            if color is None:
                with torch.no_grad():
                    colors = self.predictColor(points).cpu().numpy()
                points = (points+center).cpu()
                pointcloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
                pointcloud.colors = o3d.utility.Vector3dVector(colors)
                return pointcloud
            color = np.random.rand(1,3)
            points = (points+center).cpu()
            pointcloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
            colors = np.ones_like(points).astype(np.float64)
            colors[:,2] = colors[:,2]*color[2]
            colors[:,1] = colors[:,1]*color[1]
            colors[:,0] = colors[:,1]*color[0]
            pointcloud.colors = o3d.utility.Vector3dVector(colors)
            return pointcloud


if __name__ == '__main__':
    xyz = torch.tensor(np.load("testPointsXYZ.npy")).float().cuda()[list(range(70000))*2]
    xyz = xyz-xyz.min(axis=0)[0][None,:]
    xyz = xyz-xyz.max(axis=0)[0][None,:]//2
    rgb = torch.tensor(np.load("testPointsRGB.npy")).float().cuda()[list(range(70000))*2]
    rec = NeuralConvexReconstruction()
    for i in range(1300):
        lastLoss, _ = rec.train(xyz,torch.ones((len(xyz))).float().cuda())
        colorloss = rec.trainColor(xyz,torch.ones((len(xyz))).float().cuda(), rgb)
        if i%100==0:
            print(lastLoss.detach().cpu().item(), "   ", colorloss.detach().cpu().item())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
    o3d.visualization.draw_geometries([rec.show(torch.tensor([[0.,0.,0.]]).float().cuda())+pcd])
    o3d.visualization.draw_geometries([rec.show(torch.tensor([[0.,0.,0.]]).float().cuda())])
