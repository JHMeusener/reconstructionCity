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
def randCam(cubesize = 700):
    """returns a pyredner camera that is positioned in a 700 meter cube around the object at 0,0,0"""
    pos = torch.rand(3)
    pos = pos/pos.sum() * cubesize//2
    for i in range(3):
        if torch.rand(1) > 0.5:
            pos[i] *= -1.
    lookat = torch.zeros(3)
    up = torch.tensor([0.00001,1.,0.00001])
    cam = pyredner.Camera(pos, lookat, up, torch.tensor([60.],dtype=torch.float32), clip_near= 0.01, 
                          resolution= (480, 600), camera_type=pyredner.camera_type.perspective)
    return cam

#https://github.com/mono/opentk/blob/master/Source/OpenTK/Math/Matrix4.cs
def matrixLookat(eye, target, up):
    """Eye position, target position, up position. Returns rotationmatrix"""
    z = eye - target
    x = up.cross(z)
    y = z.cross(x)
    
    x=x/(x**2).sum()**0.5
    y=y/(y**2).sum()**0.5
    z=z/(z**2).sum()**0.5
    
    rot = torch.zeros((3,3))
    rot[0][0] = x[0]
    rot[0][1] = y[0]
    rot[0][2] = z[0]
    #rot[0][3] = 0
    rot[1][0] = x[1]
    rot[1][1] = y[1]
    rot[1][2] = z[1]
    #rot[1][3] = 0
    rot[2][0] = x[2]
    rot[2][1] = y[2]
    rot[2][2] = z[2]
    #rot[2][3] = 0
    
    # eye not need to be minus cmp to opentk 
    # perhaps opentk has z inverse axis
    return rot

def createInputVector_planeHitModel(center, camPosition, camVectorNorm):
    '''The Vector camVectorNormal hits 90 degree to a plane through center. This function returns the Center-Hitpoint-Vector 
    Inputs: center (1,3)  camPosition (1,3), camVectorNormal(n,3)'''
    vecCam2Center_FULL = center-camPosition
    vecCam2Center_length = (vecCam2Center_FULL**2).sum(dim=-1)**0.5 #hypotenusenlänge
    vecCam2Center_norm = vecCam2Center_FULL/vecCam2Center_length[:,None]
    cosalpha = camVectorNorm@vecCam2Center_norm.permute(1,0) #dot product with norms
    lenCamVectorFULL = vecCam2Center_length*cosalpha #Ankathete = Hypotenuse/cos(alpha)
    return vecCam2Center_FULL- lenCamVectorFULL*camVectorNorm

import torch.nn as nn
#Siren Net
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=True, dynScale=False, phaseShift=False, dynaPhaseShift=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.dynScaleB = dynScale
        self.scaleB = scale
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.forwardFunctions = []
        
        if scale:
            self.scaleLayer = torch.nn.Linear(1, out_features, bias=False)
        if dynScale:
            self.dynScaleLayer = torch.nn.Linear(in_features, out_features, bias=False)
        if phaseShift:
            self.phaseShiftLayer = torch.nn.Linear(1, out_features, bias=False)
        if dynaPhaseShift:
            self.dynPhaseShiftLayer = torch.nn.Linear(in_features, out_features, bias=False)
        self.init_weights()
        if scale:
            self.scale = self.simple_scale 
        if dynScale:
            self.scale = self.dynScaled 
        if phaseShift:
            self.shift = self.simple_phaseShift 
        if dynaPhaseShift:
            self.shift = self.dynPhaseShift 
    
    def init_weights(self):
        with torch.no_grad():
            if self.dynScaleB:
                self.dynScaleLayer.weight.uniform_(-1 / self.in_features, 
                                                 1 / self.in_features) 
            if self.scaleB:
                self.scaleLayer.weight.uniform_(0.99, 
                                                1.00) 
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    
    def shift(self,input):
        return 0.

    def scale(self,input):
        return 1.

    def forward(self, input):
        return self.scale(input)*torch.sin(self.omega_0 * self.linear(input) + self.shift(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    def simple_scale(self, input):
        return self.scaleLayer(torch.tensor([1.]).cuda())

    def simple_phaseShift(self, input):
        return self.phaseShiftLayer(torch.tensor([1.]).cuda())

    def dynPhaseShift(self, input):
        phaseShift = self.phaseShiftLayer(torch.tensor([1.]).cuda())
        dynPhaseShift = self.dynPhaseShiftLayer(input)
        return dynPhaseShift+phaseShift
    
    def dynScaled(self, input):
        return self.dynScaleLayer(input)+self.scaleLayer(torch.tensor([1.]).cuda())

class SIREN(nn.Module):
    '''
    this model is a fully connected model with sin active func.
    basic implementation of Implicit Neural Representations with Periodic Activation Functions
    '''
    def __init__(self, dims,lastlayer=True, scale = True, dynScale = True, phaseShift=True, dynaPhaseShift=False):
        super(SIREN, self).__init__()
        self.layers = nn.ModuleList()
        self.lastLayer = torch.nn.Linear(dims[-1], 1, bias=True)
        for i in range(len(dims)-1):
            self.layers.append(SineLayer(dims[i],dims[i+1], scale=scale, dynScale=dynScale, phaseShift=phaseShift, dynaPhaseShift=dynaPhaseShift))
        with torch.no_grad():
                self.lastLayer.weight.uniform_(-1.0, 
                                                1.0)
        self.lastlayer_bool = lastlayer

    def forward(self, x):
        # save middle result for gradient calculation
        for layer in self.layers:
            x = layer(x)
        # last layer
        if self.lastlayer_bool:
            result = self.lastLayer(x)
            return result
        else: return x

def circular2sinCosC(inputVectorsTAR):
    """returns a few sin/cos coordinates"""
    sinVec = torch.sin(inputVectorsTAR[:,0])[:,None]
    cosVec = torch.cos(inputVectorsTAR[:,0])[:,None]
    cosVec2 = torch.cos(inputVectorsTAR[:,1])[:,None]
    sinVec2 = torch.sin(inputVectorsTAR[:,1])[:,None]
    sinVecB = torch.sin(inputVectorsTAR[:,0]*2)[:,None]
    cosVecB = torch.cos(inputVectorsTAR[:,0]*2)[:,None]
    sinVec2B = torch.sin(inputVectorsTAR[:,1]*4)[:,None]
    cosVec2B = torch.cos(inputVectorsTAR[:,1]*4)[:,None]
    sinVecC = torch.sin(inputVectorsTAR[:,0]*4)[:,None]
    cosVecC = torch.cos(inputVectorsTAR[:,0]*4)[:,None]
    sinVec2C = torch.sin(inputVectorsTAR[:,1]*8)[:,None]
    cosVec2C = torch.cos(inputVectorsTAR[:,1]*8)[:,None]
    return torch.cat([sinVec,cosVec,cosVec2,sinVec2,sinVecB,cosVecB,sinVec2B,cosVec2B, sinVecC, cosVecC, sinVec2C, cosVec2C, inputVectorsTAR[:,2:]],dim=1)

import sys, os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def getView(objects):
    with HiddenPrints():
        camera = randCam()
        scene = pyredner.Scene(camera = camera, objects = objects)
        light = pyredner.PointLight(position = (camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                                                        intensity = torch.tensor((20000.0, 30000.0, 20000.0), device = pyredner.get_device()))
        img = pyredner.render_deferred(scene = scene, lights = [light])
        rgb = torch.pow(img, 1.0/2.2).cpu()
        img = pyredner.render_deferred(scene = scene, lights = [light])
        img = pyredner.render_g_buffer(scene = scene, channels = [pyredner.channels.position,
                                                                                                             pyredner.channels.shading_normal,
                                                                                                             pyredner.channels.diffuse_reflectance])
        pos = img[:, :, :3]/100
        mask = pos.abs().sum(dim=2) > 0
    return img, pos, mask, camera


def bound2Mesh(bound):
    testPoints = torch.rand(500000,3)*6.-3.
    boundsTest = linAlgHelper.getPointDistances2PlaneNormal(testPoints[None,:,:], bound[None,:,:].detach())[0]
    inside = boundsTest>0
    inside = inside.sum(dim=1)==inside.shape[1]
    filtered = testPoints[inside]
    filteredIdx = torch.arange(len(filtered))
    hull = ConvexHull(filtered)
    verts_ = torch.tensor(hull.vertices)
    vertIdx = torch.arange(len(verts_))
    filteredIdx[verts_.long()] = vertIdx
    faces_ = torch.tensor(hull.simplices)
    return filtered[verts_.long()], filteredIdx[faces_.long()]

def bound2Pointcloud(bound, center = torch.zeros((1,3)), color=[0,0,1]):
    testPoints = torch.rand(50000,3)*6.-3.
    boundsTest = linAlgHelper.getPointDistances2PlaneNormal(testPoints[None,:,:], bound[None,:,:].cpu().detach())[0]
    inside = boundsTest>0
    inside = inside.sum(dim=1)==inside.shape[1]
    filtered = testPoints[inside] + center.detach().cpu()
    pointcloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(filtered))
    pointcloud.colors = o3d.utility.Vector3dVector(np.hstack([np.ones((filtered.shape[0],1)), 
                                                              np.zeros((filtered.shape[0],1))  , 
                                                              np.zeros((filtered.shape[0],1))]).astype(np.float64))
    return pointcloud

def meshIt(model, simple=True, sections = 100):
        ball = tm.primitives.Capsule(radius=1., height=0.,sections=sections)
        sphericalInput = linAlgHelper.asSpherical(torch.tensor(ball.vertices).float())[:,:2].cuda()
        circularOut = circular2sinCosC(sphericalInput)
        with torch.no_grad():
            if simple:
                distances = model(circularOut.cuda()).abs()
            else:
                circularOut = torch.cat((circularOut,torch.ones_like(circularOut[:,0])*0.005),dim=1)
                distances = model(circularOut)
            predictedSpherical = torch.cat((sphericalInput,distances),dim=1)
            points = linAlgHelper.asCartesian(predictedSpherical).cpu()
            #distances2Bounds = linAlgHelper.getPointDistances2PlaneNormal(points[None,:,:], self.bounds[None,:,:])
            #wenn etwas negativ ist, berechne das dreieck ankathete-bound, hypotenusenvektor-punkt
        
        return points, torch.tensor(ball.faces)

def compare2CenteredModels(model1,model2, centerdifference=torch.tensor([[0.,0.,0.]])):
    sphericalTarget = linAlgHelper.asSpherical(torch.rand(15000,3)-0.5)[:,:2].float().cuda()
    circularIn = circular2sinCosC(sphericalTarget)
    with torch.no_grad():
        target = model1(circularIn)
        target2 = model2(circularIn)
    predictedSpherical1 = torch.cat((sphericalTarget,target),dim=1)
    predictedSpherical2 = torch.cat((sphericalTarget,target2),dim=1)
    points1 = linAlgHelper.asCartesian(predictedSpherical1).cpu()    
    points2 = linAlgHelper.asCartesian(predictedSpherical2).cpu()  - centerdifference  
    
    pointcloud1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points1))
    pointcloud1.colors = o3d.utility.Vector3dVector(np.hstack([np.ones((points1.shape[0],1)), 
                                                               np.zeros((points1.shape[0],1)) , 
                                                               np.zeros((points1.shape[0],1))]).astype(np.float64))

    pointcloud2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points2))
    pointcloud2.colors = o3d.utility.Vector3dVector(np.hstack([np.zeros((points1.shape[0],1)), 
                                                               np.ones((points1.shape[0],1)) , 
                                                               np.zeros((points1.shape[0],1))]).astype(np.float64))
    return pointcloud1, pointcloud2

ball = tm.primitives.Capsule(radius=1., height=0.,sections=100)
def modelCenterCorrection(model, modelLastlayer):
    '''returns a vector in the direction of the model center prediction'''
    samples = torch.tensor(ball.sample(500)).float().cuda()
    sphericalInput = linAlgHelper.asSpherical(samples)[:,:2]
    sphericalInput2 = linAlgHelper.asSpherical(-samples)[:,:2]
    circularOut = circular2sinCosC(sphericalInput)
    circularOut2 = circular2sinCosC(sphericalInput2)
    with torch.no_grad():
        prediction1 = model(circularOut)
        distances = modelLastlayer(prediction1)
        prediction1 = model(circularOut2)
        distances2 = modelLastlayer(prediction1)
    weightedPoints = samples*distances - samples*distances2
    return weightedPoints.mean(dim=0)

ball = tm.primitives.Capsule(radius=1., height=0.,sections=8)
def getPredictionPoints(offset = 0.05, samples = 16):
    sphericalInput = linAlgHelper.asSpherical(torch.tensor(ball.sample(samples)).float())[:,:2].cuda()
    circularOut = circular2sinCosC(sphericalInput)
    with torch.no_grad():
        distances = abs(model(circularOut.cuda()))
    offsetDistances = distances + offset
    predictedSpherical = torch.cat((sphericalInput,distances),dim=1)
    predictedSphericalOffset = torch.cat((sphericalInput,offsetDistances),dim=1)
    points = linAlgHelper.asCartesian(predictedSpherical).cpu()   
    pointsOffset = linAlgHelper.asCartesian(predictedSphericalOffset).cpu()   
    return points, pointsOffset

def bound2bounds(bound):
    '''transforms the first two bound coordinates into a cube around xyz'''
    bounds = torch.cat([torch.zeros((6,3)),bound[2:,:]],0)
    bounds[0,0] = abs(bound[0,0])
    bounds[1,0] = -abs(bound[0,1])
    bounds[2,1] = abs(bound[0,2])
    bounds[3,1] = -abs(bound[1,0])
    bounds[4,2] = abs(bound[1,1])
    bounds[5,2] = -abs(bound[1,2])
    return bounds

def meshBoundsTM(bound):
    #meshing the bounds
    size = bound[:6].detach().abs().max()
    data = torch.cat((torch.rand(50000,3)*2*size-size,bound),0)
    boundsTest = linAlgHelper.getPointDistances2PlaneNormal(data[None,:,:], bound.detach()[None,:,:])[0]
    inside = boundsTest>-0.0001
    inside = inside.sum(dim=1)==inside.shape[1]
    filtered = data[inside]
    filteredIdx = torch.arange(len(filtered))
    hull = ConvexHull(filtered)
    verts_ = torch.tensor(hull.vertices)
    vertIdx = torch.arange(len(verts_))
    filteredIdx[verts_.long()] = vertIdx
    faces_ = torch.tensor(hull.simplices)
    vertices, faces =  filtered[verts_.long()], filteredIdx[faces_.long()]
    mesh = tm.Trimesh(vertices=vertices, faces=faces)
    return mesh

def mesh2pointcloud(mesh, center = np.array([[0,0,0]]), color=[1,0,0]):
    #visualisieren der Zielfiguren
    try:
        pointcloudPoints = mesh.sample(2000)+center
    except:
        pointcloudPoints = mesh.sample(2000)+center.numpy()
    pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
    colors = np.ones_like(pointcloudPoints).astype(np.float64)
    colors[:,2] = colors[:,2]*color[2]
    colors[:,1] = colors[:,1]*color[1]
    colors[:,0] = colors[:,1]*color[0]
    pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
    return(pointcloudMesh)

def array2Pointcloud(array, center = torch.tensor([[0,0,0]]), color=[1,0,0]):
    pointcloudPoints = array+center
    pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
    colors = np.ones_like(pointcloudPoints).astype(np.float64)
    colors[:,2] = colors[:,2]*color[2]
    colors[:,1] = colors[:,1]*color[1]
    colors[:,0] = colors[:,1]*color[0]
    pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
    return(pointcloudMesh)


