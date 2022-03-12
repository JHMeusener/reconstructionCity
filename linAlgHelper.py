import torch
import numpy as np

def asSpherical(xyz):
    rad = torch.zeros_like(xyz)
    rad[:,2] =  (xyz[:,0]**2+xyz[:,1]**2+xyz[:,2]**2)**0.5
    zeroMask=rad[:,2]!=0
    rad[:,0][zeroMask]   =  torch.acos(xyz[:,2][zeroMask]/rad[:,2][zeroMask])*torch.sign(xyz[:,0][zeroMask])
    #rad[:,1]   =  torch.atan2(xyz[:,1],xyz[:,0])
    rad[:,1][(xyz[:,0] > 0.)]= torch.atan(xyz[:,1]/xyz[:,0])[(xyz[:,0] > 0.)] 
    rad[:,1][(xyz[:,0] < 0.)]= (xyz[:,1][(xyz[:,0] < 0.)] >= 0.) * (torch.atan(xyz[:,1]/xyz[:,0]))[(xyz[:,0] < 0.)]\
    + (xyz[:,1][(xyz[:,0] < 0.)] < 0.) * (torch.atan(xyz[:,1]/xyz[:,0]))[(xyz[:,0] < 0.)]
    rad[:,1][(xyz[:,0] == 0.)] = (xyz[:,1] > 0.)[(xyz[:,0] == 0.)]* np.pi/2.\
    + (xyz[:,1] < 0.)[(xyz[:,0] == 0.)] * -np.pi/2.
    return rad

def asCartesian(thetaphir):
    #takes list rthetaphi (single coord)
    xyz = torch.zeros_like(thetaphir)
    xyz[:,0] =  thetaphir[:,2] * torch.sin( thetaphir[:,0] ) * torch.cos( thetaphir[:,1] )
    xyz[:,1] = thetaphir[:,2] * torch.sin( thetaphir[:,0] ) * torch.sin( thetaphir[:,1] )
    xyz[:,2] =  thetaphir[:,2] * torch.cos(   thetaphir[:,0]  )
    return xyz

def p2NormLastDim(tensor):
    return (tensor**2).sum(dim=-1)**0.5

def getPlaneNormalFrom3Points(pArray):
    '''JHM: 
    input: [batch,planes,3,xyz]
    output: [batch,planes,3]
    creates a normal vector that is the supporting vector
    of the plane spanned by the 3 points'''
    #supportVector = pArray[:,:,0,:].clone()[:,:,None,:]
    #zero the pArray
    #pArray = pArray-pArray[:,:,0,:][:,:,None,:]
    #get normal
    normal = torch.cross(pArray[:,:,1,:]-pArray[:,:,0,:], pArray[:,:,2,:]-pArray[:,:,0,:], dim=-1, out=None)[:,:,None,:]
    #project the supporting-vector onto the normal
    normal = normal*1000*torch.sign(torch.matmul(normal,(pArray[:,:,0,:][:,:,None,:]).permute(0,1,3,2))).repeat(1,1,1,3)
    return normal[:,:,0,:]

def projectVectorOnVector(vector, projection, epsilon=0.001):
    '''Batch projects a Vector onto another Vector
    Input: Vector [Batch, Planes,1, XYZ]
    Input: Projection [Batch, Planes, 1, XYZ]'''
    projectionLength = p2NormLastDim(projection)
    vectorLength = p2NormLastDim(vector)
    vectorLength[vectorLength == 0.] += epsilon
    normVector = vector/(vectorLength[:,:,:,None])
    projection[projectionLength == 0.] = normVector[projectionLength == 0.]*epsilon
    ankathete = torch.matmul(projection,vector.permute(0,1,3,2))[:,:,:,0]/(vectorLength)
    return (normVector * (ankathete[:,:,:,None])+torch.sign(ankathete[:,:,:,None])*epsilon)[:,:,0,:]

def getPointDistances2PlaneNormal(points,normal):
    '''JHM:
    input:points[batch,number,3]
          normal[batch,planes,3]
    output: dist[batch,number,1]
    Gets the distances from the plane by the normal vector (plane goes to 0, so if 
    distance == planeNormal.norm(p=2), then the plane goes through the points)'''
    pointsLength = p2NormLastDim(points)
    normalLength = p2NormLastDim(normal)
    ankathete = torch.matmul(points[:,:,None,None,:],normal[:,None,:,:,None])[:,:,:,0,0]/(normalLength[:,None,:])
    return normalLength[:,None,:]-ankathete
   
def getLineIntersection(p1Normal, p2Normal):
    '''JHM: gets the intersection-Line of 2 planes
    Input: plane1Normal [B,3] (normed to the distance of plane to origin)
           plane2Normal [B,3]
    Output: [lineEQ [B,3]
            lineNormal[B,3] (Stützstelle)]
    '''
    #lineEQ = torch.cross(p1Normal, p2Normal, dim=-1, out=None)
    # for getting the lineNormal we use the fact that lineEQ is Perpendicular to normal1 and normal2
    #first we get V1 and V2, which are the missing perpendicular lines between L and the planeNormals
    #V1 = torch.cross(p1Normal, lineEQ, dim=-1, out=None)
    #V2 = torch.cross(p2Normal, lineEQ, dim=-1, out=None)
    #L1 = pNorm1 + a V1
    #L2 = pNorm2 + b V2
    #--->  pNorm1 + a V1 = pNorm2 + b V2
    #---> a V1 = (pNorm2 - pNorm1) + b V2
    #---> a (V1 X V2) = (pNorm2 - pNorm1) X V2
    #---> a = ||(pNorm2 - pNorm1) X V2|| / ||(V1 X V2)||
    # and with a we can get the Pi with pNorm1-a*V1
    lineEQ = torch.cross(p1Normal, p2Normal, dim=-1, out=None)
    lineEQ = lineEQ/p2NormLastDim(lineEQ)[:,None]
    L1 = torch.cross(p1Normal, lineEQ, dim=-1, out=None)
    L2 = torch.cross(p2Normal, lineEQ, dim=-1, out=None)
    L1 = L1/p2NormLastDim(L1)[:,None]
    L2 = L2/p2NormLastDim(L2)[:,None]
    crosss = torch.cross((p2Normal-p1Normal),L2,dim=-1)
    a = p2NormLastDim(crosss)
    Pi = -a[:,None]*L1+p1Normal
    return lineEQ, Pi

def linePlaneIntersection(pNormal, lineEQ, lineNormal):
    '''Gets the intersection of a line with a plane
        input: pNormal[B,b,3] (normalized to be its own supporting Vector)
               lineEQ[B,b,3]
               lineNormal[B,b,3]
        output: intersectionPoints[B,b,3]
    '''
    cosPNormalLineEQ = torch.matmul(pNormal[:,:,None,:],lineEQ[:,:,:,None])
    linePlaneDiff = lineNormal-pNormal
    s = -torch.matmul(pNormal[:,:,None,:],linePlaneDiff[:,:,:,None])/cosPNormalLineEQ
    intersection = linePlaneDiff+s[:,:,:,0]*lineEQ+pNormal
    return intersection

def getObjectBinIdx(objectTensor,device="cpu"):
    '''gets an index which corresponds to the object of the object tensor in one dimension
    and the correct bin position in the other dimension
    input: objectTensor[bins] - every bin has a object index 
                ex. 5 bins 3 objects: torch.tensor([0,1,1,1,2])
    output: index[bins,maxBinsPerObject]
                ex. 5 bins 3 objects: tensor([[0, 0, 1, 2, 0],
                                               [0, 1, 1, 1, 2]])
            maxBinsPerObject
                 ex. 5 bins 3 objects: 3
            objectNr
                 ex. 5 bins 3 objects: 3
            binNr
                 ex. 5 bins 3 objects: 5
            '''
    consecutive = torch.unique_consecutive(objectTensor, return_counts=True, dim=0)
    objectNr = consecutive[0].shape[0]
    binNr = objectTensor.shape[0]
    maxBinsPerObject = consecutive[1].max()
    aranges = torch.arange(maxBinsPerObject+1, device=device).repeat(objectNr,1)
    aranges[torch.arange(consecutive[0].shape[0], device=device),consecutive[1]] = torch.ones(objectNr, device=device).long()*(-maxBinsPerObject**2)
    idx1 = aranges[(aranges.cumsum(dim=1)>-1)].flatten()
    idx2 = torch.arange(consecutive[0].shape[0], device=device).repeat_interleave(consecutive[1])
    idx = torch.stack([idx1,idx2],dim=0).unique_consecutive(dim=1)
    return idx, maxBinsPerObject, objectNr, binNr

def createPlaneExample(normals,extrema, device="cpu"):
    '''Creates Vertices and Faces from Normals and Extrema 
    by creating a Point an the plane perpendicular to the other 2
    Input: normals [B,3]
            extrema [B,3]
    output: vertices [3*B,3]
            faces [B,3]
    '''
    other = torch.cross(normals, extrema, dim=-1, out=None)
    other = (other/p2NormLastDim(other)[:,None])*2.+normals
    vertices = torch.cat([other,normals,extrema],dim=-2)
    faces = torch.arange(vertices.shape[-2]//3, device=device)
    faces = faces[:,None].repeat(1,3)
    faces[:,1] += vertices.shape[-2]//3
    faces[:,2] += 2*vertices.shape[-2]//3
    return vertices,faces

def getPerObjectIdx(Idx, objectTensor, device="cpu"):
    '''gets the objects from the 2d index
    input: index[objects,items2getPerObject]
           objects[Objects,Bins,otherValues]
    output: [objects,items2getPerObject,otherValues]
    '''
    #reshaping the tensor to get the per object maximum
    itemsToGet = Idx.shape[1]
    binsize = objectTensor.shape[1]
    objects = objectTensor.shape[0]
    otherValues = objectTensor.shape[2]
    Idx2 = Idx+torch.arange(objects, device=device).repeat(itemsToGet,1).permute(1,0) * binsize
    vals = objectTensor.reshape(objectTensor.shape[0]*objectTensor.shape[1],otherValues)[Idx2.flatten(),:]
    return vals.reshape(objects,itemsToGet,otherValues)

def maskEvenTensorOutput(rawMask, values, maskedValue=None, device="cpu"):
    '''This function adds "True" to a rawMask until all Batches have the same length
    all usually masked values get maskedValue or will not be changed
    input: rawMask boolean[batch,objects]
           values [batch,objects,values]
           maskedValue float('NaN') oder 0 
           device "cpu" oder torch.device(0)
    output: values will be changed
            mask [batch, objects]
            trueTrues flat [index] which True of the mask is a true true'''
    sumValid = rawMask.sum(dim=1)
    maxVal = sumValid.max()
    if maskedValue:
        values[~rawMask] *= maskedValue
    batchsize = rawMask.shape[0]
    batch = torch.arange(batchsize, device = device).repeat_interleave(rawMask.shape[1],dim=0)
    arr, indices = (rawMask*1).sort(descending=True)
    indices = (indices+(batch*indices.shape[1]).reshape(indices.shape))[:,:maxVal.item()].flatten()
    rawMask.flatten()[indices] = (torch.ones(indices.flatten().numel(), device=device)==1)
    return rawMask
