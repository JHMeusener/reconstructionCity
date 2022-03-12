import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh as tm
import pybullet as p
import pybullet_data as pd
from skimage import measure

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

def voxelDict(tmObj, voxSize=0.01, samplesize = 1000000):
    samples, faceidx = tm.sample.sample_surface(tmObj, samplesize)
    normals = tmObj.face_normals[faceidx]
    colors = np.array(tmObj.visual.face_colors[faceidx,:3]/255.)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxSize)
    normalinGridPosition = ((samples)/voxel_grid.voxel_size).astype(int)
    #create mean normal (and color) for each voxel
    voxels = {}
    for i in range(len(normals)):
        if tuple(normalinGridPosition[i]) in voxels:
            voxels[tuple(normalinGridPosition[i])][0].append(normals[i])
            voxels[tuple(normalinGridPosition[i])][1].append(colors[i])
        else:
            voxels[tuple(normalinGridPosition[i])] = [[normals[i]],[colors[i]]]
    for key in voxels.keys():
        if len(voxels[key][0]) == 1:
            norm = voxels[key][0][0]
            col = voxels[key][1][0]
            voxels[key] = {"normal":norm,"color":col}
        else:
            norm = np.stack(voxels[key][0],0).mean(axis=0)
            col = np.stack(voxels[key][1],0).mean(axis=0)
            voxels[key] = {"normal":norm,"color":col}
    return voxels

#grow voxels in the direction of their normals for about 5 voxels
def growVoxelsNormal(voxels, step=5, DirectionSign=-1):
    '''input: voxels     dict of voxels with key ["normal"] and ["color"]  ([x,y,z], [r,g,b])
    output: voxels     dict of voxels with key ["normal"] and ["color"]  ([x,y,z], [r,g,b]) and "step" step of the line'''
    newVoxels = {}
    for key in voxels.keys():
        #print("doing work for key ",key)
        #print("with voxel ",voxels[key])
        for i in range(step):
            direction = (voxels[key]["normal"] * DirectionSign* (1.49+i)).astype(int)
            if tuple((key[0]+direction[0], key[1]+direction[1], key[2]+direction[2])) in voxels:
                #print("    there is a voxel at ",(key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))
                break
            else:
                if tuple((key[0]+direction[0], key[1]+direction[1], key[2]+direction[2])) in newVoxels:
                    if newVoxels[tuple((key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))]["step"] < i:
                        newVoxels[tuple((key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))]["step"] = i
                    #print("   there is a new voxel at ",(key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))
                    continue
                else:
                    #print("   creating new voxel at",(key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))
                    newVoxels[tuple((key[0]+direction[0], key[1]+direction[1], key[2]+direction[2]))] = {"normal":voxels[key]["normal"], "color":voxels[key]["color"], "step":i}
    return newVoxels

def upSample2(voxels):
    newVoxels = {}
    for key in voxels.keys():
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    newVoxels[tuple((key[0]*2+x,key[1]*2+y,key[2]*2+z))] = {"normal":voxels[key]["normal"], "color":voxels[key]["color"]}
    return newVoxels

def downSample2(voxels):
    newVoxels = {}
    for key in voxels.keys():
        newkey = tuple((key[0]//2,key[1]//2,key[2]//2))
        if newkey in newVoxels:
            newVoxels[newkey]["normal"].append(voxels[key]["normal"])
            newVoxels[newkey]["color"].append(voxels[key]["color"])
        else:
            newVoxels[newkey] = {"normal":[voxels[key]["normal"]], "color":[voxels[key]["color"]]}
    for key in newVoxels.keys():
        if len(newVoxels[key]["normal"]) == 1:
            newVoxels[key]["normal"] = newVoxels[key]["normal"][0]
            newVoxels[key]["color"] = newVoxels[key]["color"][0]
        else:
            newVoxels[key]["normal"] = np.stack(newVoxels[key]["normal"],0).mean(axis=0)
            newVoxels[key]["color"] = np.stack(newVoxels[key]["color"],0).mean(axis=0)
    return newVoxels

def growInEveryDirection(killVoxel):
    keys = tuple(killVoxel.keys())
    for key in keys:
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                for z in [-1,0,1]:
                    newkey = tuple((key[0]+x,key[1]+y,key[2]+z))
                    if newkey in killVoxel:
                        continue
                    else:
                        killVoxel[newkey] = {"normal":killVoxel[key]["normal"],"color":killVoxel[key]["color"]}
                        
def killGrowVoxel(growVoxel,killVoxel):
    for key in killVoxel.keys():
        if key in growVoxel:
            del growVoxel[key]

    #kill of single growVoxels
    killist = []
    for key in growVoxel.keys():
        neighbours = 0
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                for z in [-1,0,1]:
                    newkey = tuple((key[0]+x,key[1]+y,key[2]+z))
                    if newkey in growVoxel:
                        neighbours+=1
        if neighbours < 3:
            killist.append(key)
    for key in killist:
        del growVoxel[key]

def growGrowVoxel(growVoxel, killVoxel, neighbourVoxelarray):
    newGrowVoxel = {}
    #grow the growVoxel
    for key in tuple(growVoxel.keys()):
        neighbours = 0
        xNeighbours = []
        yNeighbours = []
        zNeighbours = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                for z in [-1,0,1]:
                    newkey = tuple((key[0]+x,key[1]+y,key[2]+z))
                    if newkey in growVoxel:
                        neighbours+=1
                        xNeighbours.append(x)
                        yNeighbours.append(y)
                        zNeighbours.append(z)
                    for arr in neighbourVoxelarray:
                        if newkey in arr:
                            neighbours+=1
                            xNeighbours.append(x)
                            yNeighbours.append(y)
                            zNeighbours.append(z)
        if neighbours > 9:
            #grow new growpx in direction of partial neighbours
            xNeighbours = np.array(xNeighbours)
            yNeighbours = np.array(yNeighbours)
            zNeighbours = np.array(zNeighbours)
            planes = [xNeighbours,yNeighbours,zNeighbours]
            growOptions = []
            for plane in planes:
                planeOptions = [0]
                minimal = (plane==-1).sum() > 0
                maximal = (plane==1).sum() > 0
                if minimal:
                    planeOptions.append(-1)
                if maximal:
                    planeOptions.append(1)
                growOptions.append(planeOptions)
            for x in growOptions[0]:
                for y in growOptions[1]:
                    for z in growOptions[2]:
                        newkey = tuple((key[0]+x,key[1]+y,key[2]+z))
                        isUsed = False
                        if newkey in growVoxel:
                            isUsed = True
                        for arr in neighbourVoxelarray:
                                if newkey in arr:
                                    isUsed = True
                        if isUsed:
                            continue
                        else:
                            if newkey in newGrowVoxel:
                                newGrowVoxel[newkey]["normal"].append(growVoxel[key]["normal"])
                                newGrowVoxel[newkey]["color"].append(growVoxel[key]["color"])
                            else:
                                newGrowVoxel[newkey] = {"normal":[growVoxel[key]["normal"]], "color":[growVoxel[key]["color"]]}
    for key in newGrowVoxel.keys():
        if len(newGrowVoxel[key]["normal"]) == 1:
            newGrowVoxel[key]["normal"] = newGrowVoxel[key]["normal"][0]
            newGrowVoxel[key]["color"] = newGrowVoxel[key]["color"][0]
        else:
            newGrowVoxel[key]["normal"] = np.stack(newGrowVoxel[key]["normal"],0).mean(axis=0)
            newGrowVoxel[key]["color"] = np.stack(newGrowVoxel[key]["color"],0).mean(axis=0)

    #kill newGrowVoxels outside of perimeter
    for key in killVoxel.keys():
            if key in newGrowVoxel:
                del newGrowVoxel[key]
    for arr in neighbourVoxelarray:
        for key in arr.keys():
                if key in newGrowVoxel:
                    del newGrowVoxel[key]
    #newGrowVoxel are now growVoxels
    return newGrowVoxel   

def visualizeVoxels(voxelarray, voxSize=0.01):
    pcd = o3d.geometry.PointCloud()
    points = []
    colors = []
    for i,arr in enumerate(voxelarray):
        for key in arr.keys():
            points.append(key)
            colors.append(arr[key]["color"]*1./(1.+i))
    pcd.points = o3d.utility.Vector3dVector(np.stack(points)*voxSize)
    pcd.colors = o3d.utility.Vector3dVector(np.stack(colors))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxSize)
    o3d.visualization.draw_geometries([voxel_grid])
    
    
def killFlankedKillVoxels(killVoxel, neighbourVoxelarray, growVoxel):
    for key in tuple(killVoxel.keys()):
        neighbours = 0
        xNeighbours = []
        yNeighbours = []
        zNeighbours = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                for z in [-1,0,1]:
                    newkey = tuple((key[0]+x,key[1]+y,key[2]+z))
                    if newkey in growVoxel:
                        neighbours+=1
                        xNeighbours.append(x)
                        yNeighbours.append(y)
                        zNeighbours.append(z)
                    for arr in neighbourVoxelarray:
                        if newkey in arr:
                            neighbours+=1
                            xNeighbours.append(x)
                            yNeighbours.append(y)
                            zNeighbours.append(z)
        #grow new growpx in direction of partial neighbours
        flanked = 0
        if -1 in xNeighbours and 1 in xNeighbours:
            flanked+=1
        if -1 in yNeighbours and 1 in yNeighbours:
            flanked+=1
        if -1 in zNeighbours and 1 in zNeighbours:
            flanked+=1
        if flanked > 1:
            del killVoxel[key]

#meshing and creating surfaceVoxels
def vertexKey(n1,n2):
    nbs = [n1,n2]
    return tuple(np.array(nbs).mean(axis=0))

    
def meshIt(voxelFinal, voxSize=0.01):
    vertices = {}
    vertexList = []
    faces = []
    face_normals = []
    face_colors = []

    surfaceVoxels = {}
    for key in voxelFinal:
        normalDirection = np.array([0,0,0])
        voxelThere = False
        surfaceDirections = []
        if tuple((key[0]-1,key[1],key[2])) not in voxelFinal:
            normalDirection[0] -= 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0]-1,key[1],key[2])))
        if tuple((key[0]+1,key[1],key[2])) not in voxelFinal:
            normalDirection[0] += 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0]+1,key[1],key[2])))
        if tuple((key[0],key[1]-1,key[2])) not in voxelFinal:
            normalDirection[1] -= 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0],key[1]-1,key[2])))
        if tuple((key[0],key[1]+1,key[2])) not in voxelFinal:
            normalDirection[1] += 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0],key[1]+1,key[2])))
        if tuple((key[0],key[1],key[2]-1)) not in voxelFinal:
            normalDirection[2] -= 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0],key[1],key[2]-1)))
        if tuple((key[0],key[1],key[2]+1)) not in voxelFinal:
            normalDirection[2] += 1
            voxelThere = True
            surfaceDirections.append(tuple((key[0],key[1],key[2]+1)))
        if voxelThere:
            if abs(normalDirection).sum() > 0:
                surfaceVoxels[key] = {"normal":normalDirection}
                selfKey = np.array(key)
                for direction in surfaceDirections:
                    #two triangle-faces per direction
                    dirKey = np.array(direction)
                    face_normals.append(dirKey-selfKey)
                    face_normals.append(dirKey-selfKey)
                    face_colors.append(voxelFinal[key]["color"])
                    face_colors.append(voxelFinal[key]["color"])
                    variableList = []
                    fixedIdx = -1
                    fixedDirection = 0
                    for i in range(3):
                        if face_normals[-1][i] == 0:
                            variableList.append(i)
                        else: 
                            fixedIdx = i
                            fixedDirection = face_normals[-1][i]
                    #face1vertices
                    vtx1nbr = np.array(key)
                    vtx1nbr[fixedIdx] += fixedDirection
                    vtx1nbr[variableList[0]] += 1
                    vtx1nbr[variableList[1]] += 1

                    vtx2nbr = np.array(key)
                    vtx2nbr[fixedIdx] += fixedDirection
                    vtx2nbr[variableList[0]] += -1
                    vtx2nbr[variableList[1]] += 1

                    vtx3nbr = np.array(key)
                    vtx3nbr[fixedIdx] += fixedDirection
                    vtx3nbr[variableList[0]] += -1
                    vtx3nbr[variableList[1]] += -1

                    #face2vertices
                    Bvtx1nbr = np.array(key)
                    Bvtx1nbr[fixedIdx] += fixedDirection
                    Bvtx1nbr[variableList[0]] += -1
                    Bvtx1nbr[variableList[1]] += -1

                    Bvtx2nbr = np.array(key)
                    Bvtx2nbr[fixedIdx] += fixedDirection
                    Bvtx2nbr[variableList[0]] += 1
                    Bvtx2nbr[variableList[1]] += -1

                    Bvtx3nbr = np.array(key)
                    Bvtx3nbr[fixedIdx] += fixedDirection
                    Bvtx3nbr[variableList[0]] += 1
                    Bvtx3nbr[variableList[1]] += 1

                    vtxIdx = []

                    #getVertexIdx
                    for vtxNbr in [vtx1nbr,vtx2nbr,vtx3nbr,Bvtx1nbr,Bvtx2nbr,Bvtx3nbr]:
                        vtxKey = vertexKey(list(vtxNbr),list(key))
                        if vtxKey not in vertices:
                            #create vtx
                            vertices[vtxKey] = len(vertexList)
                            vertexList.append(list(vtxKey))
                        vtxIdx.append(vertices[vtxKey])
                    #bugfix if face-normals get ignored
                    if face_normals[-1][0] > 0: #x dir
                        faces.append([vtxIdx[0],vtxIdx[1],vtxIdx[2]])
                        faces.append([vtxIdx[3],vtxIdx[4],vtxIdx[5]])
                    elif face_normals[-1][0] < 0: #x dir
                        faces.append([vtxIdx[2],vtxIdx[1],vtxIdx[0]])
                        faces.append([vtxIdx[5],vtxIdx[4],vtxIdx[3]])
                    elif face_normals[-1][1] < 0: #y dir
                        faces.append([vtxIdx[0],vtxIdx[1],vtxIdx[2]])
                        faces.append([vtxIdx[3],vtxIdx[4],vtxIdx[5]])
                    elif face_normals[-1][1] > 0: #y dir
                        faces.append([vtxIdx[2],vtxIdx[1],vtxIdx[0]])
                        faces.append([vtxIdx[5],vtxIdx[4],vtxIdx[3]])
                    elif face_normals[-1][2] > 0: #z dir
                        faces.append([vtxIdx[0],vtxIdx[1],vtxIdx[2]])
                        faces.append([vtxIdx[3],vtxIdx[4],vtxIdx[5]])
                    elif face_normals[-1][2] < 0: #z dir
                        faces.append([vtxIdx[2],vtxIdx[1],vtxIdx[0]])
                        faces.append([vtxIdx[5],vtxIdx[4],vtxIdx[3]])

            else:
                continue
    voxelPoints = np.array(list(surfaceVoxels.keys()))
    bounds = voxelPoints.max(axis=0)
    voxelNormals = np.zeros_like(voxelPoints)

    voxelPoints = np.array(list(surfaceVoxels.keys()))
    bounds = np.stack([voxelPoints.max(axis=0),voxelPoints.min(axis=0)])
    voxelNormals = np.zeros_like(voxelPoints)
    for i,key in enumerate(surfaceVoxels.keys()):
        voxelNormals[i] = surfaceVoxels[key]["normal"]
    return np.array(vertexList)*voxSize,faces,face_normals,face_colors,surfaceVoxels,bounds*voxSize
