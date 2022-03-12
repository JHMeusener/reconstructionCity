import numpy as np
from numpy.core.fromnumeric import shape
import open3d as o3d
import trimesh as tm
import os
from pathlib import Path
import subprocess
import convexPoints2HRepresentation
import trimesh as tm
import torch
import linAlgHelper

def meshVoxelIntersection(part,voxelFinal, voxelsize = 1., solid = False):
    ''' input: part: trimesh mesh, voxelFinal: dict of voxels
    returns: dict of voxels in voxelFinal and mesh(part)'''
    partSurface = {}
    #shoot ray from one side:
    if solid:
        voxelHullIdx = np.array(list(voxelsFromMesh_solid(part, voxelsize = voxelsize)))
    voxelHullIdx = (np.array(part.voxelized(pitch=voxelsize).sparse_indices)+ part.bounds[0,:]+0.5).astype(int)
    voxelMap = {}
    for i in range(len(voxelHullIdx)):
        key = tuple(voxelHullIdx[i][:2])
        if key in voxelMap:
            voxelMap[key].append(voxelHullIdx[i][2])
        else:
            voxelMap[key] = [voxelHullIdx[i][2]]
    for key in voxelMap.keys():
        voxelMap[key] = [min(voxelMap[key]),max(voxelMap[key])]

    partSurface = {}
    for key in voxelMap.keys():
        for i in range(voxelMap[key][0],voxelMap[key][1]):
            key2 = tuple((float(key[0]),float(key[1]),float(i)))
            if key2 in voxelFinal:
                partSurface[key2] = voxelFinal[key2]
    return partSurface

def voxelDifference(original, substractDic):
    ''' input: original     dictionaries of voxels
              substractDic: dictionaries of voxels
        output: set(keys)'''
    keyset1 = set(original.keys())
    keyset2 = set(substractDic.keys())
    voxelDifference = keyset1 - keyset2
    return voxelDifference

def voxelsFromMesh_solid(part, voxelsize = 1.):
    ''' input: part: trimesh mesh
    returns: set of voxels in mesh'''
    #shoot ray from one side:
    voxelHullIdx = (np.array(part.voxelized(pitch=voxelsize).sparse_indices)+ part.bounds[0,:]+0.5).astype(int)
    voxelMap = {}
    for i in range(len(voxelHullIdx)):
        key = tuple(voxelHullIdx[i][:2])
        if key in voxelMap:
            voxelMap[key].append(voxelHullIdx[i][2])
        else:
            voxelMap[key] = [voxelHullIdx[i][2]]
    for key in voxelMap.keys():
        voxelMap[key] = [min(voxelMap[key]),max(voxelMap[key])]
    voxels = set()
    for key in voxelMap.keys():
        for i in range(voxelMap[key][0],voxelMap[key][1]+1):
            key2 = tuple((int(key[0]),int(key[1]),int(i)))
            voxels.add(key2)
    return voxels

#meshing and creating surfaceVoxels
def _vertexKey(n1,n2):
    nbs = [n1,n2]
    return tuple(np.array(nbs).mean(axis=0))
def meshIt(voxelFinal):
    ''' input: voxelFinal: dictionary of voxels to be divided(with key "color":[r,g,b])
    output: vertexList:   np.array of vertex coordinates
            ,faces:       list of vertices for face
            ,face_normals  list of normal direction for a face
            ,face_colors    list of colors for a face
            ,surfaceVoxels  dict of voxels on the surface
            ,bounds         max bounds of the mesh'''
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
                        vtxKey = _vertexKey(list(vtxNbr),list(key))
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
    for i,key in enumerate(surfaceVoxels.keys()):
        voxelNormals[i] = surfaceVoxels[key]["normal"]
    return vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds

def _dividePartList(partList, boollist_dividable, voxelFinal,vhacdPath, pfad):
    dividedParts = []
    dividable = []
    for i,part_ in enumerate(partList):
        #print("dividing original part ",i)
        if boollist_dividable[i]:
            subList = _dividePart(part_, voxelFinal, vhacdPath, pfad)
            #print("   it has ",len(subList),"parts")
            if len(subList) == 1:
                dividedParts.append(subList[0])
                dividable.append(False)
            else:
                for spart in subList:
                    dividedParts.append(spart)
                    dividable.append(True)
        else:
            #print("   not dividable")
            boollist_dividable.append(False)
            dividedParts.append(part_)
    return dividedParts, dividable

def _dividePart(part, voxelFinal, vhacdPath, pfad):
    partSurface = {}
    #shoot ray from one side:
    partSurface = meshVoxelIntersection(part,voxelFinal, solid=True)
    if len(partSurface) == 0:
        print("divide-part: no surface voxels found")
        return [part]
    vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = meshIt(partSurface)
    tempPart_ = tm.Trimesh(vertices=vertexList, faces=faces, face_normals=None, vertex_normals=None, face_colors=face_colors, vertex_colors=None, face_attributes=None, vertex_attributes=None, metadata=None, process=True, validate=False, use_embree=True, initial_cache=None, visual=None)
    tempPart_.export("temp_.obj")
    inputfile = pfad+"/temp_.obj"
    outputfile = pfad+"/bla_vhacd2.obj"
    resolution = 10000 #maximum number of voxels generated during the voxelization stage
    depth = 12 #maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane 	20 	1-32
    concavity =0.35 #maximum concavity 	0.0025 	0.0-1.0
    planeDownsampling = 2 #controls the granularity of the search for the "best" clipping plane 	4 	1-16
    convexhullDownsampling = 1 #controls the precision of the convex-hull generation process during the clipping plane selection stage 	4 	1-16
    alpha = 0.05 #controls the bias toward clipping along symmetry planes 	0.05 	0.0-1.0
    beta = 0.05 #controls the bias toward clipping along revolution axes 	0.05 	0.0-1.0
    gamma = 0.005 #maximum allowed concavity during the merge stage 	0.00125 	0.0-1.0
    pca = 0 #enable/disable normalizing the mesh before applying the convex decomposition 	0 	0-1
    mode = 0 #voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition 	0 	0-1
    maxNumVerticesPerCH = 16 #controls the maximum number of triangles per convex-hull 	64 	4-1024
    minVolumePerCH = 0.01 #controls the adaptive sampling of the generated convex-hulls 	0.0001 	0.0-0.01
    subprocess.call("{} --input '{}' --output '{}' --resolution {} --depth {} --concavity {} --planeDownsampling {} --convexhullDownsampling {} --alpha {} --beta {} --gamma {} --pca {} --mode {} --maxNumVerticesPerCH {} --minVolumePerCH {}".format(vhacdPath,
                    inputfile,outputfile,resolution,depth,concavity, planeDownsampling,convexhullDownsampling,alpha,beta,gamma,pca,mode,maxNumVerticesPerCH,minVolumePerCH), shell=True, stdout=subprocess.PIPE)
    mesh2_temp = tm.load("bla_vhacd2.obj")
    meshes_temp = mesh2_temp.split()
    return meshes_temp

def divideMesh(mesh, voxelOriginal, vhacdPath, pfad, iterations=1):
    ''' input: mesh: trimesh mesh to be divided according to voxelOriginal
                voxelOriginal: dictionary of original voxels
                iterations: number of times to divide the mesh
                voxelFinal: dictionary of final voxels
        output: list of trimesh meshes'''
    iterations = iterations-1
    mesh.export(pfad+"/bla.obj")
    inputfile = pfad+"/bla.obj"
    outputfile = pfad+"/bla_vhacd2.obj"
    resolution = 100000 #maximum number of voxels generated during the voxelization stage
    depth = 32 #maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane 	20 	1-32
    concavity =0.05 #maximum concavity 	0.0025 	0.0-1.0
    planeDownsampling = 4 #controls the granularity of the search for the "best" clipping plane 	4 	1-16
    convexhullDownsampling = 4 #controls the precision of the convex-hull generation process during the clipping plane selection stage 	4 	1-16
    alpha = 0.05 #controls the bias toward clipping along symmetry planes 	0.05 	0.0-1.0
    beta = 0.05 #controls the bias toward clipping along revolution axes 	0.05 	0.0-1.0
    gamma = 0.005 #maximum allowed concavity during the merge stage 	0.00125 	0.0-1.0
    pca = 0 #enable/disable normalizing the mesh before applying the convex decomposition 	0 	0-1
    mode = 0 #voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition 	0 	0-1
    maxNumVerticesPerCH = 16 #controls the maximum number of triangles per convex-hull 	64 	4-1024
    minVolumePerCH = 0.000001 #controls the adaptive sampling of the generated convex-hulls 	0.0001 	0.0-0.01
    subprocess.call("{} --input '{}' --output '{}' --resolution {} --depth {} --concavity {} --planeDownsampling {} --convexhullDownsampling {} --alpha {} --beta {} --gamma {} --pca {} --mode {} --maxNumVerticesPerCH {} --minVolumePerCH {}".format(vhacdPath,
                    inputfile,outputfile,resolution,depth,concavity, planeDownsampling,convexhullDownsampling,alpha,beta,gamma,pca,mode,maxNumVerticesPerCH,minVolumePerCH), shell=True, stdout=subprocess.PIPE)
    mesh2 = tm.load(pfad+"/bla_vhacd2.obj")
    meshes = mesh2.split()
    dividable = []
    for i in range(len(meshes)):
        dividable.append(True)
    for i in range(iterations):
        meshes, dividable = _dividePartList(meshes, dividable, voxelOriginal, vhacdPath=vhacdPath, pfad=pfad)
    return meshes

def voxelDict2VoxelArray(voxelDict):
    ''' input: voxelDict: dictionary of voxels
        output: voxelArray: array of voxels
                colorArray: array of colors'''
    voxelArray = []
    voxelColor = []
    for key in voxelDict:
        voxelArray.append(key)
        voxelColor.append(voxelDict[key]["color"])
    return np.array(voxelArray), np.array(voxelColor)

def voxelGrid2VoxelArray(voxelGrid):
    ''' input: O3d voxelGrid
        output: voxelarray
                colorarray'''
    voxelArray = []
    voxelColor = []
    for voxel in voxelGrid.get_voxels():
        voxelArray.append(voxel.grid_index)
        voxelColor.append(voxel.color)
    return np.array(voxelArray), np.array(voxelColor)

def voxelArray2VoxelDict(voxelArray, voxelColor):
    ''' input: voxelArray: array of voxels
                voxelColor: array of colors
        output: voxelDict: dictionary of voxels'''
    voxelDict = {}
    for i in range(len(voxelArray)):
        voxelDict[(voxelArray[i][0],voxelArray[i][1],voxelArray[i][2])] = {"color":voxelColor[i]}
    return voxelDict
    
def input2PointCloud(input, color=[-1.,-1.,-1.]):
    '''input: voxelDict, (Vertex,Color), Vertex, trimesh mesh, O3d voxelGrid
        output: pointCloud: o3d pointCloud'''
    if type(color) == list:
        color = np.array(color)
    if shape(color) == (3,):
        for i in range(3):
            if color[i] < 0:
                color[i] = np.random.rand()
        color = np.array(color)
    if type(input) == dict:
        voxelArray, voxelColor = voxelDict2VoxelArray(input)
    if type(input) == np.ndarray:
        voxelArray = input
        if color.shape == (3,):
            voxelColor = np.ones(input.shape) * color
    if type(input) == tm.Trimesh:
        voxelArray = input.sample(2000)
        voxelColor = np.ones(voxelArray.shape) * color
    if type(input) == tuple:
        voxelArray = input[0]
        voxelColor = input[1]
    if type(input) == list:
        voxelArray = input[0]
        voxelColor = input[1]
    if type(input) == o3d.geometry.VoxelGrid:
        voxelArray, voxelColor = voxelGrid2VoxelArray(input)
    pointCloud = o3d.geometry.PointCloud()
    if len(voxelArray) == 0:
        return pointCloud
    pointCloud.points = o3d.utility.Vector3dVector(voxelArray)
    pointCloud.colors = o3d.utility.Vector3dVector(voxelColor.astype(np.float64))
    return pointCloud

def mapArrayVertices2Pointcloud(vertices, pointcloud, voxelsizeOfVertices = 0.0):
    '''input: vertices: array of vertices
            pointcloud: o3d pointcloud
        output: vertices: array of vertices'''
    minBound = np.asarray(pointcloud.points).min(axis=0)
    maxBound = np.asarray(pointcloud.points).max(axis=0)
    vertexListMinBounds = vertices.min(axis=0)
    vertexListMaxBounds = vertices.max(axis=0)
    #bring to zero
    vertices = vertices - vertexListMinBounds
    #normalize
    vertices = vertices / (vertexListMaxBounds - vertexListMinBounds)
    #scale to pointcloud
    vertices = vertices * (maxBound - minBound-voxelsizeOfVertices)
    #bring to pointcloud
    vertices = vertices + minBound + voxelsizeOfVertices/2
    return vertices

def convexPoints2HRep(vertices, getPointcloud=False):
    '''input: vertices: array of vertices
    output: hrep    np.array of normals
            middle  np.array of middlepoints'''
    points = torch.tensor(vertices, dtype=torch.float32).cuda()
    minBounds = points.min(dim=0)[0]
    maxBounds = points.max(dim=0)[0]
    middle = (minBounds + maxBounds) / 2
    points = points - middle
    Hrep = convexPoints2HRepresentation.step_withoutNaN(convexPoints2HRepresentation.startPlanes, points)
    rHrep = Hrep.clone()
    rHrep[:,0]=Hrep[:,3]
    rHrep[:,1]=Hrep[:,5]
    rHrep[:,3]=Hrep[:,0]
    rHrep[:,4]=Hrep[:,1]
    rHrep[:,5]= Hrep[:,4]
    maybePointcloud = o3d.geometry.PointCloud()
    if getPointcloud:
        testPoints = torch.rand(15000,3).cuda()*(maxBounds-minBounds)*2.
        testPoints = torch.cat((testPoints,points),0)
        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(testPoints[None,:,:], Hrep[None,:,:])[0]
        inside = boundsTest>0
        completeInner = inside.sum(dim=1)==inside.shape[1]
        insidePoints = testPoints[completeInner] 
        insidePoints = insidePoints[:5000]  + middle
        maybePointcloud = input2PointCloud(insidePoints.cpu().numpy())
    return rHrep.cpu().numpy(), middle.cpu().numpy(), maybePointcloud

def voxelArrayIn2DMap(voxelArray):
    '''Input: voxelArray: array of VoxelIndices
        output: voxelDict_2D_Map     dict with (x,y) as key and "minZ", "maxZ" and "voxels [x,y,z]" as value
                voxel2D               2D-Map array of nr of voxels starting at (0,0)
        '''
    boundsMin = voxelArray.min(axis=0)
    boundsMax = voxelArray.max(axis=0)
    voxel2D = np.zeros((int(boundsMax[0]-boundsMin[0])+1, int(boundsMax[1]-boundsMin[1])+1)).astype(np.int16)
    voxelmap = {}
    for voxel in voxelArray:
        voxel2D[int(voxel[0]-boundsMin[0]), int(voxel[1]-boundsMin[1])] += 1
        key = (voxel[0], voxel[1])
        if key in voxelmap:
            voxelmap[key]["min"] = min(voxelmap[key]["min"],voxel[2])
            voxelmap[key]["max"] = max(voxelmap[key]["max"],voxel[2])
            voxelmap[key]["voxels"].append(voxel)
        else:
            voxelmap[key] = {"min":voxel[2],"max":voxel[2], "voxels":[voxel]}
    return voxelmap, np.array(voxel2D)


def Mesh2VoxelDict(tmObj, voxSize=0.01, samplesize = 1000000):
    '''input: tmObj: Trimesh object
            voxSize: size of voxels
            samplesize: number of samples
    output voxelDict     dict with (x,y,z) as key and "normal" and "color" as values ([dx,dy,dz], [r,g,b])'''
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


def fillVoxelColumns(voxelmap, voxel2dMap):
    '''input: voxelmap: dict with (x,y) as key and "zmin, zmax, voxels" as keys
                voxel2dMap: 2d xy array with nr of voxels
        output: voxelArray of new Voxels
    '''
    fillstart = np.array((voxel2dMap==0).nonzero()).swapaxes(0,1)
    #create fillColumns
    newVoxels = []
    x_max, y_max = voxel2dMap.shape
    notfilledyet = []
    tempFillmap = {}
    for i in range(len(fillstart)):
        x,y = fillstart[i]
        #check n8 neighbourhood
        z_values = []
        for x_check in [max(0,x-1),x,min(x_max,x+1)]:
            for y_check in [max(0,y-1),y,min(y_max,y+1)]:#
                if (x_check,y_check) in voxelmap:
                    if len(voxelmap[(x_check,y_check)]["voxels"]) > 0:
                        for voxel in voxelmap[(x_check,y_check)]["voxels"]:
                            if voxel[2] not in z_values:
                                z_values.append(voxel[2])
        if len(z_values) > 0:
            inz = z_values[0]
            for z_value in z_values:
                    newVoxels.append([x,y,z_value])
                    if (x,y) in tempFillmap:
                        tempFillmap[(x,y)].append(z_value)
                    else:
                        tempFillmap[(x,y)] = [z_value]
        if len(z_values) == 0:
            notfilledyet.append([x,y])
    #fill columns again:
    for z in range(3):
        for coord in notfilledyet:
            if (coord[0],coord[1]) not in tempFillmap:
                z_vals = []
                for x_check in [max(0,coord[0]-1),coord[0],min(x_max,coord[0]+1)]:
                    for y_check in [max(0,coord[1]-1),coord[1],min(y_max,coord[1]+1)]:
                         if (x_check,y_check) in tempFillmap:
                                for z_value in tempFillmap[(x_check,y_check)]:
                                    if z_value not in z_vals:
                                        z_vals.append(z_value)
                                        newVoxels.append([coord[0],coord[1],z_value])
                tempFillmap[(coord[0],coord[1])] = z_vals
    newVoxels = np.array(newVoxels)
    return newVoxels

def upSampleArrays2(voxels, colors = []):
    '''input: voxels: array of voxels
        colors : array of colors
        output; upsampled array of voxels
                (upsampled array of colors)'''
    newVoxels = []
    for voxel in voxels:
        for dx in [1,0]:
            for dy in [1,0]:
                for dz in [1,0]:
                    newVoxels.append([voxel[0]*2+dx,voxel[1]*2+dy,voxel[2]*2+dz])
    if len(colors) > 0:
        for color in colors:
            for dx in [1,0]:
                for dy in [1,0]:
                    for dz in [1,0]:
                        colors.append(color)
    if len(colors) > 0:
        return np.array(newVoxels), np.array(colors)
    else:
        return np.array(newVoxels)

def upSampleVoxelDict(voxeldict):
    '''input: voxeldict with (x,y,z) as input and normal(dx,dy,dz) and color(r,g,b) as value'''
    newVoxels = {}
    for v in voxeldict.keys():
        for dx in [1,0]:
            for dy in [1,0]:
                for dz in [1,0]:
                    newVoxels[(v[0]*2+dx,v[1]*2+dy,v[2]*2+dz)] = voxeldict[v]
    return newVoxels