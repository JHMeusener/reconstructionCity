# -%%
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh as tm
import os
from pathlib import Path
import subprocess
import convexPoints2HRepresentation
import torch

debug = True

pfad = "/home/jhm/Desktop/Arbeit/ConvexNeuralVolume"
#pfad = "/files/Code/convexNeuralVolumeData"

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

def checkEmptyPoints(x_block,y_block,z_block):
    my_file = Path(pfad+"/emptyBlocks/e_{}x{}y{}z.npy".format(x_block,y_block,z_block))
    if my_file.is_file():
        return True
    return False

def appendPoints(x_block,y_block,z_block, points):
    my_file = Path(pfad+"/blocks/{}x{}y{}z.npy".format(x_block,y_block,z_block))
    if my_file.is_file():
        #load existing Points and append new ones
        oldpoints = np.load(my_file)
        np.save(my_file,np.concatenate([oldpoints,points],0))
    else:
        np.save(my_file,points)

import os
if len(os.listdir(pfad+"/blocks")) == 0:
    print("creating Blocks")
    import laspy 
    with laspy.open("/out_final_resample_first_return_only.las") as input_las:
        for i, points in enumerate(input_las.chunk_iterator(100000)):
            print("points",i*100000, "to", (i+1)*100000, "of", input_las.header.point_records_count)
            intensity = np.array(points.intensity)
            pointa = np.concatenate([np.array([points.X, points.Y, points.Z]).T/100.,
                           np.array([points.red, points.green, points.blue]).T/float(2**16)],1)[intensity>10000]
            pointDicts = {}
            for p in pointa:
                block = (int(p[0]//35),int(p[1]//35),int(p[2]//35))
                if block in pointDicts:
                    pointDicts[block].append(p)
                else:
                    pointDicts[block] = [p]
            for key in pointDicts.keys():
                appendPoints(key[0],key[1],key[2], np.stack(pointDicts[key],0))


def appendEmptyPoints(x_block,y_block,z_block, points):
    my_file = Path(pfad+"/emptyBlocks/e_{}x{}y{}z.npy".format(x_block,y_block,z_block))
    if my_file.is_file():
        #load existing Points and append new ones
        oldpoints = np.load(my_file)
        np.save(my_file,np.concatenate([oldpoints,points],0))
    else:
        np.save(my_file,points)

def createVoxelPointsXYDict(voxels):
    idx1 = np.argsort(voxels[:,2])
    idx2 = np.argsort(voxels[idx1][:,1],kind='mergesort')
    idx3 = np.argsort(voxels[idx1][idx2][:,0],kind='mergesort')
    voxels=voxels[idx1][idx2][idx3]
    voxelIdx, firstIndex, counts = np.unique(voxels[:,:2],axis=0, return_index=True, return_counts=True)
    voxelmap = {}
    index = 0
    for x in range(int(voxelIdx.max(axis=0)[0]+1)):
        row = {}
        voxelmap[float(x)] = row
        for y in range(int(voxelIdx.max(axis=0)[1]+1)):
            column = {"min":0,"max":0,"voxels":[]}
            row[float(y)] = column
            if index == len(voxelIdx):
                continue
            if voxelIdx[index][0] == x and voxelIdx[index][1] == y:
                minZ = 999999
                maxZ = 0
                for voIdx in range(counts[index]):
                    column["voxels"].append(voxels[firstIndex[index] + voIdx])
                    if column["voxels"][-1][2] < minZ:
                        minZ = column["voxels"][-1][2]
                    if column["voxels"][-1][2] > maxZ:
                        maxZ = column["voxels"][-1][2]
                column["max"] = maxZ
                column["min"] = minZ
                index += 1
                
    return voxelmap, voxelIdx

def fillVoxelColumns(voxelmap, voxelIdx):
    twoDMap = np.zeros(voxelIdx.max(axis=0)+1)
    twoDMap[np.swapaxes(voxelIdx,1,0)[0],np.swapaxes(voxelIdx,1,0)[1]] += 1
    fillstart = np.stack((1-twoDMap).nonzero(),-1)
    #create fillColumns
    newVoxels = []
    x_max, y_max = voxelIdx.max(axis=0)
    notfilledyet = []
    tempFillmap = {}
    for i in range(len(fillstart)):
        x,y = fillstart[i]
        #check n8 neighbourhood
        z_values = []
        for x_check in [max(0,x-1),x,min(x_max,x+1)]:
            for y_check in [max(0,y-1),y,min(y_max,y+1)]:
                if len(voxelmap[x_check][y_check]["voxels"]) > 0:
                    for voxel in voxelmap[x_check][y_check]["voxels"]:
                        if voxel[2] not in z_values:
                            z_values.append(voxel[2])
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

def fillVoxelMinimalDown(voxelmap, voxelIdx):
    x_max, y_max = voxelIdx.max(axis=0)
    newVoxels = []
    for x in voxelmap.keys():
        for y in voxelmap[0].keys():
            if len(voxelmap[x][y]["voxels"]) > 0:
                #we have voxels (so it will get filled by "fillColumns")
                z_start = voxelmap[x][y]["min"]
                #now do like the inverse column fill
                #check neighbourhood
                z_min = z_start
                rangeNr = 11
                for x_check in range(rangeNr):#[max(0,x-1),x,min(x_max,x+1)]:
                    for y_check in range(rangeNr):#[max(0,y-1),y,min(y_max,y+1)]:
                        x_check = min(max(0,x-x_check+rangeNr//2),x_max)
                        y_check = min(max(0,y-y_check+rangeNr//2),y_max)
                        if voxelmap[x_check][y_check]["min"] < z_min:
                            z_min = voxelmap[x_check][y_check]["min"]
                for i in range(int(z_start-z_min)):
                      newVoxels.append([x,y,z_start-1-i]) 
    newVoxels = np.array(newVoxels)
    return newVoxels
                   

def translateOldVoxels2New(oldNewVoxels):
    if len(oldNewVoxels.shape) == 1:
        oldNewVoxels = oldNewVoxels[None,:]
    newVoxels = np.zeros((oldNewVoxels.shape[0]*8,oldNewVoxels.shape[1]))
    for i in range(len(oldNewVoxels)):
        x_old,y_old,z_old = oldNewVoxels[i]
        newVoxels[i*8] = np.array((x_old*2,y_old*2,z_old*2))
        newVoxels[i*8+1] = np.array((x_old*2,y_old*2,z_old*2+1))
        newVoxels[i*8+2] = np.array((x_old*2,y_old*2+1,z_old*2))
        newVoxels[i*8+3] = np.array((x_old*2,y_old*2+1,z_old*2+1))
        newVoxels[i*8+4] = np.array((x_old*2+1,y_old*2+1,z_old*2))
        newVoxels[i*8+5] = np.array((x_old*2+1,y_old*2+1,z_old*2+1))
        newVoxels[i*8+6] = np.array((x_old*2+1,y_old*2,z_old*2))
        newVoxels[i*8+7] = np.array((x_old*2+1,y_old*2,z_old*2+1))
    return newVoxels

def fillVoxelsFurther(newVoxels, oldvoxels):
    '''Fills a neighbour Voxel if there is a filled neighbour in the old scale "newFillMap" and a neighbour in the new map'''
    secondVoxels = []
    x_max, y_max = voxelIdx.max(axis=0)
    for i in range(len(newVoxels)):
        x,y,z = newVoxels[i,:]
        for x_check in [max(0,x-1),x,min(x_max,x+1)]:
            for y_check in [max(0,y-1),y,min(y_max,y+1)]:
                if len(voxelmap[x_check][y_check]["voxels"]) == 0: #not self occupied
                    hasNeighbourInNewMap = False
                    hasNeighbourInOldMap = False
                    nrOfNeighbours = 0
                    for x_check2 in [max(0,x_check-1),x,min(x_max,x_check+1)]:
                        for y_check2 in [max(0,y_check-1),y,min(y_max,y_check+1)]:
                            if len(voxelmap[x_check2][y_check2]["voxels"]) > 0: #has a real neighbour
                                for vx in voxelmap[x_check2][y_check2]["voxels"]:
                                    if vx[2] == z:
                                        secondVoxels.append([x_check,y_check,z])
                                        nrOfNeighbours += 1
    return secondVoxels

def fillVoxelsFurther2(newVoxels):
    '''Fills a neighbour Voxel if there are at least 4 neighbours present'''
    secondVoxels = []
    voxelmap,voxelIdx = createVoxelPointsXYDict(newVoxels)
    x_max, y_max = voxelIdx.max(axis=0)
    for i in range(len(newVoxels)):
        x,y,z = newVoxels[i,:]
        for x_check in [max(0,x-1),x,min(x_max,x+1)]:
            for y_check in [max(0,y-1),y,min(y_max,y+1)]:
                if len(voxelmap[x_check][y_check]["voxels"]) == 0: #not self occupied
                    #start checking n8 for 4 or more neighbours
                    nrOfNeighbours = 0
                    for x_check2 in [max(0,x_check-1),x,min(x_max,x_check+1)]:
                        for y_check2 in [max(0,y_check-1),y,min(y_max,y_check+1)]:
                            if len(voxelmap[x_check2][y_check2]["voxels"]) > 0: #has a real neighbour
                                for vx in voxelmap[x_check2][y_check2]["voxels"]:
                                    if vx[2] == z:
                                        nrOfNeighbours += 1
                    if nrOfNeighbours > 2:
                        secondVoxels.append([x_check,y_check,z])
    return secondVoxels

def voxelGrid2Voxels(voxelgrid):
    voxels = []
    for i in voxelgrid.get_voxels():
        voxels.append(i.grid_index)
    try:
        voxels = np.stack(voxels)
    except: print("no voxels in voxelgrid!")
    return voxels

def voxelGrid2Colors(voxelgrid):
    voxels = []
    for i in voxelgrid.get_voxels():
        voxels.append(i.color)
    voxels = np.stack(voxels)
    return voxels

def makeVoxelDict(array, color=None):
    if color is None:
        color = np.zeros_like(array)
    vdict = {}
    for i in range(len(array)):
        vdict[(array[i][0],array[i][1],array[i][2])] = {"color":color[i]}
    return vdict

#meshing and creating surfaceVoxels
def vertexKey(n1,n2):
    nbs = [n1,n2]
    return tuple(np.array(nbs).mean(axis=0))
def meshIt_(voxelFinal):
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
    for i,key in enumerate(surfaceVoxels.keys()):
        voxelNormals[i] = surfaceVoxels[key]["normal"]
    return vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds

times = 0

cl = None
pcd = None


###############################################
breakAll = False

partSize = 6


if __name__ == "__main__":
    heights = {}
    minBound__ = np.ones(3) * 9999999
    maxBound__ = np.ones(3) * 0
    for name in os.listdir(pfad+"/blocks"):
        x = int(name.split("x")[0])
        y = int(name.split("x")[1].split("y")[0])
        z = int(name.split("x")[1].split("y")[1].split("z")[0])
        if x < minBound__[0]:
            minBound__[0] = x
        if y < minBound__[1]:
            minBound__[1] = y
        if z < minBound__[2]:
            minBound__[2] = z
        if x > maxBound__[0]:
            maxBound__[0] = x
        if y > maxBound__[1]:
            maxBound__[1] = y
        if z > maxBound__[2]:
            maxBound__[2] = z
        if (x,y) in heights:
            heights[(x,y)] = [min(z,heights[(x,y)][0]),max(z,heights[(x,y)][1])]
        else:
            heights[(x,y)] = [z,z]
    np.save(pfad+"/heights.npy", np.array(list(heights.keys())))
    np.save(pfad+"/heightsValues.npy", np.array(list(heights.values())))


    # -%%
    #for x_mid in range(int(maxBound__[0]-minBound__[0])//partSize):
    #    x_mid = x_mid*partSize+int(minBound__[0])+partSize//2
    #    for y_mid in range(int(maxBound__[1]-minBound__[1])//partSize):
    #        y_mid = y_mid*partSize+int(minBound__[1])+partSize//2
            #for z_mid in range(int(maxBound__[2]-minBound__[2])//partSize):
            #    z_mid = z_mid*partSize+int(minBound__[2])
            # we grab all z to have a bottom
    # -%%
    x_mid = 134//6-2
    y_mid = 155//6-3
    for ttt2 in range(1):
        #for x_mid in range(int(maxBound__[0]-minBound__[0])//partSize):
        x_mid = x_mid*partSize+int(minBound__[0])+partSize//2
        for ttt in range(1):
            #for y_mid in range(int(maxBound__[1]-minBound__[1])//partSize):
            y_mid = y_mid*partSize+int(minBound__[1])+partSize//2
            if True:
                print("Doing ",x_mid,y_mid)
                alreadyRefined = False
                points = None
                for x in range(-partSize//2+x_mid,partSize//2+x_mid):
                    for y in range(-partSize//2+y_mid,partSize//2+y_mid):
                        for z in range(int(minBound__[2]),int(maxBound__[2]+1)):
                            try:
                                if checkEmptyPoints(x,y,z):
                                    alreadyRefined = True
                                if alreadyRefined:
                                    continue
                                points_ = loadPoints(x,y,z)
                                if points_ is not None:
                                    try:
                                        points = np.concatenate([points,points_],0)
                                    except:
                                        points = points_
                            except: pass # no points to load here
                if points is not None:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
                        pcd.colors = o3d.utility.Vector3dVector(points[:,3:])
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
                        cl, ind = cl.remove_radius_outlier(nb_points=2, radius=20.05)
                        if debug:
                            o3d.visualization.draw_geometries([cl])
                else:
                        continue
                if len(cl.points) == 0:
                    #no real points make all empty
                    for x in range(-partSize//2+x_mid,partSize//2+x_mid):
                        for y in range(-partSize//2+y_mid,partSize//2+y_mid):
                            for z in range(int(minBound__[2]),int(maxBound__[2]+1)):
                                points_ = loadPoints(x,y,z)
                                if points_ is not None:
                                    appendEmptyPoints(x,y,z, np.array([[x*35.,y*35.,z*35.]]))
                    continue
                if alreadyRefined:
                    continue
                voxelSize = 32.0
                rangemax = 2
                #create EmptyUP Voxels (5 iterations)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize/2**(rangemax-1))
                gridBounds = voxelGrid2Voxels(voxel_grid)
                voxelGridMin = gridBounds.min(axis=0)
                voxelGridMax = gridBounds.max(axis=0)

                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize)
                gridBounds = voxelGrid2Voxels(voxel_grid)
                for i in range(rangemax-1):
                    gridBounds = translateOldVoxels2New(gridBounds)
                voxels = makeVoxelDict(gridBounds-1)
                
                emptyVox = {}
                for i in range(5):
                    for key in voxels.keys():
                        if (key[0],key[1],key[2]+1) in voxels:
                            continue
                        else:
                            emptyVox[(key[0],key[1],key[2]+1)] = {"color":np.array([0,0,0])}
                    for key in emptyVox.keys():
                        voxels[key] = emptyVox[key]
                emptyVox = emptyVox.keys()
                    


                oldVoxels = []
                
                for i in range(rangemax):
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                                voxel_size=voxelSize)
                    voxels = voxelGrid2Voxels(voxel_grid)
                    #print(voxels.max(axis=0))
                    voxelmap,voxelIdx = createVoxelPointsXYDict(voxels)
                    newVoxels = fillVoxelColumns(voxelmap,voxelIdx)
                    if len(oldVoxels) > 0:
                        try:
                            newVoxels = np.concatenate((newVoxels,oldVoxels))
                        except:
                            if len(newVoxels) > len(oldVoxels):
                                pass
                            else:
                                newVoxels = oldVoxels
                    #secondVoxels = fillVoxelsFurther2(newVoxels)
                    #newVoxels = np.concatenate((newVoxels,secondVoxels))
                    newVoxels = np.unique(newVoxels,axis=0)
                    if i < rangemax-1:
                        if len(newVoxels) > 0:
                            newVoxels = translateOldVoxels2New(newVoxels)
                    oldVoxels = newVoxels
                    voxelSize = voxelSize * 0.5
                    
                voxelSize = voxelSize*2
                try:
                    voxelmap,voxelIdx = createVoxelPointsXYDict(np.concatenate((voxels,newVoxels),0))
                except:
                    if len(voxels) > 0: voxelmap,voxelIdx = createVoxelPointsXYDict(voxels)
                    if len(newVoxels) > 0: voxelmap,voxelIdx = createVoxelPointsXYDict(newVoxels)
                newFill = fillVoxelMinimalDown(voxelmap, voxelIdx)
                try:
                    newVoxels = np.concatenate((voxels,newFill,newVoxels),0)
                except:
                    if len(voxels) > 0 and len(newFill) > 0: newVoxels = np.concatenate((voxels,newFill),0)
                    elif len(newVoxels) > 0  and len(newFill) > 0: newVoxels = np.concatenate((newFill,newVoxels),0)
                    elif len(voxels) > 0 and len(newVoxels) > 0: newVoxels = np.concatenate((voxels,newVoxels),0)
                    elif len(voxels) > 0: newVoxels = voxels
                    else: pass
                try:
                    voxelFinal =    makeVoxelDict(np.concatenate((newVoxels,voxels),0))
                except:
                    if len(voxels) > 0: voxelFinal =    makeVoxelDict(voxels)
                    if len(newVoxels) > 0: voxelFinal =    makeVoxelDict(newVoxels)
                try:
                    vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = meshIt_(voxelFinal)
                except: 
                    print("could not mesh this")
                    #continue
                meshedMesh = tm.Trimesh(vertices=vertexList, faces=faces, face_normals=None, vertex_normals=None, face_colors=face_colors, vertex_colors=None, face_attributes=None, vertex_attributes=None, metadata=None, process=True, validate=False, use_embree=True, initial_cache=None, visual=None)
                meshedMesh.export(pfad+"/bla.obj")
                print("exported")
                inputfile = pfad+"/bla.obj"
                outputfile = pfad+"/bla_vhacd2.obj"
                resolution = 200000 #maximum number of voxels generated during the voxelization stage
                depth = 32 #maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane 	20 	1-32
                concavity =0.05 #maximum concavity 	0.0025 	0.0-1.0
                planeDownsampling = 2 #controls the granularity of the search for the "best" clipping plane 	4 	1-16
                convexhullDownsampling = 1 #controls the precision of the convex-hull generation process during the clipping plane selection stage 	4 	1-16
                alpha = 0.05 #controls the bias toward clipping along symmetry planes 	0.05 	0.0-1.0
                beta = 0.05 #controls the bias toward clipping along revolution axes 	0.05 	0.0-1.0
                gamma = 0.005 #maximum allowed concavity during the merge stage 	0.00125 	0.0-1.0
                pca = 0 #enable/disable normalizing the mesh before applying the convex decomposition 	0 	0-1
                mode = 0 #voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition 	0 	0-1
                maxNumVerticesPerCH = 16 #controls the maximum number of triangles per convex-hull 	64 	4-1024
                minVolumePerCH = 0.000001 #controls the adaptive sampling of the generated convex-hulls 	0.0001 	0.0-0.01

                vhacdPath = pfad+"/v-hacd/src/build/test/testVHACD"

                subprocess.call("{} --input '{}' --output '{}' --resolution {} --depth {} --concavity {} --planeDownsampling {} --convexhullDownsampling {} --alpha {} --beta {} --gamma {} --pca {} --mode {} --maxNumVerticesPerCH {} --minVolumePerCH {}".format(vhacdPath,
                                inputfile,outputfile,resolution,depth,concavity, planeDownsampling,convexhullDownsampling,alpha,beta,gamma,pca,mode,maxNumVerticesPerCH,minVolumePerCH), shell=True, stdout=subprocess.PIPE)
                mesh2 = tm.load(pfad+"/bla_vhacd2.obj")
                meshes = mesh2.split()
                
                vertexListMinBounds = voxelGridMin
                vertexListMaxBounds = voxelGridMax
                
                empty = np.array(list(emptyVox),dtype=np.float)
                #bring vertices from grid-coordinates back to pointcloud-coordinates
                minBound = np.asarray(cl.points).min(axis=0)
                maxBound = np.asarray(cl.points).max(axis=0)
                zeroed = np.array(mesh2.vertices)+vertexListMinBounds
                if len(empty) > 0:
                    empty_z = empty+vertexListMinBounds

                zeroed = zeroed/(vertexListMaxBounds-vertexListMinBounds)
                if len(empty) > 0:
                    empty_z = empty_z/(vertexListMaxBounds-vertexListMinBounds)


                mesh2.vertices = zeroed*(maxBound-minBound)+minBound
                if len(empty) > 0:
                    empty_z = empty_z*(maxBound-minBound)+minBound
                if len(meshes) == 0:
                    continue
                meshes = mesh2.split()
                points = torch.zeros((len(meshes),16,3))
                centers = []


                for i in range(len(meshes)):
                    verticesNr = len(meshes[i].vertices)
                    points[i][:verticesNr] = torch.tensor(meshes[i].vertices - meshes[i].bounds.mean(axis=0))
                Hrep = convexPoints2HRepresentation.step_withoutNaN(convexPoints2HRepresentation.startPlanes, points.cuda())
                rHrep = Hrep.clone()
                rHrep[:,0]=Hrep[:,3]
                rHrep[:,1]=Hrep[:,5]
                rHrep[:,3]=Hrep[:,0]
                rHrep[:,4]=Hrep[:,1]
                rHrep[:,5]= Hrep[:,4]
                if debug:
                    #create points for every mesh
                    meshesPoints = []
                    for i,mesh in enumerate(meshes):
                        pointcloudPoints = mesh.sample(2000)
                        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
                        colors = np.ones_like(pointcloudPoints).astype(np.float64)
                        colors[:,2] = colors[:,2]*np.random.rand()
                        colors[:,1] = colors[:,1]*np.random.rand()
                        colors[:,0] = colors[:,0]*np.random.rand()
                        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
                        meshesPoints.append(pointcloudMesh)
                    #visualize empty 
                    pointcloudEmpty = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(empty_z))
                    colors = np.ones_like(empty_z).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    pointcloudEmpty.colors = o3d.utility.Vector3dVector(colors)
                    import linAlgHelper
                    #create pointcloud
                    allpoints = None 
                    for i in range(len(meshes)):
                        pointstemp = (torch.rand((10000,3))*2-1.)*(meshes[i].bounds[1]-meshes[i].bounds[0])[None,:]# + meshes[i].bounds.mean(axis=0)
                        boundsTest = linAlgHelper.getPointDistances2PlaneNormal(pointstemp[None,:,:].cuda().double(), Hrep[i][None,:,:].cuda().double())[0]
                        near = boundsTest>0
                        completeNear = near.sum(dim=1)==near.shape[1]
                        pointstemp = pointstemp[completeNear] + meshes[i].bounds.mean(axis=0)
                        try:
                            allpoints = torch.cat((allpoints,pointstemp),dim=0)
                        except:
                            allpoints = pointstemp
                    pointcloude = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(allpoints))
                    colors = np.ones_like(allpoints).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    pointcloude.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries(meshesPoints + [cl])
                    o3d.visualization.draw_geometries(meshesPoints+[pointcloude])
                    o3d.visualization.draw_geometries([cl]+[pointcloudEmpty])



                starts = []
                for m in meshes:
                    starts.append(m.bounds.mean(axis=0))


                #write empty Blocks (up to 27000 per 30*30*30 volume)
                if len(empty) > 0:
                    pointDicts = {}
                    for p in empty_z:
                        try:
                            block = (int(p[0]//35),int(p[1]//35),int(p[2]//35))
                        except:
                            print("point is NaN")
                            continue
                            if block in pointDicts:
                                pointDicts[block].append(p)
                            else:
                                pointDicts[block] = [p]
                    for key in pointDicts.keys():
                        appendEmptyPoints(key[0],key[1],key[2], np.stack(pointDicts[key],0))

                #write startVolumes
                for i in range(len(rHrep)):
                    try:
                        my_file = Path(pfad+"/neuralVolumes/{}x{}y{}z_{}.npy".format(int(starts[i][0]//35),
                                                                                int(starts[i][1]//35),
                                                                                int(starts[i][2]//35),i))
                        np.save(my_file,np.concatenate([starts[i][None,:],rHrep[i].cpu()],0))
                    except:
                        print("NaN in Hrep")
                        continue
                #save refined Points
                refPoints = np.concatenate([np.asarray(cl.points),np.asarray(cl.colors)],1)

                pointDicts = {}
                for p in refPoints:
                    try:
                        block = (int(p[0]//35),int(p[1]//35),int(p[2]//35))
                        if block in pointDicts:
                            pointDicts[block].append(p)
                        else:
                            pointDicts[block] = [p]
                    except:
                        print("NaN in P")
                    continue
                for key in pointDicts.keys():
                    my_file = Path(pfad+"/blocks/{}x{}y{}z.npy".format(key[0],key[1],key[2]))
                    np.save(my_file,np.stack(pointDicts[key],0))


    #- %%
