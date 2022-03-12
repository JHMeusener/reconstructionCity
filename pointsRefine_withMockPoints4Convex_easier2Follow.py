# %%

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh as tm
import os
from pathlib import Path
import subprocess
import convexPoints2HRepresentation
import torch
from pointsRefine_emptyUp import meshIt_, vertexKey, makeVoxelDict, voxelGrid2Colors, voxelGrid2Voxels, fillVoxelsFurther2, fillVoxelsFurther, translateOldVoxels2New, fillVoxelMinimalDown, fillVoxelColumns, createVoxelPointsXYDict, appendEmptyPoints, loadPoints, loadEmptyPoints, checkEmptyPoints, appendPoints, appendEmptyPoints, createVoxelPointsXYDict
import voxelstuff
import linetrace
import voxelstuff
import pointCloudOperations

debug = True

pfad = "/home/jhm/Desktop/Arbeit/ConvexNeuralVolume"
vhacdPath = pfad+"/v-hacd/src/build/test/testVHACD"

times = 0

cl = None
pcd = None


breakAll = False

partSize = 6

#################################################################################
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
                print("Loading ",x_mid,y_mid)
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
                            pass
                            #o3d.visualization.draw_geometries([cl])
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
                voxelSize = 8.0
                rangemax = 1
                #create EmptyUP Voxels (5 iterations)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize/2**(rangemax-1))
                gridBounds = voxelGrid2Voxels(voxel_grid)
                voxelGridMin = gridBounds.min(axis=0)
                voxelGridMax = gridBounds.max(axis=0)
                print("     filling columns ")
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize)
                #o3d.visualization.draw_geometries([voxel_grid])
                gridBounds = voxelGrid2Voxels(voxel_grid)
                voxelmap, voxel2D = pointCloudOperations.voxelArrayIn2DMap(gridBounds)
                emptyVox = {}
                minMaxKey = (0,0)
                minMaxZ = 999999
                for key in voxelmap.keys():
                    #only take the voxels that have no empty column neighbours (to prevent stray voxels to count)
                    hasEmptyNeighbour = False
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            if (key[0]+dx,key[1]+dy) not in voxelmap:
                                hasEmptyNeighbour = True
                                break
                    if hasEmptyNeighbour:
                        continue
                    if voxelmap[key]["max"] < minMaxZ:
                        minMaxZ = voxelmap[key]["max"]
                        minMaxKey = key
                    if  voxelmap[key]["max"]-voxelmap[key]["min"]<2:
                        emptyVox[(x,y,voxelmap[key]["min"]+1)] = {"color":np.array([1.,0,0])}
                        emptyVox[(x,y,voxelmap[key]["min"]+2)] = {"color":np.array([1.,0,0])}
                lowestVox = (minMaxKey[0],minMaxKey[1],minMaxZ+1)
                if lowestVox not in emptyVox:
                    emptyVox[lowestVox] = {"color":np.array([1.,0,0])}
                #translate to new resolution
                for i in range(rangemax-1):
                    temp = {}
                    for key in emptyVox.keys():
                        for dx in [0,1]:
                            for dy in [0,1]:
                                for dz in [0,1]:
                                    temp[key[0]+dx,key[1]+dy,key[2]+dz] = emptyVox[key]
                    emptyVox = temp
                emptyVox = np.array(list(emptyVox.keys()))

                #fill columns in every progressive detail
                resolutionFillvoxels = np.array([])
                for i in range(rangemax):
                    resolutionFillvoxels = pointCloudOperations.upSampleArrays2(resolutionFillvoxels)
                    voxSize = voxelSize/(2**(i))
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                                voxel_size=voxSize)
                    gridVoxels = voxelGrid2Voxels(voxel_grid)
                    if len(resolutionFillvoxels > 0):
                        gridVoxels = np.concatenate([resolutionFillvoxels,gridVoxels],axis=0)
                    voxelmap, voxel2dMap = pointCloudOperations.voxelArrayIn2DMap(gridVoxels)
                    resolutionFillvoxels = pointCloudOperations.fillVoxelColumns(voxelmap, voxel2dMap)
                    #o3d.visualization.draw_geometries([pointCloudOperations.input2PointCloud(voxel_grid),pointCloudOperations.input2PointCloud(resolutionFillvoxels)])
                    
                #fill down
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                                voxel_size=voxSize)
                gridVoxels = voxelGrid2Voxels(voxel_grid)
                if len(resolutionFillvoxels > 0):
                        gridVoxels = np.concatenate([resolutionFillvoxels,gridVoxels],axis=0)
                voxelmap, voxel2dMap = pointCloudOperations.voxelArrayIn2DMap(gridVoxels)
                downFillVoxels = []
                minZ = 999999
                for key in voxelmap.keys():
                    if voxelmap[key]["min"] < minZ:
                        minZ = voxelmap[key]["min"]
                minZ = minZ-2
                for key in voxelmap.keys():
                    for z in range(minZ, voxelmap[key]["min"]):
                        downFillVoxels.append((key[0],key[1],z))
                downFillVoxels = np.array(list(downFillVoxels))
                print("     dividing in segments ")
                #o3d.visualization.draw_geometries([pointCloudOperations.input2PointCloud(downFillVoxels),pointCloudOperations.input2PointCloud(voxel_grid),pointCloudOperations.input2PointCloud(resolutionFillvoxels)])
                originalVoxels, _ = pointCloudOperations.voxelGrid2VoxelArray(voxel_grid)
                allVoxels = originalVoxels
                for v in [downFillVoxels,resolutionFillvoxels]:
                    try: allVoxels = np.concatenate([allVoxels,v],axis=0)
                    except: pass
                voxelFinalDict = pointCloudOperations.voxelArray2VoxelDict(allVoxels, np.zeros_like(allVoxels))
                vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = pointCloudOperations.meshIt(voxelFinalDict)                
                #meshing and convex decomposition
                meshedMesh = tm.Trimesh(vertices=vertexList, faces=faces, face_normals=None, vertex_normals=None, face_colors=face_colors, vertex_colors=None, face_attributes=None, vertex_attributes=None, metadata=None, process=True, validate=False, use_embree=True, initial_cache=None, visual=None)
                meshedMesh.export(pfad+"/bla.obj")
                meshparts = pointCloudOperations.divideMesh(meshedMesh, voxelFinalDict, vhacdPath, pfad, iterations=1)
                additionalSurfacePoints = []
                #in higher resolution
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                                voxel_size=voxSize/2)
                for i in range(len(meshparts)):
                    meshparts[i].vertices = meshparts[i].vertices*2
                allVoxels = pointCloudOperations.upSampleArrays2(allVoxels)
                print("     incorporating original points in segments")
                #get normals
                voxelWithNormals = []
                for mesh in meshparts:
                    voxelWithNormals.append(pointCloudOperations.Mesh2VoxelDict(mesh, voxSize=1.0, samplesize = 50000))
                #%%
                #ensure all voxels have a mesh and normal
                oneVoxDict = {}
                for i in range(len(voxelWithNormals)):
                    for v in voxelWithNormals[i].keys():
                        if v not in oneVoxDict:
                            oneVoxDict[v]  = voxelWithNormals[i][v]
                            oneVoxDict[v]["mesh"] = i
                #ensure all original-voxels are present and have a normal as well as having a convexnr
                from sklearn.neighbors import KDTree
                voxIdx = np.array(list(oneVoxDict.keys()))
                kdt = KDTree(voxIdx, leaf_size=30, metric='manhattan')
                for voxel in allVoxels:
                    key = tuple(voxel)
                    if key not in oneVoxDict:
                        ind = kdt.query(voxel[None,:], k=1, return_distance=False)[0]
                        key2 = tuple(voxIdx[ind][0])
                        oneVoxDict[key] = oneVoxDict[key2]
                org_vox, colors = pointCloudOperations.voxelGrid2VoxelArray(voxel_grid)
                for voxel in org_vox:
                    key = tuple(voxel)
                    if key not in oneVoxDict:
                        ind = kdt.query(voxel[None,:], k=1, return_distance=False)[0]
                        key2 = tuple(voxIdx[ind][0])
                        oneVoxDict[key] = oneVoxDict[key2]
                #separate parts again (now they have all voxels)
                partFullVoxelDict = []
                for i in range(len(meshparts)):
                    partFullVoxelDict.append({})
                for voxel in oneVoxDict.keys():
                    partFullVoxelDict[oneVoxDict[voxel]["mesh"]][voxel] = oneVoxDict[voxel]
                # get the original voxels and grow deleteVoxels from them deleting everything in the normal line in a cone
                partOriginalVDict = []
                for i in range(len(meshparts)):
                    partOriginalVDict.append({})
                org_vox, colors = pointCloudOperations.voxelGrid2VoxelArray(voxel_grid)
                for i,voxel in enumerate(org_vox):
                    meshNr = oneVoxDict[tuple(voxel)]["mesh"]
                    partOriginalVDict[meshNr][tuple(voxel)] = oneVoxDict[tuple(voxel)]
                    partOriginalVDict[meshNr][tuple(voxel)]["color"] = colors[i]
                #create alldeleteVoxel Dict
                allDeleteVoxelDict = {}
                for meshNr in range(len(meshparts)):
                    deleteVoxelsLine = voxelstuff.growVoxelsNormal(partOriginalVDict[meshNr], step=4, DirectionSign=1)
                    #complete cone shape
                    deleteVoxels = {}
                    for voxel in deleteVoxelsLine.keys():
                        for dx in range(-int(deleteVoxelsLine[voxel]["step"]*0.51),int(deleteVoxelsLine[voxel]["step"]*0.51)+1):
                            for dy in range(-int(deleteVoxelsLine[voxel]["step"]*0.51),int(deleteVoxelsLine[voxel]["step"]*0.51)+1):
                                for dz in range(-int(deleteVoxelsLine[voxel]["step"]*0.51),int(deleteVoxelsLine[voxel]["step"]*0.51)+1):
                                    deleteVoxels[(voxel[0]+dx,voxel[1]+dy,voxel[2]+dz)] = deleteVoxelsLine[voxel]
                    for voxel in deleteVoxels.keys():
                        allDeleteVoxelDict[voxel] = deleteVoxels[voxel]
                    for voxel in partOriginalVDict[meshNr].keys():
                        allDeleteVoxelDict[voxel] = partOriginalVDict[meshNr][voxel]
                #create part based deletevoxels
                for meshNr in range(len(meshparts)):
                    deleteVoxelsLine = voxelstuff.growVoxelsNormal(partOriginalVDict[meshNr], step=6, DirectionSign=1)
                    #complete cone shape
                    deleteVoxels = {}
                    for voxel in deleteVoxelsLine.keys():
                        for dx in range(-int(deleteVoxelsLine[voxel]["step"]*0.81),int(deleteVoxelsLine[voxel]["step"]*0.81)+1):
                            for dy in range(-int(deleteVoxelsLine[voxel]["step"]*0.81),int(deleteVoxelsLine[voxel]["step"]*0.81)+1):
                                for dz in range(-int(deleteVoxelsLine[voxel]["step"]*0.81),int(deleteVoxelsLine[voxel]["step"]*0.81)+1):
                                    deleteVoxels[(voxel[0]+dx,voxel[1]+dy,voxel[2]+dz)] = deleteVoxelsLine[voxel]
                    for voxel in deleteVoxels.keys():
                        if voxel in partFullVoxelDict[meshNr]: del partFullVoxelDict[meshNr][voxel]
                    # add back one voxel in the direction of the original
                    addVoxelsLine = voxelstuff.growVoxelsNormal(partOriginalVDict[meshNr], step=1, DirectionSign=-1)
                    for voxel in addVoxelsLine.keys():
                        partFullVoxelDict[meshNr][voxel] = addVoxelsLine[voxel]
                    for voxel in partOriginalVDict[meshNr].keys():
                        partFullVoxelDict[meshNr][voxel] = partOriginalVDict[meshNr][voxel]
                    #get additional surface-points
                    vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = pointCloudOperations.meshIt(partFullVoxelDict[meshNr])
                    mesh = tm.Trimesh(vertices=vertexList, faces=faces, face_normals=face_normals, face_colors=face_colors)
                    surfacePoints = mesh.sample(10000)
                    #delete surfacePoints near the original voxels
                    surfacePointVoxel = surfacePoints.copy().astype(int)
                    surfacePointMask = np.ones(len(surfacePoints), dtype=bool)
                    for i,voxel in enumerate(surfacePointVoxel):
                            if tuple(voxel) in partOriginalVDict[meshNr]:
                                        surfacePointMask[i] = False
                            if tuple(voxel) in allDeleteVoxelDict:
                                        surfacePointMask[i] = False
                    additionalSurfacePoints.append(surfacePoints[surfacePointMask])
                print("     sorting points for meshing and texture learning")
                partPoints = []
                partPointsColorLearning = []
                partPointsColorLearningColors = []
                #get mapping parameters
                voxelSize = voxSize/2
                boundMin = org_vox.min(axis=0)
                boundMax = org_vox.max(axis=0)
                CloudMinBound = np.asarray(cl.points).min(axis=0)
                CloudMaxBound = np.asarray(cl.points).max(axis=0)
                #bring pointcloudpoints to voxelarray to get a map
                pointCloudPoints = np.asarray(cl.points)
                #bring to zero
                pointCloudPoints = pointCloudPoints - CloudMinBound
                #normalize
                pointCloudPoints = pointCloudPoints / (CloudMaxBound - CloudMinBound+voxelSize)
                #scale to voxelgrid
                pointCloudPoints = pointCloudPoints * (boundMax - boundMin)
                pointCloudPoints = pointCloudPoints.astype(int)
                pointCloudPartMask = np.zeros(len(pointCloudPoints), dtype=int)-1
                for i,point in enumerate(pointCloudPoints):
                    for meshNr in range(len(meshparts)):
                        if tuple(point) in partFullVoxelDict[meshNr]:
                            pointCloudPartMask[i] = meshNr
                
                # get relevant pointcloudpoints
                for meshNr in range(len(meshparts)):
                    partPoints.append(np.asarray(cl.points)[pointCloudPartMask==meshNr]) 
                # get nearby points for texture-Learning
                for meshNr in range(len(meshparts)):
                    kdt = KDTree(pointCloudPoints[pointCloudPartMask!=meshNr], leaf_size=30, metric='manhattan')
                    ind = kdt.query(additionalSurfacePoints[meshNr].mean(axis=0)[None,:], k=100, return_distance=False)[0]
                    #add those as well as the original points
                    partPointsColorLearning.append(np.concatenate((np.asarray(cl.points)[pointCloudPartMask!=meshNr][ind],np.asarray(cl.points)[pointCloudPartMask==meshNr]),axis=0))
                    partPointsColorLearningColors.append(np.concatenate((np.asarray(cl.colors)[pointCloudPartMask!=meshNr][ind],np.asarray(cl.colors)[pointCloudPartMask==meshNr]),axis=0))
                # map additionalpoints to pointcloud locations
                for meshNr in range(len(meshparts)):
                    vertices = additionalSurfacePoints[meshNr]
                    #bring to zero
                    vertices = vertices - boundMin
                    #normalize
                    vertices = vertices / (boundMax - boundMin)
                    #scale to pointcloud
                    vertices = vertices * (CloudMaxBound - CloudMinBound-voxelSize)
                    #bring to pointcloud
                    vertices = vertices + CloudMinBound + voxelSize/2.0
                    additionalSurfacePoints[meshNr] = vertices
                for meshNr in range(len(meshparts)):
                    np.save(pfad+"/additionalSurfacePoints.npy",additionalSurfacePoints[meshNr])
                    np.save(pfad+"/partPoints.npy",partPoints[meshNr])
                    np.save(pfad+"/partPointsColorLearning.npy",partPointsColorLearning[meshNr])
                    np.save(pfad+"/partPointsColorLearningColors.npy",partPointsColorLearningColors[meshNr])
                    
                    #get views for every part from every direction
                    ## Run vvtool commands
                    p = subprocess.Popen([config['Toolset']['vvtool'],config['RENDER_SEG_PATH']],stdout=subprocess.PIPE)
                    outs, errs = p.communicate()
                    #do vis2mesh reconstruction

                    #do color texture learning

                    #save to file

                    #apply blender subscript 
               

pc = []
for i in range(len(meshparts)):
    pc.append(pointCloudOperations.input2PointCloud(partOriginalVDict[i]))
    pc.append(pointCloudOperations.input2PointCloud(additionalSurfacePoints[i]))
o3d.visualization.draw_geometries(pc)


