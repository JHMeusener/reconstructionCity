#todo

#mehrfach v-hacd
#voxel growing
#surface-voxel freistellen
#nahe richtiger punkte alles löschen
#übriggebliebene surface-voxel als mock-punkte nehmen

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
from pointsRefine_emptyUp import meshIt_, vertexKey, makeVoxelDict, voxelGrid2Colors, voxelGrid2Voxels, fillVoxelsFurther2, fillVoxelsFurther, translateOldVoxels2New, fillVoxelMinimalDown, fillVoxelColumns, createVoxelPointsXYDict, appendEmptyPoints, loadPoints, loadEmptyPoints, checkEmptyPoints, appendPoints, appendEmptyPoints, createVoxelPointsXYDict
import voxelstuff
import linetrace

debug = True

pfad = "/home/jhm/Desktop/Arbeit/ConvexNeuralVolume"
vhacdPath = pfad+"/v-hacd/src/build/test/testVHACD"

times = 0

cl = None
pcd = None


breakAll = False

partSize = 6

def voxelsFromMesh(part,voxelFinal, voxelsize = 1.):
    partSurface = {}
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

    partSurface = {}
    for key in voxelMap.keys():
        for i in range(voxelMap[key][0],voxelMap[key][1]):
            key2 = tuple((float(key[0]),float(key[1]),float(i)))
            if key2 in voxelFinal:
                partSurface[key2] = voxelFinal[key2]
    return partSurface

def voxelsFromMesh_solid(part, voxelsize = 1.):
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
            

def dividePartList(partList, boollist_dividable, voxelFinal):
    dividedParts = []
    dividable = []
    for i,part_ in enumerate(partList):
        #print("dividing original part ",i)
        if boollist_dividable[i]:
            subList = dividePart(part_, voxelFinal)
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

def dividePart(part, voxelFinal):
    partSurface = {}
    #shoot ray from one side:
    partSurface = voxelsFromMesh(part,voxelFinal)
    if len(partSurface) == 0:
        print("divide-part: no surface voxels found")
        return [part]
    vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = meshIt_(partSurface)
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


def createVoxelPointsXYDict2(voxels):
    voxelmap = {}
    for voxel in voxels:
        key = (voxel[0], voxel[1])
        if key in voxelmap:
            voxelmap[key]["min"] = min(voxelmap[key]["min"],voxel[2])
            voxelmap[key]["max"] = max(voxelmap[key]["max"],voxel[2])
        else:
            voxelmap[key] = {"min":voxel[2],"max":voxel[2]}
    return voxelmap


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
                voxelSize = 16.0
                rangemax = 2
                #create EmptyUP Voxels (5 iterations)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize/2**(rangemax-1))
                gridBounds = voxelGrid2Voxels(voxel_grid)
                voxelGridMin = gridBounds.min(axis=0)
                voxelGridMax = gridBounds.max(axis=0)

                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize)
                o3d.visualization.draw_geometries([voxel_grid])
                gridBounds = voxelGrid2Voxels(voxel_grid)
                voxelmap = createVoxelPointsXYDict2(gridBounds)
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
                # we have voxelfinal
                try:
                    vertexList,faces,face_normals,face_colors,surfaceVoxels,bounds = meshIt_(voxelFinal)
                except: 
                    print("could not mesh this")
                    #continue
                
                #meshing and convex decomposition
                meshedMesh = tm.Trimesh(vertices=vertexList, faces=faces, face_normals=None, vertex_normals=None, face_colors=face_colors, vertex_colors=None, face_attributes=None, vertex_attributes=None, metadata=None, process=True, validate=False, use_embree=True, initial_cache=None, visual=None)
                meshedMesh.export(pfad+"/bla.obj")
                print("exported")
                if debug:
                    pointcloudPoints = meshedMesh.sample(10000)
                    pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
                    colors = np.ones_like(pointcloudPoints).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
                    pointcloudorig = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(list(voxelFinal.keys())))
                    colors = np.ones_like(pointcloudPoints).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    pointcloudorig.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries([pointcloudMesh,pointcloudorig])
                inputfile = pfad+"/bla.obj"
                outputfile = pfad+"/bla_vhacd2.obj"
                resolution = 500000 #maximum number of voxels generated during the voxelization stage
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
                subprocess.call("{} --input '{}' --output '{}' --resolution {} --depth {} --concavity {} --planeDownsampling {} --convexhullDownsampling {} --alpha {} --beta {} --gamma {} --pca {} --mode {} --maxNumVerticesPerCH {} --minVolumePerCH {}".format(vhacdPath,
                                inputfile,outputfile,resolution,depth,concavity, planeDownsampling,convexhullDownsampling,alpha,beta,gamma,pca,mode,maxNumVerticesPerCH,minVolumePerCH), shell=True, stdout=subprocess.PIPE)
                mesh2 = tm.load(pfad+"/bla_vhacd2.obj")
                meshes = mesh2.split()
                if debug:
                    meshClouds = []
                    for mesh in meshes:
                        pointcloudPoints = mesh.sample(2000)
                        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
                        colors = np.ones_like(pointcloudPoints).astype(np.float64)
                        colors[:,2] = colors[:,2]*np.random.rand()
                        colors[:,1] = colors[:,1]*np.random.rand()
                        colors[:,0] = colors[:,0]*np.random.rand()
                        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
                        meshClouds.append(pointcloudMesh)
                    pointcloudorig = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(list(voxelFinal.keys())))
                    colors = np.ones_like(pointcloudPoints).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    pointcloudorig.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries(meshClouds+[pointcloudorig])
                ####################################################################DIVIDE further
                #create dividablelist
                #dividable = []
                #for i in range(len(meshes)):
                #    dividable.append(True)
                ##divide
                #for i in range(1):
                #    meshes, dividable = dividePartList(meshes, dividable, voxelFinal)
                ####################################################################### get Voxels in Volume
                #################### get voxels in all Meshparts
                solidVolueVoxels = {}
                extraVolumeVoxelSize = 16.
                for mesh in meshes:
                    partVoxels = voxelsFromMesh_solid(mesh, voxelsize=1.)
                    for voxel in partVoxels:
                        solidVolueVoxels[voxel] = {"color":np.array([0.,0.,0.])}
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=extraVolumeVoxelSize)
                origVoxels = voxelGrid2Voxels(voxel_grid)
                #for i in range(rangemax-1):
                #    gridBounds = translateOldVoxels2New(gridBounds)
                origVoxels = makeVoxelDict(origVoxels-1)
                #################### get the lowest 2 emptyvoxels
                if len(emptyVox) > 0:
                    minimumZempty = np.array(list(emptyVox))[:,2].min(axis=0)
                relevantEmptyVoxels = set()
                for vox in emptyVox:
                    if vox[2] < minimumZempty+2:
                        relevantEmptyVoxels.add(vox)
                if debug:
                    origpoints = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array(list(origVoxels.keys()))))
                    colors = np.ones_like(np.array(list(origVoxels))).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    origpoints.colors = o3d.utility.Vector3dVector(colors)
                    meshClouds = []
                    for mesh in meshes:
                        partVoxels = voxelsFromMesh_solid(mesh, voxelsize=1.)
                        partvoxpoints = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array(list(partVoxels))))
                        colors = np.ones_like(np.array(list(partVoxels))).astype(np.float64)
                        colors[:,2] = colors[:,2]*np.random.rand()
                        colors[:,1] = colors[:,1]*np.random.rand()
                        colors[:,0] = colors[:,0]*np.random.rand()
                        partvoxpoints.colors = o3d.utility.Vector3dVector(colors)
                        meshClouds.append(partvoxpoints)
                    if len(emptyVox) > 0:
                        emptypoints = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array(list(relevantEmptyVoxels))))
                        colors = np.ones_like(np.array(list(relevantEmptyVoxels))).astype(np.float64)
                        colors[:,2] = colors[:,2]*0.
                        colors[:,1] = colors[:,1]*0.
                        colors[:,0] = colors[:,0]*1.
                        emptypoints.colors = o3d.utility.Vector3dVector(colors)
                        o3d.visualization.draw_geometries(meshClouds+[origpoints,emptypoints])
                    else:
                        o3d.visualization.draw_geometries(meshClouds+[origpoints])
                #################### for every volumevoxel: if a line can be traced towards an emptyvoxel without encountering an originalvoxel - kill it

                #################### kill outermost volumevoxels
                #### do finer voxelization with original+volumevoxels
                #### cut voxels by part-volumes and get surfaces - if it was an "original" -voxel: do nothing ---- else: these are backside points

                partSurfaces = []
                for mesh in meshes:
                    partSurfaces.append(voxelsFromMesh(mesh,voxelFinal))
                ############ get non-original voxels
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cl,
                                                                            voxel_size=voxelSize)
                voxels = voxelGrid2Voxels(voxel_grid)
                voxels = makeVoxelDict(voxels)
                nonOriginSurfacePoints = []
                for meshNr in range(len(meshes)):
                    nonOriginSurfacePoints.append({})
                    for key in partSurfaces[meshNr].keys():
                        if key not in voxels:
                            nonOriginSurfacePoints[-1][key] = partSurfaces[meshNr][key]
                ############ add those points to the volumepoints
                vertexListMinBounds = voxelGridMin
                vertexListMaxBounds = voxelGridMax
                empty = np.array(list(emptyVox),dtype=np.float)
                #bring vertices from grid-coordinates back to pointcloud-coordinates
                minBound = np.asarray(cl.points).min(axis=0)
                maxBound = np.asarray(cl.points).max(axis=0)
                volumeSurfacePoints = []
                for meshNr in range(len(meshes)):
                    volumeSurfaceColors_ = []
                    nonOriginSurfaceP = np.array(list(nonOriginSurfacePoints[meshNr].keys()))
                    if len(nonOriginSurfaceP) == 0:
                        continue
                    zeroed = nonOriginSurfaceP + vertexListMinBounds
                    normed = zeroed / (vertexListMaxBounds - vertexListMinBounds)
                    nonOriginSurfaceP = normed * (maxBound - minBound) + minBound
                    volumeSurfacePoints.append(nonOriginSurfaceP)
                zeroed = []
                for mesh in meshes:
                    zeroed.append(np.array(mesh.vertices)+vertexListMinBounds)
                if len(empty) > 0:
                    empty_z = empty+vertexListMinBounds
                for meshNr in range(len(zeroed)):
                    zeroed[meshNr] = zeroed[meshNr]/(vertexListMaxBounds-vertexListMinBounds)
                if len(empty) > 0:
                    empty_z = empty_z/(vertexListMaxBounds-vertexListMinBounds)
                for meshnr in range(len(meshes)):
                    meshes[meshnr].vertices = zeroed[meshnr]*(maxBound-minBound)+minBound
                if len(empty) > 0:
                    empty_z = empty_z*(maxBound-minBound)+minBound
                if len(meshes) == 0:
                    continue
                if debug:
                    emptyzpoints = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(empty_z))
                    colors = np.ones_like(empty_z).astype(np.float64)
                    colors[:,2] = colors[:,2]*np.random.rand()
                    colors[:,1] = colors[:,1]*np.random.rand()
                    colors[:,0] = colors[:,0]*np.random.rand()
                    emptyzpoints.colors = o3d.utility.Vector3dVector(colors)
                    meshClouds = []
                    for mesh in meshes:
                        pointcloudPoints = mesh.sample(2000)
                        pointcloudMesh = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloudPoints))
                        colors = np.ones_like(pointcloudPoints).astype(np.float64)
                        colors[:,2] = colors[:,2]*np.random.rand()
                        colors[:,1] = colors[:,1]*np.random.rand()
                        colors[:,0] = colors[:,0]*np.random.rand()
                        pointcloudMesh.colors = o3d.utility.Vector3dVector(colors)
                        meshClouds.append(pointcloudMesh)
                    o3d.visualization.draw_geometries(meshClouds+[cl])
                    o3d.visualization.draw_geometries(meshClouds+[emptyzpoints])
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
                        additionalPointsPath = Path(pfad+"neuralVolumesAdditionalPoints//{}x{}y{}z_{}.npy".format(int(starts[i][0]//35),
                                                                                int(starts[i][1]//35),
                                                                                int(starts[i][2]//35),i)) 
                        np.save(additionalPointsPath,np.concatenate([volumeSurfacePoints[i],volumeSurfaceColors[i]],0))
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
