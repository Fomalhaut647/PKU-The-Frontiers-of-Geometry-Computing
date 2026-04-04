"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
from tqdm import tqdm 

import trimesh
from skimage import measure
import fusion


if __name__ == "__main__":

    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    
    # vol_bnds[:, 0]: minimum bounds of the voxel volume along x, y, z
    # vol_bnds[:, 1]: maximum bounds of the voxel volume along x, y, z
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf
    
    for i in tqdm(range(n_imgs)):
        
        # Read depth image 
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i), -1).astype(float)
        # depth is saved in 16-bit PNG in millimeters
        depth_im /= 1000.  
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0  
        
        # Read camera pose, a 4x4 rigid transformation matrix
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  
        
        #######################    Task 1    #######################
        view_frust_pts = fusion.cam_to_world(
            depth_im, cam_intr, cam_pose,
            export_pc=(i == 0)
        )
        # Update voxel volume bounds
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], view_frust_pts.min(axis=0))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], view_frust_pts.max(axis=0))
        ############################################################

    print("Volume bounds:", vol_bnds)

    # Initialize TSDF voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in tqdm(range(n_imgs)):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

        # Read color image
        color_im = cv2.imread("data/frame-%06d.color.jpg"%(i))  # BGR

        # Integrate observation into voxel volume (with color)
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1., color_im=color_im)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    #######################    Task 4    #######################
    # Extract mesh using Marching Cubes
    print("Extracting mesh with Marching Cubes...")
    verts, faces, normals, _ = measure.marching_cubes(tsdf_vol.tsdf_vol, level=0)
    # Convert voxel indices to world coordinates
    verts = verts * tsdf_vol.voxel_size + tsdf_vol._vol_origin

    # Sample vertex colors from color volume
    verts_idx = np.round((verts - tsdf_vol._vol_origin) / tsdf_vol.voxel_size).astype(int)
    verts_idx = np.clip(verts_idx, 0, np.array(tsdf_vol._vol_dim) - 1)
    vert_colors = tsdf_vol.color_vol[verts_idx[:, 0], verts_idx[:, 1], verts_idx[:, 2]]
    # BGR -> RGB and convert to uint8
    vert_colors = vert_colors[:, ::-1].astype(np.uint8)

    # Save mesh as PLY
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           vertex_colors=vert_colors)
    mesh.export("mesh.ply")
    print("Mesh saved to mesh.ply")

    # Also save a version without color
    mesh_no_color = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh_no_color.export("mesh_no_color.ply")
    print("Mesh (no color) saved to mesh_no_color.ply")
    ############################################################

