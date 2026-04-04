# Copyright (c) 2018 Andy Zeng

import numpy as np
from skimage import measure
import trimesh

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self.voxel_size  # truncation on SDF

        
        #######################    Task 2    #######################
        # Compute voxel grid dimensions
        self._vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / self.voxel_size).astype(int)
        self._vol_origin = vol_bnds[:, 0]

        # Initialize TSDF volume and weight volume
        self.tsdf_vol = np.ones(self._vol_dim)
        self.weight_vol = np.zeros(self._vol_dim)
        self.color_vol = np.zeros((*self._vol_dim, 3))

        # Build voxel grid coordinates (N, 3) - integer indices
        xv, yv, zv = np.meshgrid(
            np.arange(self._vol_dim[0]),
            np.arange(self._vol_dim[1]),
            np.arange(self._vol_dim[2]),
            indexing='ij'
        )
        self.vox_coords = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        ############################################################

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1., color_im=None):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. 
        """

        im_h, im_w = depth_im.shape

        #######################    Task 2    #######################
        # Voxel grid integer coords -> world coordinates
        world_pts = self._vol_origin + self.vox_coords * self.voxel_size  # (N, 3)

        # World -> camera coordinates
        world_pts_h = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1))], axis=1)
        cam_pts = (np.linalg.inv(cam_pose) @ world_pts_h.T).T  # (N, 4)
        cam_z = cam_pts[:, 2]

        # Camera -> pixel coordinates
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]
        pix_x = np.round(fx * cam_pts[:, 0] / cam_z + cx).astype(int)
        pix_y = np.round(fy * cam_pts[:, 1] / cam_z + cy).astype(int)

        # Eliminate pixels outside depth image and behind camera
        valid = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (cam_z > 0)
        valid_idx = np.where(valid)[0]

        # Sample depth values
        depth_val = depth_im[pix_y[valid_idx], pix_x[valid_idx]]

        # Filter out invalid depth (zero depth)
        valid_depth = depth_val > 0
        valid_idx = valid_idx[valid_depth]
        depth_val = depth_val[valid_depth]
        ############################################################

        #######################    Task 3    #######################
        # Compute TSDF: min(1.0, (depth - z) / trunc_margin)
        tsdf = np.minimum(1.0, (depth_val - cam_z[valid_idx]) / self.trunc_margin)
        # Only keep voxels within truncation region
        valid_tsdf = tsdf >= -1.0
        valid_idx = valid_idx[valid_tsdf]
        tsdf = tsdf[valid_tsdf]
        ############################################################

        #######################    Task 4    #######################
        # Convert flat indices to 3D voxel indices
        vx = self.vox_coords[valid_idx, 0]
        vy = self.vox_coords[valid_idx, 1]
        vz = self.vox_coords[valid_idx, 2]

        # Weighted average fusion
        w_old = self.weight_vol[vx, vy, vz]
        d_old = self.tsdf_vol[vx, vy, vz]
        w_new = w_old + obs_weight
        self.tsdf_vol[vx, vy, vz] = (w_old * d_old + obs_weight * tsdf) / w_new
        self.weight_vol[vx, vy, vz] = w_new

        # Color fusion (bonus task)
        if color_im is not None:
            # Re-compute valid pixel coords for color sampling
            px = pix_x[valid_idx]
            py = pix_y[valid_idx]
            new_color = color_im[py, px, :]  # (M, 3), BGR
            c_old = self.color_vol[vx, vy, vz]
            self.color_vol[vx, vy, vz] = (w_old[:, None] * c_old + obs_weight * new_color) / w_new[:, None]
        ############################################################


def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False):
    """Get 3D point cloud from depth image and camera pose
    
    Args:
        depth_im (ndarray): Depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
        export_pc (bool): Whether to export pointcloud to a PLY file.
        
    Returns:
        world_pts (ndarray): The 3D point cloud of shape (N, 3).
    """
    
    #######################    Task 1    #######################
    H, W = depth_im.shape
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    # Build pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_im

    # Filter out invalid depth
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    # Pixel -> camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cam_pts = np.stack([x, y, z], axis=-1)  # (N, 3)

    # Camera -> world coordinates
    cam_pts_h = np.concatenate([cam_pts, np.ones((cam_pts.shape[0], 1))], axis=1)  # (N, 4)
    world_pts = (cam_pose @ cam_pts_h.T).T[:, :3]  # (N, 3)
    ############################################################
    
    if export_pc:
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export("pointcloud.ply")
    
    return world_pts
