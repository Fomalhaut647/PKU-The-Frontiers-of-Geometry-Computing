"""Render point cloud and mesh results for the report."""
import numpy as np
import trimesh
import cv2

def render_trimesh_views(mesh_or_pc, filename_prefix, is_pointcloud=False):
    """Render a trimesh object from multiple views and save as images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if is_pointcloud:
        pts = np.array(mesh_or_pc.vertices)
        # Subsample for visualization
        if len(pts) > 50000:
            idx = np.random.choice(len(pts), 50000, replace=False)
            pts = pts[idx]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1, c=pts[:, 2], cmap='viridis', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud (Frame 0)')
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename_prefix}.png")
    else:
        verts = np.array(mesh_or_pc.vertices)
        faces = np.array(mesh_or_pc.faces)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample faces for rendering speed
        if len(faces) > 100000:
            idx = np.random.choice(len(faces), 100000, replace=False)
            faces_sub = faces[idx]
        else:
            faces_sub = faces

        # Get vertex colors if available
        if mesh_or_pc.visual and hasattr(mesh_or_pc.visual, 'vertex_colors') and mesh_or_pc.visual.vertex_colors is not None:
            vc = np.array(mesh_or_pc.visual.vertex_colors)[:, :3] / 255.0
            face_colors = vc[faces_sub].mean(axis=1)
            poly = Poly3DCollection(verts[faces_sub], alpha=0.8)
            poly.set_facecolor(face_colors)
        else:
            poly = Poly3DCollection(verts[faces_sub], alpha=0.6)
            poly.set_facecolor('steelblue')

        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

        # Set axis limits
        ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
        ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
        ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.savefig(f'{filename_prefix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename_prefix}.png")


if __name__ == "__main__":
    # Also save a sample depth image and color image for the report
    depth_im = cv2.imread("data/frame-000000.depth.png", -1).astype(float)
    depth_im /= 1000.0
    depth_im[depth_im == 65.535] = 0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(depth_im, cmap='plasma')
    ax.set_title('Depth Image (Frame 0)')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    plt.tight_layout()
    plt.savefig('report_depth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved report_depth.png")

    color_im = cv2.imread("data/frame-000000.color.jpg")
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(color_im)
    ax.set_title('Color Image (Frame 0)')
    plt.tight_layout()
    plt.savefig('report_color.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved report_color.png")

    # Render point cloud
    print("Loading point cloud...")
    pc = trimesh.load("pointcloud.ply")
    render_trimesh_views(pc, 'report_pointcloud', is_pointcloud=True)

    # Render mesh without color
    print("Loading mesh (no color)...")
    mesh = trimesh.load("mesh_no_color.ply")
    render_trimesh_views(mesh, 'report_mesh', is_pointcloud=False)

    # Render mesh with color
    print("Loading mesh (with color)...")
    mesh_color = trimesh.load("mesh.ply")
    render_trimesh_views(mesh_color, 'report_mesh_color', is_pointcloud=False)

    print("All visualizations saved.")
