import open3d as o3d
import numpy as np

def npToO3d(pcd,data):
    if data.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(data)
        return pcd
    else:
        print("Point cloud should be 3D, check dimensions.")


def vis(pcdlist, small_cube_linesets=[], point_show_normal=False, indx="0"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'{indx}', width=1920, height=1080, left=50, top=50)
    for pcd in pcdlist:
        vis.add_geometry(pcd)
    for lineset in small_cube_linesets:
        vis.add_geometry(lineset)

    render_option = vis.get_render_option()
    render_option.point_size = 6.0
    render_option.background_color = np.asarray([255, 255, 255])
    render_option.light_on = True
    vis.run()
    vis.destroy_window()



def vis_with_spheres(pcdlist, small_cube_linesets=[], indx="0", sphere_radius=0.008, screenshot_path=None):
    import open3d as o3d
    import numpy as np
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'{indx}', width=2048, height=1286, left=50, top=50)
    for lineset in small_cube_linesets:
        vis.add_geometry(lineset)


    sphere_template = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere_template.compute_vertex_normals()

    merged_mesh = o3d.geometry.TriangleMesh()
    for pcd in pcdlist:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        for point, color in zip(points, colors):

            sphere = sphere_template.translate(point, relative=False)
            vertex_colors = np.tile(color, (np.asarray(sphere.vertices).shape[0], 1))
            sphere.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


            merged_mesh += sphere

    vis.add_geometry(merged_mesh)
    render_option = vis.get_render_option()
    render_option.light_on = True

    vis.run()
    if screenshot_path:
        vis.capture_screen_image(screenshot_path)
    vis.destroy_window()