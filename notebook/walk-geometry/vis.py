import open3d as o3d
from glob import glob
import time

# 全てのPLYファイルを読み込む
filepaths = glob("./*.ply")
# plyファイルは0~1000の番号がついているので、番号順にソートする
filepaths.sort(key=lambda x: int(x[2:-4]))

meshes = [o3d.io.read_triangle_mesh(filepath) for filepath in filepaths]


# Open3DのVisualizerを初期化
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)
mesh = meshes[0]
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

current_idx = 0
switch_interval = 0.1  # モデルを切り替える間隔（秒）
last_switch_time = time.time()

while True:
    # 経過時間を監視
    if time.time() - last_switch_time > switch_interval:
        # 次のモデルに切り替える
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        # print(cam_params)
        current_idx = (current_idx + 1) % len(meshes)
        vis.remove_geometry(meshes[(current_idx - 1) % len(meshes)])
        mesh = meshes[current_idx]
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

        ctr.convert_from_pinhole_camera_parameters(cam_params)
        # print(cam_params)
        last_switch_time = time.time()

    # 描画を更新
    vis.poll_events()
    vis.update_renderer()