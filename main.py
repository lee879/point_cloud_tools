import open3d as o3d
from tools.vis import vis, vis_with_spheres, npToO3d
import numpy as np
from tools.trans import PointDataTransform
from tools.se3 import np_inverse

# 设置随机种子以确保结果可重现
np.random.seed(0)

# 初始化点云处理类
pdt = PointDataTransform()

def load_and_visualize_point_cloud():
    """步骤1: 加载数据并建立点云对象"""
    print("=== 步骤1: 加载原始点云数据 ===")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pc_path = "./data/modelnet40/airplane_0054.npy"
    pc_data = np.load(pc_path).reshape(-1, 3)  # shape = (10000,3)
    
    # 创建点云对象并设置颜色
    npToO3d(pcd=pcd, data=pc_data)
    pcd.paint_uniform_color((1, 0, 0))  # 红色
    
    print(f"原始点云形状: {pc_data.shape}")
    
    # 可视化原始点云
    print("显示原始点云...")
    vis([pcd])
    vis_with_spheres([pcd])
    
    return pc_data

def create_source_and_target_point_clouds(pc_data):
    """步骤2: 创建源点云和目标点云"""
    print("\n=== 步骤2: 创建源点云和目标点云 ===")
    
    # 从原始点云中采样两个子集
    pc_data_src, pc_data_remaining = pdt.sample_point_cloud(pc_data, 4096)
    pc_data_ref, _ = pdt.sample_point_cloud(pc_data_remaining, 4096)
    
    # 创建点云对象
    pcd_src = o3d.geometry.PointCloud()
    pcd_ref = o3d.geometry.PointCloud()
    
    npToO3d(pcd=pcd_src, data=pc_data_src)
    npToO3d(pcd=pcd_ref, data=pc_data_ref)
    
    # 设置颜色：源点云蓝色，目标点云红色
    pcd_src.paint_uniform_color((0, 0, 1))  # 蓝色
    pcd_ref.paint_uniform_color((1, 0, 0))   # 红色
    
    print(f"源点云形状: {pc_data_src.shape}")
    print(f"目标点云形状: {pc_data_ref.shape}")
    
    return pc_data_src, pc_data_ref, pcd_src, pcd_ref

def apply_transformation_demo(pc_data_src, pc_data_ref, pcd_src, pcd_ref):
    """步骤3: 演示点云变换"""
    print("\n=== 步骤3: 应用几何变换 ===")
    
    # 生成随机变换矩阵
    transform_matrix = pdt.generate_transform(90, 1)  # 90度旋转，1单位平移
    transform_matrix_inv = np_inverse(transform_matrix)
    
    print(f"变换矩阵形状: {transform_matrix.shape}")
    
    # 对源点云应用变换
    pc_data_src_transformed = pdt.apply_transformation(pc_data_src, transform_matrix_inv)
    pcd_src_transformed = o3d.geometry.PointCloud()
    npToO3d(pcd=pcd_src_transformed, data=pc_data_src_transformed)
    pcd_src_transformed.paint_uniform_color((0, 0, 1))  # 蓝色
    
    print("显示变换后的点云...")
    vis([pcd_src_transformed, pcd_ref])
    
    return pc_data_src_transformed, pcd_src_transformed, transform_matrix

def crop_point_clouds_demo(pc_data_src_transformed, pc_data_ref, pcd_src_transformed, pcd_ref):
    """步骤4: 演示点云切割"""
    print("\n=== 步骤4: 点云切割 ===")
    
    # 对点云进行随机切割（保留70%的点）
    _, pc_data_src_cropped = pdt.crop(pc_data_src_transformed, 0.7)
    _, pc_data_ref_cropped = pdt.crop(pc_data_ref, 0.7)
    
    # 更新点云对象
    npToO3d(pcd=pcd_src_transformed, data=pc_data_src_cropped)
    npToO3d(pcd=pcd_ref, data=pc_data_ref_cropped)
    
    pcd_src_transformed.paint_uniform_color((0, 0, 1))  # 蓝色
    pcd_ref.paint_uniform_color((1, 0, 0))              # 红色
    
    print(f"切割后源点云形状: {pc_data_src_cropped.shape}")
    print(f"切割后目标点云形状: {pc_data_ref_cropped.shape}")
    
    print("显示切割后的点云...")
    vis([pcd_src_transformed, pcd_ref])
    
    return pc_data_src_cropped, pc_data_ref_cropped

def restore_transformation_demo(pc_data_src_cropped, pc_data_ref_cropped, transform_matrix, pcd_src_transformed, pcd_ref):
    """步骤5: 演示变换恢复"""
    print("\n=== 步骤5: 恢复原始变换 ===")
    
    # 将变换后的点云恢复到原始位置
    pc_data_src_restored = pdt.apply_transformation(pc_data_src_cropped, transform_matrix)
    npToO3d(pcd=pcd_src_transformed, data=pc_data_src_restored)
    pcd_src_transformed.paint_uniform_color((0, 0, 1))  # 蓝色
    
    print("显示恢复后的点云...")
    vis([pcd_src_transformed, pcd_ref])
    
    print("点云处理演示完成！")

def main():
    """主函数：演示完整的点云处理流程"""
    print("点云处理教学演示")
    print("=" * 50)
    
    # 步骤1: 加载和可视化原始点云
    pc_data = load_and_visualize_point_cloud()
    
    # 步骤2: 创建源点云和目标点云
    pc_data_src, pc_data_ref, pcd_src, pcd_ref = create_source_and_target_point_clouds(pc_data)
    
    # 步骤3: 应用几何变换
    pc_data_src_transformed, pcd_src_transformed, transform_matrix = apply_transformation_demo(
        pc_data_src, pc_data_ref, pcd_src, pcd_ref
    )
    
    # 步骤4: 点云切割
    pc_data_src_cropped, pc_data_ref_cropped = crop_point_clouds_demo(
        pc_data_src_transformed, pc_data_ref, pcd_src_transformed, pcd_ref
    )
    
    # 步骤5: 恢复变换
    restore_transformation_demo(
        pc_data_src_cropped, pc_data_ref_cropped, transform_matrix, pcd_src_transformed, pcd_ref
    )

if __name__ == "__main__":
    main()