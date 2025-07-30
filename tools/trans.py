import random
import numpy as np
import open3d as o3d
from tools import se3
from scipy.spatial import cKDTree

class PointDataTransform:
    def preprocess_point_cloud(self,point_cloud, rotation_range=np.pi * 0.5):
        centroid = np.mean(point_cloud, axis=0)

        point_cloud_centered = point_cloud - centroid

        angle = np.random.uniform(-rotation_range, rotation_range)

        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)  

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis
        one_minus_cos = 1 - cos_angle
        rotation_matrix = np.array([
            [cos_angle + ux ** 2 * one_minus_cos, ux * uy * one_minus_cos - uz * sin_angle,
             ux * uz * one_minus_cos + uy * sin_angle],
            [uy * ux * one_minus_cos + uz * sin_angle, cos_angle + uy ** 2 * one_minus_cos,
             uy * uz * one_minus_cos - ux * sin_angle],
            [uz * ux * one_minus_cos - uy * sin_angle, uz * uy * one_minus_cos + ux * sin_angle,
             cos_angle + uz ** 2 * one_minus_cos]
        ])

        processed_point_cloud = np.dot(point_cloud_centered, rotation_matrix.T)

        processed_point_cloud += centroid

        return processed_point_cloud

    def sample_point_cloud(self,point_cloud, num_samples):

        np.random.shuffle(point_cloud)

        if num_samples > point_cloud.shape[0]:
            raise ValueError("error")

        random_indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
        remaining_point_cloud = np.delete(point_cloud, random_indices, axis=0)

        sampled_point_cloud = point_cloud[random_indices, :]

        return sampled_point_cloud, remaining_point_cloud


    def uniform_2_sphere(self, num = None):
        if num is not None:
            phi = np.random.uniform(0.0, 2 * np.pi, num)
            cos_theta = np.random.uniform(-1.0, 1.0, num)
        else:
            phi = np.random.uniform(0.0, 2 * np.pi)
            cos_theta = np.random.uniform(-1.0, 1.0)

        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.stack((x, y, z), axis=-1)

    def normalize_point_cloud(self, pc):
        min_vals = np.min(pc, axis=0)
        max_vals = np.max(pc, axis=0)

        scale_factors = max_vals - min_vals

        normalized_pc = (pc - min_vals) / (scale_factors + 1e-8)

        return normalized_pc, min_vals, scale_factors

    def crop(self, points, p_keep, rand_xyz=None):
        if p_keep == 1.0:
            mask = np.ones(shape=(points.shape[0],), dtype=bool)
        else:
            if rand_xyz is None:
                rand_xyz = self.uniform_2_sphere()  
            centroid = np.mean(points[:, :3], axis=0)  
            points_centered = points[:, :3] - centroid  
            dist_from_plane = np.dot(points_centered, rand_xyz)  

            if p_keep == 0.5:
                mask = dist_from_plane > 0  
            else:
                mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100) 

        cut_part = points[~mask] 
        remaining_part = points[mask]  

        return cut_part, remaining_part

    def rand_scal(self, sample,s0,s1):

        integer_value = random.choice([0, 1, 2])
        float_value = random.uniform(s0,s1)

        sample["points_src"][..., integer_value] = sample["points_src"][..., integer_value] * float_value
        sample["points_ref"][..., integer_value] = sample["points_ref"][..., integer_value] * float_value
        sample["points_ref_copy"][..., integer_value] = sample["points_ref_copy"][..., integer_value] * float_value

        return sample

    def generate_transform(self,rot_mag,trans_mag,t=True):
        if t:
            anglex = np.random.uniform(low= -1,high= 1) * np.pi * rot_mag / 180.0
            angley = np.random.uniform(low= -1,high= 1) * np.pi * rot_mag / 180.0
            anglez = np.random.uniform(low= -1,high= 1) * np.pi * rot_mag / 180.0
        else:
            rand_val_x = np.random.choice([-1, 1]) * np.random.uniform(low=0.5, high=1)
            rand_val_y = np.random.choice([-1, 1]) * np.random.uniform(low=0.5, high=1)
            rand_val_z = np.random.choice([-1, 1]) * np.random.uniform(low=0.5, high=1)

            anglex = rand_val_x * np.pi * rot_mag / 180.0
            angley = rand_val_y * np.pi * rot_mag / 180.0
            anglez = rand_val_z * np.pi * rot_mag / 180.0


        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3

    def restore_rotation(self,rand_SE3):

        R_ab = rand_SE3[:3, :3]

        return R_ab

    def apply_transformation(self,xyz, pose,only_r=False):

        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]

        if not only_r:
            transformed_points = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t
        else:
            transformed_points = np.einsum('...ij,...bj->...bi', rot, xyz)

        return transformed_points




    def _resample(self, points, k):
        if k < points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([
                np.random.choice(points.shape[0], points.shape[0], replace=False),
                np.random.choice(points.shape[0], k - points.shape[0], replace=True)
            ])
            return points[rand_idxs, :]

    def _jitter(self, pts, size,scale,clip0,clip1):
        pts = pts.reshape(-1, 3)
        noise = np.clip(np.random.normal(0.0, scale, size=(pts.shape[0], 3)),
                        a_min=-clip0,
                        a_max=clip1)
        pts[:, :3] += noise

        return pts.reshape(size)


    def adjust_point_clouds(self, sample,target_size=None):
        points_ref = sample["points_ref"]
        points_src = sample["points_src"]

  
        if points_ref.shape[0] > target_size:
            points_ref = points_ref[np.random.choice(points_ref.shape[0], target_size, replace=False)]


        if points_src.shape[0] > target_size:
            points_src = points_src[np.random.choice(points_src.shape[0], target_size, replace=False)]

        if points_ref.shape[0] < target_size:
            points_ref = np.vstack(
                [points_ref, points_ref[np.random.choice(points_ref.shape[0], target_size - points_ref.shape[0])]])

        if points_src.shape[0] < target_size:
            points_src = np.vstack(
                [points_src, points_src[np.random.choice(points_src.shape[0], target_size - points_src.shape[0])]])

        sample["points_ref"] = points_ref
        sample["points_src"] = points_src

        return sample


    def _downsample_patch_data(self, data, target_points):

        num_patches, num_points, num_coords = data.shape

        if num_points < target_points:
            raise ValueError(f"patch: ({num_points}) < target: ({target_points})")

        indices = np.random.choice(num_points, target_points, replace=False)

        downsampled_data = data[:, indices, :]

        return downsampled_data



    def pc_normalize(self,src, tgt, pose_tra):
 
        src_transformed = self.apply_transformation(src, pose_tra)

        center = np.mean(src_transformed, axis=0, keepdims=True).reshape(1, 3)
        tgt = np.concatenate([center, tgt], axis=0)
        concatenated = np.concatenate([src, tgt], axis=0)
        normalized = concatenated - np.mean(concatenated, axis=0, keepdims=True)

        scale = np.max(np.sqrt(np.sum(normalized ** 2, axis=1))) * np.random.uniform(low=0.5,high=1.5, size=1)
        normalized /= scale

        src_norm = normalized[:len(src), :]
        tgt_norm = normalized[len(src):, :]

        src_norm_transformed = self.apply_transformation(src_norm, pose_tra,True)
        translation = tgt_norm[0] - np.mean(src_norm_transformed, axis=0, keepdims=True)
        pose_tra[:, -1] = translation

        return src_norm, tgt_norm[1:], pose_tra

    def pc_normalize_1(self,src, tgt, pose_tra):
        b,_ = src.shape
        pc = np.concatenate([src,tgt],0)

        scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) #* np.random.uniform(low=0.2,high=1.8, size=1)

        pc /= scale

        pose_tra[:, -1] = pose_tra[:, -1] / scale

        return pc[:b,:], pc[b:,:], pose_tra


    def _random_crop(self, sample):
        #crop_proportion = self.args["random_crop"]["crop_range"]

        _,sample["points_ref"] = self._crop(sample["points_ref"], 0.9)
        _, sample["points_src"] = self._crop(sample["points_src"], 0.9)
        return sample

   
    def shuffle_point_cloud_and_overlap(self,p0, p1, overlap_matrix):

        shuffle_indices_p0 = np.random.permutation(p0.shape[0])
        shuffle_indices_p1 = np.random.permutation(p1.shape[0])

   
        p0_shuffled = p0[shuffle_indices_p0]
        p1_shuffled = p1[shuffle_indices_p1]

        overlap_matrix_shuffled = overlap_matrix[shuffle_indices_p0][:, shuffle_indices_p1]

        return p0_shuffled, p1_shuffled, overlap_matrix_shuffled
