import numpy as np
from spatialmath.pose3d import SE3
from spatialmath.base import trexp


def pnp(points_3d, points_2d, K):
    iterations = 10
    pose = SE3()
    cost, last_cost = 0, 0
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    for iter in range(iterations):
        H = np.zeros(shape=(6, 6))
        b = np.zeros(shape=6)

        cost = 0
        for i in range(len(points_3d)):
            pc = (pose * points_3d[i]).flatten()
            inv_z = 1.0 / pc[2]
            inv_z2 = inv_z * inv_z

            proj = np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy])
            e = points_2d[i] - proj

            cost += np.linalg.norm(e) ** 2
            J = np.array([
                [- fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2, - fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z],
                [0, - fy * inv_z, fy * pc[1] * inv_z, fy + fy * pc[1] * pc[1] * inv_z2, - fy * pc[0] * pc[1] * inv_z2, - fy * pc[0] * inv_z]
            ])

            H += J.T @ J
            b += - J.T @ e

        try:
            dx = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break

        if iter > 0 and cost >= last_cost:
            break

        pose = SE3.Delta(dx) * pose
        last_cost = cost

        # print(f"Iter {iter}, cost = {cost}")

        if np.linalg.norm(dx) < 1e-6:
            break

    # print(f"Pose:\n{pose}")

    return pose.A


def homogenize(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return np.hstack((a, 1))
    return np.hstack((a, np.ones((len(a), 1))))

def world_to_camera(points: np.ndarray, pose) -> np.ndarray:
    return homogenize(points) @ pose.T


def camera_to_pixel(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points_pixel = points @ intrinsics.T
    points_pixel /= points_pixel[:, 2:3]
    return points_pixel[:, :-1]

def world_to_pixel(points: np.ndarray, pose, intrinsics: np.ndarray) -> np.ndarray:
    return camera_to_pixel(world_to_camera(points, pose), intrinsics)


def pnp_ransac(points_3d, points_2d, K, threshold, iterations=100, sample_size=3):
    best_pose = None
    best_inliers = []
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    num_points = len(points_3d)

    for _ in range(iterations):
        # Randomly sample a subset of points
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        sampled_points_3d = points_3d[sample_indices]
        sampled_points_2d = points_2d[sample_indices]

        pose = pnp(sampled_points_3d, sampled_points_2d, K)

        inliers = []

        proj = world_to_pixel(points_3d, pose, K)
        error = np.linalg.norm(points_2d - proj)
        inliers.append(np.where(error < threshold)[0])


        # points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        # # Apply the transformation to the 3D points
        # transformed_3d = pose @ points_3d_homogeneous.T
        # # Convert the homogeneous 2D points to Cartesian coordinates
        # points_2d_proj = transformed_3d[:, :2] / transformed_3d[:, 2:]
        #
        # error = np.linalg.norm(points_2d - points_2d_proj.T)
        #
        # # Check if the point is an inlier based on the threshold
        # inliers.append(np.where(error < threshold)[0])

        # for i in range(num_points):
        #     # Project the 3D point onto the image plane using the current pose
        #     pc = (pose[:3, :3] @ points_3d[i]).flatten()
        #     proj = np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy])
        #
        #     # Calculate the reprojection error
        #     error = np.linalg.norm(points_2d[i] - proj)
        #
        #     # Check if the point is an inlier based on the threshold
        #     if error < threshold:
        #         inliers.append(i)

        # Update the best pose and inliers if the current model is better
        if len(inliers) > len(best_inliers):
            best_pose = pose
            best_inliers = inliers

    return best_pose
