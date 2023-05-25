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
