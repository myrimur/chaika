import pypangolin as pango
from OpenGL.GL import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_point_cloud(points):
    pango.CreateWindowAndBind("Point Cloud Viewer", 1024, 768)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pm = pango.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000)
    mv = pango.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    s_cam = pango.OpenGlRenderState(pm, mv)

    ui_width = 175

    handler = pango.Handler3D(s_cam)
    d_cam = (
        pango.CreateDisplay()
        .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach.Pix(ui_width),
            pango.Attach(1),
            -1024.0 / 768.0,
        )
        .SetHandler(handler)
    )

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        d_cam.Activate(s_cam)

        glPointSize(2)
        glBegin(GL_POINTS)
        for p in points:
            glColor3d(0, 1, 0)
            glVertex3d(p[0], p[1], p[2])

        glEnd()
        pango.FinishFrame()




def plot_trajectory(trajectory):
    pango.CreateWindowAndBind("Point Cloud Viewer", 1024, 768)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pm = pango.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000)
    mv = pango.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    s_cam = pango.OpenGlRenderState(pm, mv)

    ui_width = 175

    handler = pango.Handler3D(s_cam)
    d_cam = (
        pango.CreateDisplay()
        .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach.Pix(ui_width),
            pango.Attach(1),
            -1024.0 / 768.0,
        )
        .SetHandler(handler)
    )

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        d_cam.Activate(s_cam)

        glPointSize(2)
        glColor3d(0, 1, 0)

        glBegin(GL_LINES)

        p = np.array([0, 0, 0])
        for T in trajectory:
            glVertex3d(p[0], p[1], p[2])
            p = T[:-1, -1]
            glVertex3d(p[0], p[1], p[2])

        glEnd()
        pango.FinishFrame()


def show_point_cloud_and_trajectory(points, trajectory, trajectory_gt, trajecrory_our, trajectory_our_ransac, points_lock, trajectory_lock):
    pango.CreateWindowAndBind("Point Cloud Viewer", 1024, 768)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pm = pango.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000)
    mv = pango.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    s_cam = pango.OpenGlRenderState(pm, mv)

    ui_width = 175

    handler = pango.Handler3D(s_cam)
    d_cam = (
        pango.CreateDisplay()
        .SetBounds(
            pango.Attach(0),
            pango.Attach(1),
            pango.Attach.Pix(ui_width),
            pango.Attach(1),
            -1024.0 / 768.0,
        )
        .SetHandler(handler)
    )

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        d_cam.Activate(s_cam)

        glPointSize(2)
        glBegin(GL_POINTS)
        with points_lock:
            for p in points:
                glColor3d(0, 1, 0)
                glVertex3d(p[0], p[1], p[2])

        glEnd()

        p = np.array([0, 0, 0])
        p_gt = np.array([0, 0, 0])
        p_our = np.array([0, 0, 0])
        p_sack = np.array([0, 0, 0])
        with trajectory_lock:
            for T, T_gt, T_our, T_sack in zip(trajectory, trajectory_gt, trajecrory_our, trajectory_our_ransac):
                glColor3d(1, 0, 0)
                glBegin(GL_LINES)

                glVertex3d(p[0], p[1], p[2])
                p = T[:-1, -1]
                glVertex3d(p[0], p[1], p[2])
                glEnd()

                glColor3d(0.5, 0, 0.5)
                glBegin(GL_LINES)

                glVertex3d(p_gt[0], p_gt[1], p_gt[2])
                p_gt = T_gt[:-1, -1]
                glVertex3d(p_gt[0], p_gt[1], p_gt[2])
                glEnd()

                glColor3d(1, 1, 0)
                glBegin(GL_LINES)

                glVertex3d(p_our[0], p_our[1], p_our[2])
                p_our = T_our[:-1, -1]
                glVertex3d(p_our[0], p_our[1], p_our[2])
                glEnd()

                glColor3d(0.4, 1, 0.725)
                glBegin(GL_LINES)

                glVertex3d(p_sack[0], p_sack[1], p_sack[2])
                p_sack = T_sack[:-1, -1]
                glVertex3d(p_sack[0], p_sack[1], p_sack[2])
                glEnd()

        pango.FinishFrame()


def display_video(frames):
    frame = None
    while True:
        if not frames.empty():
            frame = frames.get()
        if frame is not None:
            cv.imshow("Original video", frame)
            if cv.waitKey(1) == ord('q'):
                break
