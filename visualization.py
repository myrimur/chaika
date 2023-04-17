import pypangolin as pango
from OpenGL.GL import *
import numpy as np


class PointCloud:
    NAME = "Point Cloud"

    def __init__(self):
        pango.CreateWindowAndBind(self.NAME, 1024, 768)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pm = pango.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000)
        mv = pango.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
        self.s_cam = pango.OpenGlRenderState(pm, mv)


    def plot_points_and_trajectory(self, points, trajectory):
        # Arguments are lists of multiprocessing.managers.ListProxy type
        ui_width = 175
        handler = pango.Handler3D(self.s_cam)
        self.d_cam = (
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
            self.d_cam.Activate(self.s_cam)

            glPointSize(2)
            glColor3d(0, 1, 0)
            glBegin(GL_POINTS)
            for p in points:
                glVertex3d(*p)
            glEnd()

            glPointSize(4)
            glColor3d(1, 0, 0)
            glBegin(GL_LINES)
            p = np.array([0, 0, 0])
            for T in trajectory:
                glVertex3d(*p)
                p = T[:-1, -1]
                glVertex3d(*p)
            glEnd()

            pango.FinishFrame()
