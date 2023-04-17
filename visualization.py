import pypangolin as pango
from OpenGL.GL import *
import numpy as np


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


def show_point_cloud_and_trajectory(points, trajectory, points_lock, trajectory_lock):
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

        glColor3d(1, 0, 0)
        glBegin(GL_LINES)

        p = np.array([0, 0, 0])
        with trajectory_lock:
            for T in trajectory:
                glVertex3d(p[0], p[1], p[2])
                p = T[:-1, -1]
                glVertex3d(p[0], p[1], p[2])

        glEnd()
        pango.FinishFrame()

class PointCloud:
    def __init__(self, width=640, height=480, title='My Window'):
        self.width = width
        self.height = height
        self.title = title
        self.win = None
        self.points = None
        self.point_cloud = None

        # Create a PyPangolin window and bind it to a 3D rendering context
        self.win = pango.CreateWindowAndBind(self.title, self.width, self.height)
        self.view = pango.CreateDisplay()
        self.view.SetHandler(pango.Handler3D())
        self.view.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.view.SetLock(pango.Lock.LockLeft, True)

        # Set the PyPangolin window's background color to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Define the data source for the point cloud
        self.points = np.zeros((1, 3))

    def add_points(self, points):
        self.points = np.append(self.points, points, axis=0)

    def draw_points(self):
        # Clear the current OpenGL buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Activate the view and set its properties
        self.view.Activate(pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                            pango.ModelViewLookAt(0, -10, -8, 0, 0, 0, pango.AxisNegY))

        # Draw the points
        if self.points.shape[0] > 0:
            pango.DrawPoints(self.points)

        # Swap the OpenGL buffer to display the image
        pango.FinishFrame()

    def run(self):
        # Start the PyPangolin main loop
        while not pango.ShouldQuit():
            # Clear the PyPangolin window's buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            print(self.points)

            # d_cam.Activate(s_cam)

            # Draw your points
            self.draw_points()

            # Swap the buffers to display the new frame
            pango.FinishFrame()

        # Close the PyPangolin window
        self.win.DestroyWindow()
