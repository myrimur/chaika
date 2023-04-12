import pypangolin as pango
from OpenGL.GL import *


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
