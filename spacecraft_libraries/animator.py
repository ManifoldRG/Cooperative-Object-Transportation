import numpy as np
from manim import *
class PolyhedronSubMobjects(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=25* DEGREES, theta=30 * DEGREES)
        self.camera.set_zoom(0.6)

        octahedron = Octahedron(edge_length=1)
        octahedron.graph[0].set_color(BLUE)
        octahedron.faces[2].set_color(DARK_BLUE)
        v = np.array(octahedron.vertex_coords)
        system = VGroup()
        system.add(octahedron)
        centre = octahedron.get_center()

        for i in v:
            direection = normalize(i - centre)
            cone = Cone(base_radius=0.2, height=0.2, color=BLUE, fill_opacity=0.3, direction=-direection)
            cone.move_to(i + 0.1 * direection)
            system.add(cone)

        self.add(system)

        initial_point = np.array([0, 0, 0])
        final_point = np.array([5, 5, 5])

        system.move_to(initial_point)

        self.play(system.animate.move_to(final_point), run_time=4)