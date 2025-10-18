#  VTK visualization base for all visualizers
#  Copyright (c) 4.2021. Yinyu Nie
#  License: MIT

import vtk
from vtk.util.numpy_support import numpy_to_vtk, get_numpy_array_type
import numpy as np
import seaborn as sns
import random
import math

class VIS_BASE(object):
    def __init__(self):
        self._cam_K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]])

    @property
    def cam_K(self):
        return self._cam_K

    def add_text(self, coords, text, color=(0, 0, 0), scale=0.1):
        """
        Add text at a coordinate.

        Args:
            coords: Coordinates to add text at.
            text: Text to place.
            color: Color for text as RGB. Defaults to black.
        """
        source = vtk.vtkVectorText()
        source.SetText(text)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        follower = vtk.vtkFollower()
        follower.SetMapper(mapper)
        follower.GetProperty().SetColor(color)
        follower.SetPosition(coords)
        follower.SetScale(scale)
        return follower

    def make_bands(self, dR, numberOfBands, nearestInteger):
        '''
        Divide a range into bands
        :param: dR - [min, max] the range that is to be covered by the bands.
        :param: numberOfBands - the number of bands, a positive integer.
        :param: nearestInteger - if True then [floor(min), ceil(max)] is used.
        :return: A List consisting of [min, midpoint, max] for each band.
        '''
        bands = list()
        if (dR[1] < dR[0]) or (numberOfBands <= 0):
            return bands
        x = list(dR)
        if nearestInteger:
            x[0] = math.floor(x[0])
            x[1] = math.ceil(x[1])
        dx = (x[1] - x[0]) / float(numberOfBands)
        b = [x[0], x[0] + dx / 2.0, x[0] + dx]
        i = 0
        while i < numberOfBands:
            bands.append(b)
            b = [b[0] + dx, b[1] + dx, b[2] + dx]
            i += 1
        return bands

    def setLUT(self, class_names, color_list):
        num_class = len(class_names)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(num_class)
        lut.SetTableRange(0, num_class)

        number_of_bands = lut.GetNumberOfTableValues()

        # we will use the midpoint of the band as label
        bands = self.make_bands([0, num_class], number_of_bands, False)
        labels = []

        for i in range(len(bands)):
            labels.append('{:4.2f}'.format(bands[i][1]))

        # annotate
        for i in range(num_class):
            lut.SetAnnotation(labels[i], str(class_names[i]))
            lut.SetTableValue(i, color_list[i][0], color_list[i][1], color_list[i][2], 1)
        lut.Build()

        return lut

    def set_scalar_bar_actor(self, class_names, color_list):

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetOrientationToVertical()
        scalar_bar.SetTitle('Category')
        scalar_bar.SetLookupTable(self.setLUT(class_names, color_list))
        scalar_bar.SetNumberOfLabels(0)
        # scalar_bar.GetLabelTextProperty().SetFontSize(80)
        scalar_bar.GetTitleTextProperty().SetFontSize(40)
        scalar_bar.GetAnnotationTextProperty().SetFontSize(20)
        scalar_bar.SetMaximumWidthInPixels(150)
        scalar_bar.SetMaximumHeightInPixels(800)
        scalar_bar.GetPositionCoordinate().SetValue(0.8, 0.15)

        return scalar_bar

    def set_axes_actor(self):
        '''
        Set camera coordinate system
        '''
        transform = vtk.vtkTransform()
        transform.Translate(0., 0., 0.)
        # self defined
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.SetTotalLength(1, 1, 1)

        axes.SetTipTypeToCone()
        axes.SetConeRadius(50e-2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(40e-3)

        vtk_textproperty = vtk.vtkTextProperty()
        vtk_textproperty.SetFontSize(1)
        vtk_textproperty.SetBold(True)
        vtk_textproperty.SetItalic(False)
        vtk_textproperty.SetShadow(True)

        for label in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            label.SetCaptionTextProperty(vtk_textproperty)

        return axes

    def set_camera(self, position, focal_point, up_vec, cam_K=None, fov_y=None):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*up_vec)
        if fov_y:
            camera.SetViewAngle(fov_y)
        elif cam_K:
            camera.SetViewAngle((2 * np.arctan(cam_K[1][2] / cam_K[1][1])) / np.pi * 180)
        else:
            raise ValueError('There no camera intrinsics.')
        return camera

    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_mapper(self, prop, mode):

        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

        else:
            raise IOError('No Mapper mode found.')

        return mapper

    def set_polygon_property(self, vertices, faces):
        n_points = len(vertices)
        vtk_points = vtk.vtkPoints()
        ntype = get_numpy_array_type(vtk_points.GetDataType())
        points_vtk = numpy_to_vtk(np.asarray(vertices, order='C', dtype=ntype), deep=1)
        vtk_points.SetNumberOfPoints(n_points)
        vtk_points.SetData(points_vtk)

        vtk_triangles = vtk.vtkCellArray()
        for item in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, item[0])
            triangle.GetPointIds().SetId(1, item[1])
            triangle.GetPointIds().SetId(2, item[2])
            vtk_triangles.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_triangles)

        polydata.Modified()

        return polydata

    def set_points_property(self, point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point

    def set_arrow_actor(self, startpoint, vector, tip_len_ratio=0.2, tip_r_ratio=0.08, shaft_r_ratio=0.02,
                        mode='vector'):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: a vtk arrow actor
        '''
        if mode == 'vector':
            vector = vector / np.linalg.norm(vector) * 0.5
            endpoint = startpoint + vector
        elif mode == 'endpoint':
            endpoint = vector
        else:
            raise NotImplementedError('Cannot recognize mode.')

        arrow_source = vtk.vtkArrowSource()

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # adjust arrow length
        arrow_source.SetTipLength(tip_len_ratio/length)
        arrow_source.SetTipRadius(tip_r_ratio/length)
        arrow_source.SetShaftRadius(shaft_r_ratio/length)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_bbox_line_prop(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_obj_actor(self, obj_path):
        vtk_object = vtk.vtkOBJReader()
        vtk_object.SetFileName(obj_path)
        vtk_object.Update()
        obj_actor = self.set_actor(self.set_mapper(vtk_object, 'model'))
        return obj_actor

    def get_bbox_line_actor(self, center, vectors, color, opacity=1., width=10):
        corners, faces = self.get_box_corners(center, vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_prop(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def get_bbox_cube_actor(self, center, vectors, color, opacity=1.):
        corners, faces = self.get_box_corners(center, vectors)
        box_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        box_actor.GetProperty().SetOpacity(opacity)
        return box_actor

    def get_plane_actor(self, origin, point1, point2, color, normal=(0, 1, 0), opacity=1.):
        plane_actor = self.set_actor(self.set_mapper(self.set_plane_prop(origin, point1, point2, normal), 'box'))
        plane_actor.GetProperty().SetOpacity(opacity)
        plane_actor.GetProperty().SetColor(color)
        return plane_actor

    def set_sphere_property(self, joint, radius):
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(*joint)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(10)
        sphere_source.SetThetaResolution(10)
        return sphere_source

    def set_line_property(self, p0, p1):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        return lineSource

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()
        return plydata

    def get_box_corners(self, center, vectors):
        '''
        Convert box center and vectors to the corner-form.
        Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
         [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
        :return: corner points and faces related to the box
        '''
        corner_pnts = [None] * 8
        corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
        corner_pnts[1] = tuple(center - vectors[0] - vectors[1] + vectors[2])
        corner_pnts[2] = tuple(center - vectors[0] + vectors[1] - vectors[2])
        corner_pnts[3] = tuple(center - vectors[0] + vectors[1] + vectors[2])

        corner_pnts[4] = tuple(center + vectors[0] - vectors[1] - vectors[2])
        corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
        corner_pnts[6] = tuple(center + vectors[0] + vectors[1] - vectors[2])
        corner_pnts[7] = tuple(center + vectors[0] + vectors[1] + vectors[2])

        faces = [(0, 1, 3, 2), (1, 5, 7, 3), (4, 6, 7, 5), (0, 2, 6, 4), (0, 4, 5, 1), (2, 3, 7, 6)]

        return corner_pnts, faces

    def mkVtkIdList(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def set_plane_prop(self, origin, point1, point2, normal=(0, 1, 0), x_resolution=10, y_resolution=10):
        source = vtk.vtkPlaneSource()
        source.SetOrigin(*origin)
        source.SetPoint1(*point1)
        source.SetPoint2(*point2)
        source.SetNormal(*normal)
        source.SetXResolution(x_resolution)
        source.SetYResolution(y_resolution)
        source.Update()
        return source.GetOutput()

    def set_cube_prop(self, corners, faces, color):

        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        color = np.uint8(color)

        for i in range(8):
            points.InsertPoint(i, corners[i])

        for i in range(6):
            polys.InsertNextCell(self.mkVtkIdList(faces[i]))

        for i in range(8):
            colors.InsertNextTuple3(*color)

        # Assign the pieces to the vtkPolyData
        cube.SetPoints(points)
        del points
        cube.SetPolys(polys)
        del polys
        cube.GetPointData().SetScalars(colors)
        cube.GetPointData().SetActiveScalars('Color')
        del colors

        return cube

    def get_voxel_actor(self, tsdf_volume, voxel_centroids, voxel_vectors):

        voxel_actors = []
        colors = get_colors(tsdf_volume)

        for tsdf_value, centroid, color in zip(tsdf_volume, voxel_centroids, colors):
            if tsdf_value == 1:
                continue
            corners, faces = self.get_box_corners(centroid, voxel_vectors)
            voxel_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            voxel_actor.GetProperty().SetOpacity(1)
            voxel_actors.append(voxel_actor)

        return voxel_actors

    def set_render(self, cam_loc=(0.1, 0.1, 0.1), focal_pnt=(0., 0., 0.), up_vec=None, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''set camera'''
        up_vec = [-cam_loc[0], -cam_loc[1], cam_loc[0] ** 2 / cam_loc[2] + cam_loc[1] ** 2 / cam_loc[2]] if up_vec is None else up_vec
        camera = self.set_camera(cam_loc, focal_pnt, up_vec, cam_K=self.cam_K)
        renderer.SetActiveCamera(camera)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, offline, *args, **kwargs):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(*args, **kwargs)
        # renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(*np.int32((self.cam_K[:2, 2] * 2 + 1)))
        render_window.SetOffScreenRendering(offline)

        return render_window

    def camRT2vtk_cam(self, camRT):
        cam_position = camRT[:3, -1]
        focal_point = np.array([0., 0., 1.]).dot(camRT[:3, :3].T) + camRT[:3, -1]
        up = np.array([0., -1., 0.]).dot(camRT[:3, :3].T)
        return cam_position, focal_point, up

    def visualize(self, save_path = None, offline=False, *args, **kwargs):
        '''
        Visualize a 3D scene.
        '''
        render_window = self.set_render_window(offline, *args, **kwargs)
        render_window.Render()

        if save_path is not None:
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(render_window)
            windowToImageFilter.Update()

            writer = vtk.vtkJPEGWriter()
            writer.SetFileName(save_path)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        if not offline:
            render_window_interactor = vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)
            render_window_interactor.Start()

def get_colors(values, palatte_name = 'RdBu', color_depth = 256):
    '''
    Return color values given scalars.
    :param values: N values
    :param palatte_name: check seaborn
    :param color_depth:
    :return: Nx3 colors
    '''
    palatte = np.array(sns.color_palette(palatte_name, color_depth))
    scaled_values = np.int32((values-min(values))/(max(values) - min(values)) * (color_depth-1))
    return palatte[scaled_values]

### For 2D visualization


if __name__ == '__main__':
    viser = VIS_BASE()
    viser.visualize(offline=False)