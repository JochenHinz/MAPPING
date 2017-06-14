import meshio

points, cells, point_data, cell_data, field_data = \
    meshio.read('middle.vtk')

meshio.write(
    'middle_ascii.vtk',
    points,
    cells,
    point_data=point_data,
    cell_data=cell_data,
    field_data=field_data,
    file_format='vtk-ascii'
    )
