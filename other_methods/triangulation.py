import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from scipy.spatial import (  # pylint: disable=no-name-in-module
    Delaunay,
)
from shapely.geometry import MultiLineString, MultiPoint
from shapely.ops import polygonize, unary_union


def plot_polygon(polygon):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = 0.3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc="#999999", ec="#000000", fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def create_point_coordinates():
    size = 50

    variance_x = 5
    variance_y = 4

    mean = (0, 0)
    cov = [[variance_x, 0], [0, variance_y]]

    return np.random.multivariate_normal(mean, cov, (size, size)).reshape(-1, 2)


def handle_areas_single_triangle(vertices, faces_neighbor):
    """
    Computes the areas [a1, a2, a3] given the vertices [v1, v2, v3] such that
    a1 is between v1 and v2, a2 is between v2 and v3, and a3 is between v3 and v1.
    OBS: If a face contains no neighbor triangle, its area is set to zero.
    """
    deltas = np.roll(vertices, -1, axis=0) - vertices
    areas = np.sqrt(np.sum(deltas**2, axis=1))
    areas[np.where(faces_neighbor == -1)] = 0.0
    return areas


def compute_areas(vertices, faces_neighbor):
    areas = np.empty((len(vertices), 3))

    for i, (v, n) in enumerate(zip(vertices, faces_neighbor)):
        areas[i] = handle_areas_single_triangle(v, n)

    return areas


def handle_volume_of_triangle(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(
        x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    )


def compute_volumes(vertices):
    volumes = np.empty(len(vertices))

    for i, v in enumerate(vertices):
        volumes[i] = handle_volume_of_triangle(v)

    return volumes


def get_centroids(vertices):
    return np.average(vertices, axis=1)


def get_deltas(centroids, faces_neigbor):
    # Calculate distances even for -1 neighbors. This value will be multiplied
    # by the null area eventually, so it is not a problem...
    deltas = np.empty((len(centroids), 3, 2))
    for i, n in enumerate(faces_neigbor):
        deltas[i] = centroids[n] - centroids[i]

    return deltas


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    vertices = points[tri.simplices]
    a = (
        (vertices[:, 0, 0] - vertices[:, 1, 0]) ** 2
        + (vertices[:, 0, 1] - vertices[:, 1, 1]) ** 2
    ) ** 0.5
    b = (
        (vertices[:, 1, 0] - vertices[:, 2, 0]) ** 2
        + (vertices[:, 1, 1] - vertices[:, 2, 1]) ** 2
    ) ** 0.5
    c = (
        (vertices[:, 2, 0] - vertices[:, 0, 0]) ** 2
        + (vertices[:, 2, 1] - vertices[:, 0, 1]) ** 2
    ) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = vertices[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = MultiLineString(edge_points)
    vertices = list(polygonize(m))
    return unary_union(vertices), edge_points


def main():
    points = np.array(
        [[0, 0], [0, 1.1], [1, 0], [1, 1], [0.3, 0.4], [0.2, 0.35], [0.31, 0.7]]
    )
    # points = create_point_coordinates()

    tri = Delaunay(points)

    # list of triangles:
    # Each triangle is represented as three integers whose value represents
    # the index in to the original points array
    # print(tri.simplices)

    # List of coordinates of the vertices of each triangle:
    # print(points[tri.simplices])

    centroids = get_centroids(points[tri.simplices])

    # faces_neigbor = np.roll(tri.neighbors, 1, axis=1)

    # areas = compute_areas(points[tri.simplices], faces_neigbor)

    # deltas = get_deltas(centroids, faces_neigbor)

    # concave_hull, edge_points = alpha_shape(points, alpha=2)

    # print concave_hull
    # lines = LineCollection(edge_points)
    # plt.figure()
    # plt.gca().add_collection(lines)
    # plt.plot(points[:, 0], points[:, 1], "o")
    # plt.show()

    # Which triangles are adjacent to each other:
    # Example: [0 -1 1] indicate that the first vertex opposes triangle 0, the
    # second vertex opposes an edge (no triangle) and the third vertex opposes
    # triangle 1.
    # print(tri.neighbors)

    # plot_polygon(concave_hull)

    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    for i in range(len(points)):
        plt.text(points[i, 0], points[i, 1], str(i))
    plt.plot(centroids[:, 0], centroids[:, 1], "rx")
    for i in range(len(centroids)):
        plt.text(centroids[i, 0], centroids[i, 1], str(i))
    # plt.plot(*concave_hull.exterior.xy)
    # plt.plot(*concave_hull.wkt.xy)
    plt.show()

    print("foi")

    # vor = Voronoi(points)

    # Coordinates of input points
    # vor.points

    # Coordinates of Voronoi vertices
    # vor.vertices

    # Indices of the Voronoi vertices forming each Voronoi region
    # vor.regions

    # Index of the Voronoi region for each input point
    # vor.point_region

    # Neighboring regions of a given region
    # Remember: the Voronoi vertices are the centers of the circumcircles
    # associated with each Delauney triangulation.
    # indptr, neighbours = Delaunay(points).vertex_neighbor_vertices
    # for i, p in enumerate(points):
    #     neighbor_indices = neighbours[indptr[i] : indptr[i + 1]]
    #     print(f"Point {i} (p) neighbors are: {neighbor_indices}")

    # voronoi_plot_2d(
    #     vor,
    #     show_vertices=True,
    #     line_colors="orange",
    #     line_width=2,
    #     line_alpha=0.6,
    #     point_size=2,
    # )
    # plt.show()

    # vertices = vor.vertices
    # regions = [x for x in vor.regions if x != []]

    # regions, vertices = voronoi_finite_polygons_2d(vor)

    # pts = MultiPoint([Point(i) for i in points])
    # mask = pts.convex_hull
    # new_vertices = []
    # for region in regions:
    #     polygon = vertices[region]
    #     shape = list(polygon.shape)
    #     shape[0] += 1
    #     p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    #     poly = np.array(
    #         list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1]))
    #     )
    #     new_vertices.append(poly)
    #     plt.fill(*zip(*poly), alpha=0.4)
    # plt.plot(points[:, 0], points[:, 1], "ko")
    # for vpair in vor.ridge_vertices:
    #     if vpair[0] >= 0 and vpair[1] >= 0:
    #         v0 = vor.vertices[vpair[0]]
    #         v1 = vor.vertices[vpair[1]]
    #         # Draw a line from v0 to v1.
    #         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], "k", linewidth=2)
    # plt.title("Clipped Voronoi")
    # plt.show()


if __name__ == "__main__":
    main()
