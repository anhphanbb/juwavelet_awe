from juwavelet import transform

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors


def get_normals(verts, faces):
    """
    Compute the normal vectors of provided triangel faces.

    Parameters
    ----------
    verts : 3 x N array of floats
        Vertices of the triangles that are to be 3d-plotted.
    faces : 3 x N array of ints
        each elements indexes the three vertices that make up one triangle.

    Returns
    -------
    3 x N array of floats
    """

    # Initialize an array to store the normals
    normals = np.zeros((faces.shape[0], 3))

    # Calculate normals for each triangle
    for i, face in enumerate(faces):
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]

        # Compute the edges of the triangle
        edge1, edge2 = (v1 - v0), (v2 - v0)

        # Compute the cross product of the two edges to get the normal
        normal = np.cross(edge1, edge2)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Store the normal
        normals[i] = normal

    # The cross product has a 180° ambiguity.
    # Enforce the normal vectors to point in only one half space
    return enforce_angles(normals)

# Function to convert a vector to spherical coordinates (azimuth and zenith)


def cartesian_to_spherical(normal):
    x, y, z = normal
    azimuth = np.arctan2(y, x)
    r = np.linalg.norm([x, y, z])
    zenith = np.arcsin(z / r)
    return azimuth, zenith

# Function to convert spherical coordinates back to Cartesian


def spherical_to_cartesian(azimuth, zenith):
    x = np.cos(zenith) * np.cos(azimuth)
    y = np.cos(zenith) * np.sin(azimuth)
    z = np.sin(zenith)
    return np.array([x, y, z])

# Function to enforce azimuth and zenith angle constraints


def enforce_angles(normals):
    consistent_normals = np.zeros_like(normals)

    for i, normal in enumerate(normals):
        azimuth, zenith = cartesian_to_spherical(normal)

        # Enforce azimuth to be between 0 and pi
        if azimuth < 0:
            azimuth += np.pi  # Adjust to be within [0, pi]

        # Enforce zenith to be between -pi/2 and pi/2
        # Already satisfied if calculated correctly

        # Convert back to Cartesian coordinates
        consistent_normal = spherical_to_cartesian(azimuth, zenith)
        consistent_normals[i] = consistent_normal

    return consistent_normals


def prepare_facecolor(faces_a, faces_b, color_a, color_b):

    rgba_a = mcolors.to_rgba(color_a)
    rgba_b = mcolors.to_rgba(color_b)

    color_array_a = np.tile(rgba_a, (faces_a.shape[0], 1))
    color_array_b = np.tile(rgba_b, (faces_b.shape[0], 1))
    color_array_both = np.vstack((color_array_a, color_array_b))

    return color_array_both


def plot_3D_wave(data_array, iso_values, increments=[1, 1, 1], facecolors=['tab:blue', 'tab:red'], ls=None):

    verts_one, faces_one, _, _ = measure.marching_cubes(data_array, level=iso_values[0])
    verts_two, faces_two, _, _ = measure.marching_cubes(data_array, level=iso_values[1])

    # In order to have overlapping isosurfaces, all the vertices and faces must be plotted in one execution of the plot_trisurf() function.
    verts_both = np.vstack((verts_one, verts_two))
    faces_both = np.vstack((faces_one, faces_two + verts_one.shape[0]))

    color_array = prepare_facecolor(faces_one, faces_two, facecolors[0], facecolors[1])

    if ls is not None:

        color_copy = color_array.copy()
        normals = get_normals(verts_both, faces_both)
        shading = ls.shade_normals(normals)

        # Multiply the first three columns by the vector
        result_rgb = color_copy[:, :3] * shading[:, np.newaxis]
        # Combine the modified first three columns with the unchanged alpha channel
        color_array = np.hstack((result_rgb, color_copy[:, 3:4]))

    return ax.plot_trisurf(verts_both[:, 0] * increments[0], verts_both[:, 1] * increments[1], verts_both[:, 2] * increments[2], triangles=faces_both, facecolor=color_array, shade=False)

# The eta-function serves to generate a two-dimensional hydrostatic wave field based on equation (20) in Smith (1980).


def eta(r_hat, theta, z_hat, h, nphi=1024):

    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)            # (nphi,)
    dphi = 2 * np.pi / nphi

    phis_ = phis[:, None, None]                                     # (nphi,1,1)
    r_hat_ = r_hat[None, :, :]                                       # (1,nx,ny)
    theta_ = theta[None, :, :]                                       # (1,nx,ny)

    arg = phis_ - theta_                                             # (nphi,nx,ny)
    num = np.exp(1j * z_hat / np.cos(phis_))                         # (nphi,)
    denom = (1 - 1j * r_hat_ * np.cos(arg))**2                       # (nphi,nx,ny)

    integrand = num / denom                                          # (nphi,nx,ny)
    integral = np.sum(integrand, axis=0) * dphi                      # (nx,ny)

    return h / (2 * np.pi) * integral                                    # (nx,ny)


# 1. Create the 3-D wave field
h, a = 1000.0, 25000.0  # mountain height (m), horizontal scale (m)
U, N = 10.0, 0.02       # wind speed (m/s), BV frequency (1/s)

nx, ny, nz = 300, 250, 40
dx = dy = 2000
dz = 250
x = np.linspace(-nx * dx / 2, nx * dx / 2, nx)
y = np.linspace(-ny * dy / 2, ny * dy / 2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
z = np.linspace(0, nz * dz, nz)

R = np.hypot(X, Y)
r_hat = R / a
Theta = np.arctan2(Y, X)

wavefield = np.full((len(x), len(y), len(z)), np.nan)
for zz in range(len(z)):
    z_hat = z[zz] * N / U
    wavefield[:, :, zz] = eta(r_hat, Theta, z_hat, h, nphi=2048)


# Compute the 3-D CWT
s0 = 2 * dx
s1 = nx * dz
dj = 1. / 4
jt = 6
jp = 7
js = int(1 / dj * np.log2(s1 / s0))

result = transform.decompose3d(wavefield, dx, dy, dz, s0, dj, js, jt, jp, aspect=10,
                               nxpad=None, nypad=None, nzpad=None, opts={'param': 4, 'threshold': 0},
                               mode="scaled", dtype=np.complex64)

# Determine isosurfaces for vertical displacement amplitude
vol = np.max(np.abs(result['decomposition']), axis=(0, 1, 2))
iso_val = 70
verts, faces, _, _ = measure.marching_cubes(vol, level=iso_val)
amp_normals = get_normals(verts, faces)

# Plot the wave field and the amplitude isosurface
ls = LightSource(azdeg=60, altdeg=45)

fig = plt.figure(figsize=(16, 8))
signal = wavefield

ax = fig.add_subplot(121, projection='3d')
plot_3D_wave(signal, [-70, 70], increments=[dx * 1e-3, dy * 1e-3, dz * 1e-3], ls=ls)
ax.set_xlabel('x / km')
ax.set_ylabel('y / km')
ax.set_zlabel('z / km')
ax.set_xlim([200, 600])
ax.set_ylim([0, 600])
ax.set_zlim([0, 10])
ax.view_init(elev=10, azim=-100)
ax.set_title(r'Isosurfaces of vertical displacement of -70m and 70m')


shading = ls.shade_normals(amp_normals)
rgba = mcolors.to_rgba('tab:orange')
color_array = np.tile(rgba, (faces.shape[0], 1))
result_rgb = color_array[:, :3] * shading[:, np.newaxis]
result = np.hstack((result_rgb, color_array[:, 3:4]))

ax = fig.add_subplot(122, projection='3d')
ax.plot_trisurf(verts[:, 0] * dx * 1e-3, verts[:, 1] * dy * 1e-3, verts[:, 2] *
                dz * 1e-3, triangles=faces, facecolor=result, shade=False)
ax.set_xlabel('x / km')
ax.set_ylabel('y / km')
ax.set_zlabel('z / km')
ax.set_xlim([200, 600])
ax.set_ylim([0, 600])
ax.set_zlim([0, 10])
ax.view_init(elev=10, azim=-100)
ax.set_title(r'Isosurface of a vertical displacement amplitude of 70m')

plt.savefig('3d_decomposition_example.png', dpi=120, facecolor="w", edgecolor="w", bbox_inches="tight")
plt.savefig('3d_decomposition_example.pdf', dpi=120, facecolor="w", edgecolor="w", bbox_inches="tight")
