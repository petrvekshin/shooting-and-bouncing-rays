import os, re
from time import time
import numpy as np
import matplotlib.colors as mcolors

# from numpy.ctypeslib import ndpointer

SPEED_OF_LIGHT = 299792458.0
named_colors = mcolors.get_named_colors_mapping()


def load_stl(geometry: dict):
    vert_dict = {}
    triangles = []
    vert_ind = -1
    ind = -1
    tri_mat = []
    if "PEC" not in geometry:
        tri_mat.append(0)
        materials = [*geometry]
    else:
        materials = [key for key in geometry if key != "PEC"]
        materials.insert(0, "PEC")
    for material in materials:
        for file, unit in geometry[material]:
            with open(file) as f:
                lines = f.readlines()
                for line in lines[:-1]:
                    vert_char = line.find("vertex")
                    if vert_char > -1:
                        ind += 1
                        xyz = tuple(
                            float(s) * unit
                            for s in line[vert_char + 6 :].split()
                        )
                        if xyz not in vert_dict:
                            vert_ind += 1
                            vert_dict[xyz] = [vert_ind, 1, [ind // 3]]
                            triangles.append(vert_ind)
                        else:
                            vert_dict[xyz][1] += 1
                            vert_dict[xyz][2].append(ind // 3)
                            triangles.append(vert_dict[xyz][0])
        tri_mat.append(len(triangles) / 3)
    tri_mat = np.array(tri_mat, dtype=np.uint32)
    verts = np.empty((vert_ind + 1, 3), dtype=np.float32)
    for xyz in vert_dict:
        vert_ind = vert_dict[xyz][0]
        verts[vert_ind, :3] = xyz
    del vert_dict
    triangles = np.array(triangles, dtype=np.uint32)
    triangles = np.vstack((triangles[::3], triangles[1::3], triangles[2::3])).T
    x1, x2 = np.min(verts[:, 0]), np.max(verts[:, 0])
    y1, y2 = np.min(verts[:, 1]), np.max(verts[:, 1])
    z1, z2 = np.min(verts[:, 2]), np.max(verts[:, 2])
    print(
        f"Geometry (total): {verts.shape[0]} vertices, {triangles.shape[0]} triangles "
    )
    print(f"Dimensions [m]: dx={x2-x1:.3f}, dy={y2-y1:.3f}, dz={z2-z1:.3f}")
    return (triangles, verts, tri_mat)


def load_test_mat():
    material = "fresnel"
    # path = f"...{material}.txt"
    path = f"...{material}.txt"
    n_th = 901
    num_freqs = 2
    
    mat = {}
    for key in ["te_re", "te_im", "tm_re", "tm_im"]:
        mat[key] = np.empty((n_th, num_freqs), dtype=np.float32, order="C")

    with open(path, "r") as f:
        for line in f:
            items = (line.rstrip()).split("\t")
            f = 0 if (items[0] == "1.00E+09") else 1
            th = int(round(float(items[1])*10.0))
            mat["tm_re"][th,f] = float(items[2])
            mat["tm_im"][th,f] = float(items[3])
            mat["te_re"][th,f] = float(items[4])
            mat["te_im"][th,f] = float(items[5])
    return mat


def load_material(file, n_th, freq):
    path = f"...{file}"
    with open(path, "r") as f:
        # Hz, MHz, GHz
        # deg
        # TE.mag.dB, TE.mag, TE.Re
        # TE.phase.deg, TE.phase, TE.Im
        header = f.readline()
        header = header.split("\t")
        c1 = header[2].lower()
        c2 = header[3].lower()
        d = np.loadtxt(path, comments=["#"])
        mat = {}
        nm = d.shape[0]
        f_p = np.ascontiguousarray(np.unique(d[:, 0]), dtype=np.float32)
        if header[0].upper() == "# GHZ":
            f_p = f_p * 1e9
        elif header[0].upper() == "# MHZ":
            f_p = f_p * 1e6
        th_p = np.unique(d[:, 1])
        TE = np.empty(nm, dtype=np.complex64)
        TM = np.empty(nm, dtype=np.complex64)

        if c1[-2:] == "db":
            TE = np.power(10.0, d[:, 2] / 20.0)
            TM = np.power(10.0, d[:, 4] / 20.0)
        else:
            TE = d[:, 2]
            TM = d[:, 4]
        if c2[-3:] == "deg":
            TE = TE * np.exp(1j * d[:, 3] * np.pi / 180.0)
            TM = TM * np.exp(1j * d[:, 5] * np.pi / 180.0)
        elif c2[-2:] == "im":
            TE = TE + 1j * d[:, 3]
            TM = TM + 1j * d[:, 5]
        else:
            TE = TE * np.exp(1j * d[:, 3])
            TM = TM * np.exp(1j * d[:, 5])
        TE_re = np.ascontiguousarray(TE.real)
        TE_im = np.ascontiguousarray(TE.imag)
        TM_re = np.ascontiguousarray(TM.real)
        TM_im = np.ascontiguousarray(TM.imag)
        del TE, TM
        mat = {}
        theta = np.linspace(0.0, 90.0, n_th, endpoint=True)
        mat["te_re"] = - np.ones((n_th, len(freq)), dtype=np.float32, order="C")
        mat["te_im"] = np.zeros((n_th, len(freq)), dtype=np.float32, order="C")
        mat["tm_re"] = np.ones((n_th, len(freq)), dtype=np.float32, order="C")
        mat["tm_im"] = np.zeros((n_th, len(freq)), dtype=np.float32, order="C")
        return mat


def sweep_points(sweep: str) -> list:
    points = []
    decimals = 0
    values = sweep.split(":")
    start = float(values[0])
    try:
        decimals = max(decimals, len(values[0].split(".")[1].strip()))
    except:
        pass
    points.append(start)
    if len(values) > 1:
        stop = float(values[1])
        if len(values) == 3:
            step = float(values[2])
            try:
                decimals = max(decimals, len(values[2].split(".")[1].strip()))
            except:
                pass
        else:
            step = 1.0
        n = int((stop - start) / step) + 1
        if n > 1:
            for i in range(1, n):
                points.append(start + i * step)
    return points, decimals


def rgb_array(color_names, materials):
    colors = np.empty((len(color_names), 3), dtype=np.uint32)
    for (i, mat) in enumerate(materials, start=0):
        color = color_names[mat].lower()
        if color == "random":
            colors[i, :] = np.random.choice(256, 3)
            continue
        if color[0] != "#":
            color = named_colors[color]
        colors[i, 0] = int(color[1:3], 16)
        colors[i, 1] = int(color[3:5], 16)
        colors[i, 2] = int(color[5:], 16)
    return colors


def read_setup(setup_file):
    units = {
        "[HZ]": 1.0,
        "[KHZ]": 1.0e3,
        "[MHZ]": 1.0e6,
        "[GHZ]": 1.0e9,
        "[M]": 1.0,
        "[CM]": 0.01,
        "[MM]": 0.001,
        "[DEG]": 180.0 / np.pi,
        "[RAD]": 1.0,
    }
    frequencies = []
    theta_phi_set = set()
    theta_phi = []
    setup_errors = []
    materials = []
    color_names = {}
    geometry = {}
    # ray_spacing = .25
    refl_limit = 5
    n_th = 901
    save_views = False

    with open(setup_file, "r") as f:
        read_mats = False
        read_freqs = False
        freq_dec = 0
        read_theta_phi = False
        angle_dec = 0
        unit = None
        for line in f:
            if line[:3] == "Mat":
                key = re.search('".*"', line)[0][1:-1]
                if key != "PEC":
                    materials.append(key)
                    if not os.path.exists("./Materials/" + key):
                        setup_errors.append(f"{key} not found")
                color_names[key] = re.search("\[.*\]", line)[0][1:-1].lower()
                geometry[key] = []
                read_mats = True
            elif read_mats:
                if (line != "\n") and (line[0] != "#"):
                    path_unit = line.split("[")
                    path = path_unit[0].rstrip()
                    if len(path_unit) > 1:
                        unit = "[" + path_unit[1].rstrip().upper()
                    else:
                        unit = "[M]"
                    if os.path.exists(path):
                        geometry[key].append((path, units[unit]))
                    else:
                        setup_errors.append(f"{path} does not exist")
                else:
                    read_mats = False
            elif line[:3] == "# F":
                read_freqs = True
                unit = re.search("\[.*\]", line[-6:])[0].upper()
            elif read_freqs:
                if line[0].isdigit():
                    sweeps = line.split(",")
                    for sweep in sweeps:
                        points, decimals = sweep_points(sweep)
                        freq_dec = max(freq_dec, decimals)
                        for f in points:
                            frequencies.append(f * units[unit])
                else:
                    read_freqs = False
            elif line[:3] == "# T":
                read_theta_phi = True
            elif read_theta_phi:
                if line[0].isdigit() or (line[0] == "-"):
                    sweeps = line.split(",")
                    theta_points, th_decimals = sweep_points(sweeps[0])
                    phi_points, ph_decimals = sweep_points(sweeps[1])
                    angle_dec = max(angle_dec, th_decimals, ph_decimals)
                    for theta in theta_points:
                        for phi in phi_points:
                            if (theta, phi) not in theta_phi_set:
                                theta_phi_set.add((theta, phi))
                                theta_phi.append((theta, phi))
                else:
                    read_theta_phi = False
            elif line[:3] == "RAY":
                ray_spacing = float(line.split("=")[-1].strip())
            elif line[:3] == "MAX":
                refl_limit = int(line.split("=")[-1].strip())
            elif line[:3] == "SAV":
                save_views = line.split("=")[-1].strip().upper()
                if (save_views[:4] == "TRUE") or ((save_views[:3] == "YES")):
                    save_views = True
                else:
                    save_views = False
    
    data = {}
    # data["save_views"] = save_views
    data["refl_limit"] = refl_limit
    # data["n_ray_groups"] = 1
    data["freq"] = np.sort(np.unique(np.array(frequencies, dtype=np.float32)))
    data["decimals"] = {"freq": freq_dec, "angle": angle_dec}
    data["theta_phi"] = np.array(theta_phi, dtype=np.float32)
    data["colors"] = rgb_array(color_names, materials)
    if setup_errors:
        print(setup_errors)
    triangles, verts, data["tri_mat"] = load_stl(geometry)

    # ui32 = {num_triangles, num_x, num_y, num_hits, num_freqs, ray_tracing}
    data["ui32"] = np.empty(6, dtype=np.uint32, order="C")
    data["ui32"][0] = triangles.shape[0]
    data["ui32"][4] = len(frequencies)
    data["ui32"][5] = 1
    # fp32 = {ray_spacing, x_min, y_min, t_max,
    # S/(2*pi), min(cos_th), max(S/(2*pi*cos_th)),
    # 7: 2*(n_th-1.0)/pi}
    data["fp32"] = np.empty(8, dtype=np.float32, order="C")
    data["fp32"][0] = ray_spacing
    data["fp32"][3] = np.inf
    data["fp32"][4] = (ray_spacing ** 2) / (2 * np.pi)
    data["fp32"][5] = np.cos(89.0 * np.pi / 180.0)
    data["fp32"][6] = data["fp32"][4] / data["fp32"][5]
    data["fp32"][7] = 2.0 * (n_th - 1.0) / np.pi
    data["wavenum"] = 2 * np.pi * data["freq"] / SPEED_OF_LIGHT
    data["vx0"] = np.ascontiguousarray(verts[:, 0], dtype=np.float32)
    data["vy0"] = np.ascontiguousarray(verts[:, 1], dtype=np.float32)
    data["vz0"] = np.ascontiguousarray(verts[:, 2], dtype=np.float32)
    del verts
    data["v0"] = np.ascontiguousarray((triangles[:, 0]).astype(np.uint32))
    data["v1"] = np.ascontiguousarray((triangles[:, 1]).astype(np.uint32))
    data["v2"] = np.ascontiguousarray((triangles[:, 2]).astype(np.uint32))
    del triangles
    data["mat"] = []
    for i in range(len(materials)):
        # data["mat"].append(load_material(materials[i], n_th, data["freq"]))
        data["mat"].append(load_test_mat())

    return data
