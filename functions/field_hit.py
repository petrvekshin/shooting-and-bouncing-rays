import os, subprocess, ctypes, threading, sys
import numpy as np
from PIL import Image


c_file = "./functions/rt6.c"
o_file = c_file[:-2] + ".o"
so_file = c_file[:-2] + ".so"
dll_file = c_file[:-2] + ".dll"
is_win = sys.platform == "win32"

if __name__ == "__main__":
    from setup import read_setup
    if is_win:
        subprocess.run(["gcc", "-c", "-o", o_file, c_file])
        subprocess.run(["gcc", "-o", dll_file, "-s", "-shared", o_file])
        os.remove(o_file)
        print(dll_file, "compiled")
    else:
        subprocess.run(["cc", "-fPIC", "-shared", "-o", so_file, c_file])
        print(so_file, "compiled")
else:
    from functions.setup import read_setup

if is_win:
    sbr_lib = ctypes.cdll.LoadLibrary(os.path.abspath(dll_file))
else:
    sbr_lib = ctypes.cdll.LoadLibrary(os.path.abspath(so_file))
vu = ctypes.c_ulong
pu = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
pf = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
ppf = np.ctypeslib.ndpointer(pf, flags="C_CONTIGUOUS")

# Import c functions
_tri_nd = sbr_lib.tri_nd
_tri_nd.restype = None
_tri_nd.argtypes = [
    pu,
    pu, pu, pu,
    pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf
]

_init_hit = sbr_lib.init_hit
_init_hit.restype = None
_init_hit.argtypes = [
    pu, pf, pu,
    pu, pu, pu,
    pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf
]

_init_field = sbr_lib.init_field
_init_field.restype = None
_init_field.argtypes = [
    pu,
    pu, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pu,
    ppf, ppf,
    ppf, ppf,
    pu,
    pf, pf, pf,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf
]

_init_pol_hit = sbr_lib.init_pol_hit
_init_pol_hit.restype = None
_init_pol_hit.argtypes = [
    pu,
    pu, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pu,
    ppf, ppf,
    ppf, ppf,
    pu,
    pf, pf, pf,
    pf, pf, pf,
    pf, pf,
    pu, pu,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf
]

_next_field_hit = sbr_lib.next_field_hit
_next_field_hit.restype = None
_next_field_hit.argtypes = [
    pu,
    pu, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pf, pf, pf,
    pf, pu,
    ppf, ppf,
    ppf, ppf,
    pu,
    pf, pf, pf,
    pf, pf, pf,
    pf, pf, pf,
    pu, pu,
    pu, pu,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf,
    pf, pf
]

_sort_f = sbr_lib.sort_f
_sort_f.restype = None
_sort_f.argtypes = [vu, pu, pu, pf]

_sort_u = sbr_lib.sort_u
_sort_u.restype = None
_sort_u.argtypes = [vu, pu, pu, pu]

_sort_f_2d = sbr_lib.sort_f_2d
_sort_f_2d.restype = None
_sort_f_2d.argtypes = [vu, vu, pu, pu, pf]


class SBR:

    VIEW_PATH = "./Data/Views"
    SPEED_OF_LIGHT = 299792458.0
    TRI_PARAMS = ["n0x", "n0y", "n0z", "d0",
                  "n1x", "n1y", "n1z", "d1",
                  "n2x", "n2y", "n2z", "d2", "l"]
    RAY_FIELDS = ["te_x_re", "te_x_im", "tm_x_re", "tm_x_im",
                  "te_y_re", "te_y_im", "tm_y_re", "tm_y_im"]
    FAR_FIELDS = ["ex_x_re", "ex_x_im", "ey_x_re", "ey_x_im",
                  "ex_y_re", "ex_y_im", "ey_y_re", "ey_y_im"]


    def __init__(self, setup_file: str) -> None:
        self.setup_file = setup_file
        args = read_setup(setup_file)
        for key in args:
            setattr(self, key, args[key])
        mat_pointers = ["p_te_re", "p_te_im", "p_tm_re", "p_tm_im"]
        for key in mat_pointers:
            setattr(self, key, np.empty(
                len(self.mat),
                dtype=np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
                )
            )
        for i in range(len(self.mat)):
            for key in mat_pointers:
                getattr(self, key)[i] = self.mat[i][key[2:]].ctypes.data
        self.ray_spacing = self.fp32[0]
        self.num_triangles = self.ui32[0]
        self.num_freqs = self.ui32[4]
        self.point = None
        self.threads = []
    

    def thread_start_join(self):
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()
        self.threads.clear()
    

    def rotate_vertices(self, theta: float, phi: float) -> None:
        """
        Rotate vertices according to theta & phi (spherical coordinates).
        x-polarization is theta, y-polarization is phi
        """
        _vx = self.vx0 * np.cos(phi * np.pi / 180) + self.vy0 * np.sin(phi * np.pi / 180)
        _vy = -self.vx0 * np.sin(phi * np.pi / 180) + self.vy0 * np.cos(phi * np.pi / 180)
        self.vx, self.vy = _vx, _vy
        _vx = _vy = None
        _vz = self.vz0 * np.cos(theta * np.pi / 180) + self.vx * np.sin(theta * np.pi / 180)
        _vx = -self.vz0 * np.sin(theta * np.pi / 180) + self.vx * np.cos(theta * np.pi / 180)
        self.vx, self.vz = _vx, _vz
        _vx = _vz = None
    

    def precompute_triangles(self) -> None:
        """
        Perform triangle precomputing for Havel & Herout algorithm
        """
        print("precomputing triangles")
        for key in self.TRI_PARAMS:
            setattr(self, key, np.empty(self.ui32[0], dtype=np.float32, order="C"))
        self.ind, n_thread = self.split_ind(len(self.v0))
        print(self.ind.flags, self.ind[:])
        print(self.v0, self.v1, self.v2, self.vx, self.vy, self.vz,)
        for i in range(n_thread):
            print("self.ind[i:]", self.ind[i:])
            thread = threading.Thread(
                target=_tri_nd,
                args=(
                    self.ind[i:],
                    self.v0,
                    self.v1,
                    self.v2,
                    self.vx,
                    self.vy,
                    self.vz,
                    self.n0x,
                    self.n0y,
                    self.n0z,
                    self.d0,
                    self.n1x,
                    self.n1y,
                    self.n1z,
                    self.d1,
                    self.n2x,
                    self.n2y,
                    self.n2z,
                    self.d2,
                    self.l,
                ),
            )
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()
        self.ind = None


    def sort_arrays(self, *keys):
        for key in keys:
            args = [len(self.hit_count), self.ind, self.hit_count, getattr(self, key)]
            if getattr(self, key).dtype == np.uint32:
                func = _sort_u
            elif len(getattr(self, key).shape) == 1:
                func = _sort_f
            else:
                func = _sort_f_2d
                args.insert(1, self.ui32[4])
            thread = threading.Thread(target=func, args=args)
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()
        self.hit_count = None


    def save_view(self):
        name = f'{self.VIEW_PATH}/ph={self.phi:.{self.angle_dec}f}_th={self.theta:.{self.angle_dec}f}_x1.png'
        # if self.tri_mat[0] == 0:
        #     self.tri_mat = self.tri_mat[1:]
        x_min = np.min(self.ox)
        y_min = np.min(self.oy)
        x_ind = np.round((self.ox - x_min) / self.fp32[0])
        y_ind = np.round((self.oy - y_min) / self.fp32[0])
        x_ind = x_ind.astype(np.uint32)
        y_ind = y_ind.astype(np.uint32)
        n_x = int(np.max(x_ind) + 1)
        n_y = int(np.max(y_ind) + 1)
        img = np.empty((n_y, n_x, 3))
        img = img.astype(np.uint8)
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = 255, 255, 255
        I = self.tri < self.tri_mat[1]
        Nz = (np.abs(self.n0z[self.tri] * self.l[self.tri])) ** 0.6
        img[y_ind[I], x_ind[I], 0] = (Nz[I] * self.colors[0, 0]).astype(np.uint8)
        img[y_ind[I], x_ind[I], 1] = (Nz[I] * self.colors[0, 1]).astype(np.uint8)
        img[y_ind[I], x_ind[I], 2] = (Nz[I] * self.colors[0, 2]).astype(np.uint8)
        for i in range(1, self.colors.shape[0]):
            I = (self.tri >= self.tri_mat[i - 1]) * (self.tri < self.tri_mat[i])
            img[y_ind[I], x_ind[I], 0] = (Nz[I] * self.colors[i, 0]).astype(np.uint8)
            img[y_ind[I], x_ind[I], 1] = (Nz[I] * self.colors[i, 1]).astype(np.uint8)
            img[y_ind[I], x_ind[I], 2] = (Nz[I] * self.colors[i, 2]).astype(np.uint8)
        img = Image.fromarray(np.rot90(np.rot90(np.flipud(img))))
        img.save(name)


    @staticmethod
    def fields_to_lines(ex_x, ey_x, ex_y, ey_y):
        # magnitude
        x_x_m = np.abs(ex_x)
        y_x_m = np.abs(ey_x)
        x_y_m = np.abs(ex_y)
        y_y_m = np.abs(ey_y)
        # phase in degrees
        x_x_a = np.angle(ex_x, deg=True)
        y_x_a = np.angle(ey_x, deg=True)
        x_y_a = np.angle(ex_y, deg=True)
        y_y_a = np.angle(ey_y, deg=True)
        # make phase positive
        x_x_a[x_x_a < 0.0] = x_x_a[x_x_a < 0.0] + 360.0
        y_x_a[y_x_a < 0.0] = y_x_a[y_x_a < 0.0] + 360.0
        x_y_a[x_y_a < 0.0] = x_y_a[x_y_a < 0.0] + 360.0
        y_y_a[y_y_a < 0.0] = y_y_a[y_y_a < 0.0] + 360.0
        num_freqs = len(x_x_m)
        lines = np.empty((num_freqs, 8), dtype=np.float32, order="C")
        for i in range(num_freqs):
            lines[i,0] = max(x_x_m[i], 0.000001)
            lines[i,1] = x_x_a[i]
            lines[i,2] = max(y_x_m[i], 0.000001)
            lines[i,3] = y_x_a[i]
            lines[i,4] = max(x_y_m[i], 0.000001)
            lines[i,5] = x_y_a[i]
            lines[i,6] = max(y_y_m[i], 0.000001)
            lines[i,7] = y_y_a[i]
        return lines


    def sum_far_fields(self):
        x_x = np.sum(self.ex_x_re, 0) + 1j*np.sum(self.ex_x_im, 0)
        y_x = np.sum(self.ey_x_re, 0) + 1j*np.sum(self.ey_x_im, 0)
        x_y = np.sum(self.ex_y_re, 0) + 1j*np.sum(self.ex_y_im, 0)
        y_y = np.sum(self.ey_y_re, 0) + 1j*np.sum(self.ey_y_im, 0)
        # magnitude
        x_x_m = np.abs(x_x)
        y_x_m = np.abs(y_x)
        x_y_m = np.abs(x_y)
        y_y_m = np.abs(y_y)
        # phase in degrees
        x_x_a = np.angle(x_x, deg=True)
        y_x_a = np.angle(y_x, deg=True)
        x_y_a = np.angle(x_y, deg=True)
        y_y_a = np.angle(y_y, deg=True)
        # make phase positive
        x_x_a[x_x_a < 0.0] = x_x_a[x_x_a < 0.0] + 360.0
        y_x_a[y_x_a < 0.0] = y_x_a[y_x_a < 0.0] + 360.0
        x_y_a[x_y_a < 0.0] = x_y_a[x_y_a < 0.0] + 360.0
        y_y_a[y_y_a < 0.0] = y_y_a[y_y_a < 0.0] + 360.0
        # lines
        lines = np.empty((self.num_freqs, 8), dtype=np.float32, order="C")
        for i in range(self.num_freqs):
            lines[i,0] = x_x_m[i]
            lines[i,1] = x_x_a[i]
            lines[i,2] = y_x_m[i]
            lines[i,3] = y_x_a[i]
            lines[i,4] = x_y_m[i]
            lines[i,5] = x_y_a[i]
            lines[i,6] = y_y_m[i]
            lines[i,7] = y_y_a[i]
        return lines


    def split_ind(self, num_elements: int) -> np.ndarray:
        ind = [int(v1 * num_elements / self.num_threads) for v1 in range(self.num_threads)]
        ind.append(num_elements)
        ind = list(set(ind))
        ind.sort()
        return np.array(ind).astype(np.uint32), len(ind) - 1


    def init_hit_field(self):
        bi_theta = self.theta_phi[self.point, 0]
        bi_phi = self.theta_phi[self.point, 1]

        for key in self.RAY_FIELDS:
            setattr(self, key, None)
        for key in self.FAR_FIELDS:
            setattr(self, key, None)
        self.rotate_vertices(0.0, bi_phi)
        self.precompute_triangles()
        for item in ["n0x", "n0y", "n0z", "l"]:
            print(item, getattr(self, item))
        print("normals", self.n0x*self.l, self.n0y*self.l, self.n0z*self.l)
        # fp32 = {ray_spacing, x_min, y_min, t_max, ...}
        self.fp32[1] = x_min = np.min(self.vx) - self.fp32[0] / 2.0
        self.fp32[2] = y_min = np.min(self.vy) - self.fp32[0] / 2.0
        x_max = np.max(self.vx) + self.fp32[0] / 2.0
        y_max = np.max(self.vy) + self.fp32[0] / 2.0
        # ui32 = {num_triangles, num_x, num_y, num_hits, num_freqs}
        self.ui32[1] = n_x = int(np.ceil((x_max - x_min) / self.fp32[0]))
        self.ui32[2] = n_y = int(np.ceil((y_max - y_min) / self.fp32[0]))
        self.tri = np.empty(n_x * n_y, dtype=np.uint32, order="C")
        for key in ["ox", "oy", "oz"]:
            setattr(self, key, np.empty(n_x * n_y, dtype=np.float32, order="C"))
        self.oz.fill(-np.inf)
        _init_hit(
            self.ui32,
            self.fp32,
            self.tri,
            self.v0,
            self.v1,
            self.v2,
            self.vx,
            self.vy,
            self.vz,
            self.n0x,
            self.n0y,
            self.n0z,
            self.d0,
            self.ox,
            self.oy,
            self.oz,
        )
        # self.vx = self.vy = self.vz = None
        num_hits = self.num_init_hits = self.ui32[3]
        for key in ["tri", "ox", "oy", "oz"]:
            setattr(self, key, getattr(self, key)[:num_hits])
        # test part
        for key in self.RAY_FIELDS:
            setattr(self, key, np.empty((num_hits, self.num_freqs), dtype=np.float32, order="C"))
        for w in range(self.num_freqs):
            v = self.wavenum[w] * self.oz
            re_exp = np.cos(v)
            im_exp = np.sin(v)
            self.te_x_re[:,w] = re_exp #* np.cos(alpha)
            self.te_x_im[:,w] = im_exp #* np.cos(alpha)
            self.tm_x_re[:,w] = re_exp * 0.0 # np.sin(alpha)
            self.tm_x_im[:,w] = im_exp * 0.0 # np.sin(alpha)
            self.te_y_re[:,w] = re_exp * 0.0 # np.sin(alpha)
            self.te_y_im[:,w] = im_exp * 0.0 # np.sin(alpha)
            self.tm_y_re[:,w] = - re_exp #* np.cos(alpha)
            self.tm_y_im[:,w] = - im_exp #* np.cos(alpha)

        # rotate for bistatic scattering
        _oz = self.oz * np.cos(bi_theta * np.pi / 180) + self.ox * np.sin(bi_theta * np.pi / 180)
        _ox = -self.oz * np.sin(bi_theta * np.pi / 180) + self.ox * np.cos(bi_theta * np.pi / 180)
        self.ox, self.oz = _ox, _oz
        _ox = _oz = None
        self.rotate_vertices(bi_theta, bi_phi)
        self.precompute_triangles()
        # save_view(c)
        # end of rotation
        for key in self.FAR_FIELDS:
            setattr(self, key, np.empty((num_hits, self.num_freqs), dtype=np.float32, order="C"))

        self.dx = np.ones(num_hits, dtype=np.float32, order="C") * np.sin(bi_theta * np.pi / 180)
        self.dy = np.zeros(num_hits, dtype=np.float32, order="C")
        self.dz = - np.ones(num_hits, dtype=np.float32, order="C") * np.cos(bi_theta * np.pi / 180)
        qc = np.ones(num_hits, dtype=np.float32, order="C") * np.cos(30.0 * np.pi / 180)
        qs = np.ones(num_hits, dtype=np.float32, order="C") * np.sin(30.0 * np.pi / 180)
        self.qx = qc * np.cos(bi_theta * np.pi / 180)
        self.qy = qs
        self.qz = qs * np.sin(bi_theta * np.pi / 180)
        
        self.ind, n_thread = self.split_ind(num_hits)
        self.vis = np.empty(num_hits, dtype=np.uint32, order="C")
        self.vis_count = np.empty(n_thread, dtype=np.uint32, order="C")
        self.hit = np.empty(num_hits, dtype=np.uint32, order="C")
        self.hit_count = np.empty(n_thread, dtype=np.uint32, order="C")

        for i in range(n_thread):
            thread = threading.Thread(
                target=_next_field_hit,
                args=(
                    self.ind[i:],
                    self.ui32, self.fp32, self.wavenum,
                    self.n0x, self.n0y, self.n0z, self.d0,
                    self.n1x, self.n1y, self.n1z, self.d1,
                    self.n2x, self.n2y, self.n2z, self.d2,
                    self.l, self.tri_mat,
                    self.p_te_re, self.p_te_im,
                    self.p_tm_re, self.p_tm_im,
                    self.tri,
                    self.ox, self.oy, self.oz,
                    self.dx, self.dy, self.dz,
                    self.qx, self.qy, self.qz,
                    self.vis, self.vis_count[i:],
                    self.hit, self.hit_count[i:],
                    self.te_x_re, self.te_x_im,
                    self.tm_x_re, self.tm_x_im,
                    self.te_y_re, self.te_y_im,
                    self.tm_y_re, self.tm_y_im,
                    self.ex_x_re, self.ex_x_im,
                    self.ey_x_re, self.ey_x_im,
                    self.ex_y_re, self.ex_y_im,
                    self.ey_y_re, self.ey_y_im,
                ),
            )
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()

        num_hits = np.sum(self.hit_count)
        return num_hits
        if num_hits == 0:
            return 0

        _sort_u(n_thread, self.ind, self.hit_count, self.hit)
        self.hit = self.hit[:num_hits]
        ray_vecs = ["dx", "dy", "dz", "qx", "qy", "qz"]
        arrays_to_sort = ["tri", "ox", "oy", "oz", *ray_vecs, *self.ray_fields]
        for key in arrays_to_sort:
            setattr(self, key, getattr(self, key)[self.hit])
        return num_hits
        
        # compute far fields
        """
        for key in self.FAR_FIELDS:
            setattr(self, key, np.empty((num_hits, self.num_freqs), dtype=np.float32, order="C"))
        self.ind, n_thread = self.split_ind(num_hits)
        for i in range(n_thread):
            thread = threading.Thread(
                target=_init_field,
                args=(
                    self.ind[i:],
                    self.ui32, self.fp32, self.wavenum,
                    self.n0x, self.n0y, self.n0z, self.d0,
                    self.n1x, self.n1y, self.n1z, self.d1,
                    self.n2x, self.n2y, self.n2z, self.d2,
                    self.l, self.tri_mat,
                    self.p_te_re, self.p_te_im,
                    self.p_tm_re, self.p_tm_im,
                    self.tri,
                    self.ox, self.oy, self.oz,
                    self.ex_x_re, self.ex_x_im,
                    self.ey_x_re, self.ey_x_im,
                    self.ex_y_re, self.ex_y_im,
                    self.ey_y_re, self.ey_y_im,
                )
            )
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()
        """


    def init_pol_hit(self):
        for key in self.FAR_FIELDS:
            setattr(self, key, None)

        num_hits = len(self.tri)

        ray_vecs = ["dx", "dy", "dz", "qx", "qy"]
        for key in ray_vecs:
            setattr(self, key, np.empty(num_hits, dtype=np.float32, order="C"))
        for key in self.RAY_FIELDS:
            setattr(self, key, np.empty((num_hits, self.num_freqs), dtype=np.float32, order="C"))
        
        self.ind, n_thread = self.split_ind(num_hits)
        self.hit = np.empty(num_hits, dtype=np.uint32, order="C")
        self.hit_count = np.empty(n_thread, dtype=np.uint32, order="C")
        for i in range(n_thread):
            thread = threading.Thread(
                target=_init_pol_hit,
                args=(
                    self.ind[i:],
                    self.ui32, self.fp32, self.wavenum,
                    self.n0x, self.n0y, self.n0z, self.d0,
                    self.n1x, self.n1y, self.n1z, self.d1,
                    self.n2x, self.n2y, self.n2z, self.d2,
                    self.l, self.tri_mat,
                    self.p_te_re, self.p_te_im,
                    self.p_tm_re, self.p_tm_im,
                    self.tri,
                    self.ox, self.oy, self.oz,
                    self.dx, self.dy, self.dz,
                    self.qx, self.qy,
                    self.hit, self.hit_count[i:],
                    self.te_x_re, self.te_x_im,
                    self.tm_x_re, self.tm_x_im,
                    self.te_y_re, self.te_y_im,
                    self.tm_y_re, self.tm_y_im,
                ),
            )
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()
        num_hits = np.sum(self.hit_count)
        if num_hits == 0:
            return 0
        _sort_u(n_thread, self.ind, self.hit_count, self.hit)
        self.hit = self.hit[:num_hits]
        # sort arrays
        arrays_to_sort = ["tri", "ox", "oy", "oz", *ray_vecs, *self.RAY_FIELDS]
        for key in arrays_to_sort:
            setattr(self, key, getattr(self, key)[self.hit])
        self.qz = np.zeros(num_hits, dtype=np.float32, order="C")
        return num_hits


    def next_field_hit(self):
        num_hits = len(self.tri)
        self.ind, n_thread = self.split_ind(num_hits)
        self.vis = np.empty(num_hits, dtype=np.uint32, order="C")
        self.vis_count = np.empty(n_thread, dtype=np.uint32, order="C")
        self.hit_count = np.empty(n_thread, dtype=np.uint32, order="C")
        for key in self.FAR_FIELDS:
            setattr(self, key, np.empty((num_hits, self.ui32[4]), dtype=np.float32, order="C"))
        for i in range(n_thread):
            thread = threading.Thread(
                target=_next_field_hit,
                args=(
                    self.ind[i:],
                    self.ui32, self.fp32, self.wavenum,
                    self.n0x, self.n0y, self.n0z, self.d0,
                    self.n1x, self.n1y, self.n1z, self.d1,
                    self.n2x, self.n2y, self.n2z, self.d2,
                    self.l, self.tri_mat,
                    self.p_te_re, self.p_te_im,
                    self.p_tm_re, self.p_tm_im,
                    self.tri,
                    self.ox, self.oy, self.oz,
                    self.dx, self.dy, self.dz,
                    self.qx, self.qy, self.qz,
                    self.vis, self.vis_count[i:],
                    self.hit, self.hit_count[i:],
                    self.te_x_re, self.te_x_im,
                    self.tm_x_re, self.tm_x_im,
                    self.te_y_re, self.te_y_im,
                    self.tm_y_re, self.tm_y_im,
                    self.ex_x_re, self.ex_x_im,
                    self.ey_x_re, self.ey_x_im,
                    self.ex_y_re, self.ex_y_im,
                    self.ey_y_re, self.ey_y_im,
                ),
            )
            thread.daemon = True
            self.threads.append(thread)
        self.thread_start_join()

        n_vis = np.sum(self.vis_count)
        if n_vis > 0:
            _sort_u(n_thread, self.ind, self.vis_count, self.vis)
            self.vis = self.vis[:n_vis]
            for key in self.FAR_FIELDS:
                setattr(self, key, getattr(self, key)[self.vis])
        else:
            for key in self.FAR_FIELDS:
                setattr(self, key, None)

        num_hits = np.sum(self.hit_count)
        if num_hits == 0:
            return 0
        _sort_u(n_thread, self.ind, self.hit_count, self.hit)
        self.hit = self.hit[:num_hits]
        # sort arrays
        ray_vecs = ["dx", "dy", "dz", "qx", "qy", "qz"]
        arrays_to_sort = ["tri", "ox", "oy", "oz", *ray_vecs, *self.RAY_FIELDS]
        for key in arrays_to_sort:
            setattr(self, key, getattr(self, key)[self.hit])
        return num_hits


    @property
    def ray_spacing(self) -> None:
        return self.fp32[0]

    
    @ray_spacing.setter
    def ray_spacing(self, value: float) -> None:
        self.fp32[0] = value


    @property
    def ray_tracing(self) -> None:
        if self.ui32[5] == 0:
            return False
        return True

    
    @ray_tracing.setter
    def ray_tracing(self, value: bool) -> None:
        if value:
            self.ui32[5] = 1
        else:
            self.ui32[5] = 0


    def main(self, point):
        self.rotate_vertices(point)
        self.precompute_triangles()
        self.init_hit_field()
        n_init_ray = self.ui32[3]

        # temporary solution
        init_keys = ["tri", "ox", "oy", "oz"]
        temp = {}
        for key in init_keys:
            temp[key] = getattr(self, key).copy()
            setattr(self, key, None)

        # from this point work with ray groups
        group_ind, n_groups = self.split_ind(n_init_ray)
        for p in range(n_groups):
            # starting index for rays
            ray_start = group_ind[p]
            # number of rays in a group
            num_hits = group_ind[p + 1] - ray_start
            # treat rays in c dict as usual
            for key in init_keys:
                setattr(self, key, temp[key][ray_start:ray_start+num_hits])
            # initial field & secondary hit points
            # for name in func_iter(c["refl_limit"]):
            #     num_hits = func[name](c)
            #     if num_hits == 0:
            #         break





