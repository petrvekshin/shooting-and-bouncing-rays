from ntpath import join
import sys, os
from time import time
import numpy as np
from functions.field_hit import SBR


write_to_file = True

s = SBR("./geometry/Setup.txt")
s.num_threads = 2 * os.cpu_count()
print(s.tri_mat)

model = "tri_bi"
if write_to_file:
    for i in range(len(s.wavenum)):
        with open(f"./Data/{1+9*i}GHz_{model}.txt", "w") as f:
            pass
# print(c["colors"])
t01 = time()
for i in range(s.theta_phi.shape[0]):
    s.point = i
    theta, phi = s.theta_phi[i, :]
    s.theta = theta
    s.phi = phi
    count = 1
    num_hits = s.init_hit_field()
    print("    ", s.num_init_hits)
    ex_x = np.sum(s.ex_x_re, 0) + 1j*np.sum(s.ex_x_im, 0)
    ey_x = np.sum(s.ey_x_re, 0) + 1j*np.sum(s.ey_x_im, 0)
    ex_y = np.sum(s.ex_y_re, 0) + 1j*np.sum(s.ex_y_im, 0)
    ey_y = np.sum(s.ey_y_re, 0) + 1j*np.sum(s.ey_y_im, 0)
    # for key in s.far_fields:
    #     setattr(s, key, None)
    if s.refl_limit > 1:
        # s.ray_tracing = True
        # num_ray = s.init_pol_hit()
        if num_hits > 0:
            for j in range(2, s.refl_limit + 1):
                # if j == s.refl_limit:
                #     s.ray_tracing = False
                count += 1
                num_hits = s.next_field_hit()
                if s.ex_x_re is not None:
                    ex_x += np.sum(s.ex_x_re, 0) + 1j*np.sum(s.ex_x_im, 0)
                    ey_x += np.sum(s.ey_x_re, 0) + 1j*np.sum(s.ey_x_im, 0)
                    ex_y += np.sum(s.ex_y_re, 0) + 1j*np.sum(s.ex_y_im, 0)
                    ey_y += np.sum(s.ey_y_re, 0) + 1j*np.sum(s.ey_y_im, 0)
                if num_hits == 0:
                    break

    print(f"th={theta}, ph={phi}: {count}")
    lines = s.fields_to_lines(ex_x, ey_x, ex_y, ey_y)
    if not write_to_file:
        continue
    for i in range(len(s.wavenum)):
        line = " ".join([str(num) for num in lines[i,:]])
        with open(f"./Data/{1+9*i}GHz_{model}.txt", "a") as f:
            f.write(str(theta) + " " + line + "\n")
    # print("   ", c["ui32"][3], num_hits)
    
t02 = time() - t01
print(t02)


