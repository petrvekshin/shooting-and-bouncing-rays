#include <math.h>
#include <stdio.h>

/* transfom a fresnel array into an array 
with a required number of frequencies */
void f_req(const unsigned long n_th,
const unsigned long n_ref, const unsigned long n_req,
const float* f_ref, const float* f_req,
const float* R_ref, float* restrict R_req) {
    unsigned long i, j, t1, t2;
    float d;
    /* looping over required frequencies */
    for (i = 0; i < n_req; ++i) {
        if (f_req[i] <= f_ref[0]) {
            /* looping over cos_th points */
            for (j = 0; j < n_th; ++j) {
                R_req[i + j * n_req] = R_ref[j * n_ref];
            }
        } else if (f_req[i] >= f_ref[n_ref - 1]) {
            /* looping over cos_th points */
            for (j = 0; j < n_th; ++j) {
                R_req[i + j * n_req] = R_ref[(n_ref - 2) + j * n_ref];
            }
        } else {
            /* linear approximation */
            t2 = 1;
            while (f_req[i] > f_ref[t2]) {
                t2++;
            }
            t1 = t2 - 1;
            d = (f_req[i] - f_ref[t1]) / (f_ref[t2] - f_ref[t1]);
            /* looping over cos_th points */
            for (j = 0; j < n_th; ++j) {
                R_req[i + j * n_req] = R_ref[t1 + j * n_ref] 
                + d * (R_ref[t2 + j * n_ref] - R_ref[t1 + j * n_ref]);
            }
        }
    }
}

/* triangle precomputing for Havel & Herout algorithm */
void tri_nd(const unsigned long* ind,
const unsigned long* v0, const unsigned long* v1, const unsigned long* v2,
const float* vx, const float* vy, const float* vz,
float* restrict n0x, float* restrict n0y, float* restrict n0z, float* restrict d0,
float* restrict n1x, float* restrict n1y, float* restrict n1z, float* restrict d1,
float* restrict n2x, float* restrict n2y, float* restrict n2z, float* restrict d2,
float* restrict l) {
    unsigned long i;
    float e1x, e1y, e1z, e2x, e2y, e2z, l2;
    printf("tri_nd in c: from %lu\n", ind[0]);
    printf("tri_nd in c: to %lu\n", ind[1]);
    printf("tri_nd in c: to %lu\n", ind[2]);
    for (i = ind[0]; i < ind[1]; ++i) {
        e1x = vx[v1[i]] - vx[v0[i]];
        e1y = vy[v1[i]] - vy[v0[i]];
        e1z = vz[v1[i]] - vz[v0[i]];
        e2x = vx[v2[i]] - vx[v0[i]];
        e2y = vy[v2[i]] - vy[v0[i]];
        e2z = vz[v2[i]] - vz[v0[i]];
        n0x[i] = (e1y * e2z) - (e1z * e2y);
        n0y[i] = (e1z * e2x) - (e1x * e2z);
        n0z[i] = (e1x * e2y) - (e1y * e2x);
        printf("n0x=%f, n0y=%f, n0z=%f\n", n0x[i], n0y[i], n0z[i]);
        l2 = 1.0 / ((n0x[i] * n0x[i]) + (n0y[i] * n0y[i]) + (n0z[i] * n0z[i]));
        d0[i] = (n0x[i] * vx[v0[i]]) + (n0y[i] * vy[v0[i]]) + (n0z[i] * vz[v0[i]]);
        n1x[i] = ((e2y * n0z[i]) - (e2z * n0y[i])) * l2;
        n1y[i] = ((e2z * n0x[i]) - (e2x * n0z[i])) * l2;
        n1z[i] = ((e2x * n0y[i]) - (e2y * n0x[i])) * l2;
        d1[i] = -((n1x[i] * vx[v0[i]]) + (n1y[i] * vy[v0[i]]) + (n1z[i] * vz[v0[i]]));
        n2x[i] = ((n0y[i] * e1z) - (n0z[i] * e1y)) * l2;
        n2y[i] = ((n0z[i] * e1x) - (n0x[i] * e1z)) * l2;
        n2z[i] = ((n0x[i] * e1y) - (n0y[i] * e1x)) * l2;
        d2[i] = -((n2x[i] * vx[v0[i]]) + (n2y[i] * vy[v0[i]]) + (n2z[i] * vz[v0[i]]));
        l[i] = pow(l2, 0.5);
    }
    printf("end of tri_nd\n");
}

/* initial hit points */
void init_hit(unsigned long *ui32, float* fp32, unsigned long *tri,
const unsigned long* v0, const unsigned long* v1, const unsigned long* v2,
const float* vx, const float* vy, const float* vz,
const float* n0x, const float* n0y, const float* n0z, const float* d0,
float* restrict ox, float* restrict oy, float* restrict oz) {
    /* fp32 = {0: step, 1: x_min, 2: y_min, 3: t_max, ...}
    ui32 = {0: n_tri, 1: n_x, 2: n_y, 3: n_ray, ...} */
    float ax, ay, bx, by, cx, cy;
    float abx, aby, acx, acy, bcx, bcy;
    float x, y, z, z_inf;
    float x12[2];
    unsigned long i, i1, i2, y_ind, x_ind, j;
    unsigned long n_tri, n_x, n_y, n_ray;
    float x_min, y_min, step, inv_step;
    float xv[ui32[1]], yv[ui32[2]];
    n_tri = ui32[0];
    n_x = ui32[1];
    n_y = ui32[2];
    step = fp32[0];
    x_min = fp32[1];
    y_min = fp32[2];
    inv_step = 1.0 / step;
    z_inf = oz[0];
    /* fill in xv and yv */
    for (i = 0; i < n_x; ++i) {
        xv[i] = x_min + i * step;
    }
    for (i = 0; i < n_y; ++i) {
        yv[i] = y_min + i * step;
    }
    /* loop over triangles */
    for (i = 0; i < n_tri; ++i) {
        /* continue if ray lies in plane of triangle */
        if (n0z[i] == 0.0) {
            continue;
        }
        /* define vertices a, b and c so that ay >= by >= cy */
        if ((vy[v0[i]] >= vy[v1[i]]) && (vy[v0[i]] >= vy[v2[i]])) {
            ax = vx[v0[i]];
            ay = vy[v0[i]];
            if (vy[v1[i]] >= vy[v2[i]]) {
                bx = vx[v1[i]];
                by = vy[v1[i]];
                cx = vx[v2[i]];
                cy = vy[v2[i]];
            } else {
                bx = vx[v2[i]];
                by = vy[v2[i]];
                cx = vx[v1[i]];
                cy = vy[v1[i]];
            }
        } else if (vy[v1[i]] >= vy[v2[i]]) {
            ax = vx[v1[i]];
            ay = vy[v1[i]];
            if (vy[v0[i]] >= vy[v2[i]]) {
                bx = vx[v0[i]];
                by = vy[v0[i]];
                cx = vx[v2[i]];
                cy = vy[v2[i]];
            } else {
                bx = vx[v2[i]];
                by = vy[v2[i]];
                cx = vx[v0[i]];
                cy = vy[v0[i]];
            }
        } else {
            ax = vx[v2[i]];
            ay = vy[v2[i]];
            if (vy[v0[i]] >= vy[v1[i]]) {
                bx = vx[v0[i]];
                by = vy[v0[i]];
                cx = vx[v1[i]];
                cy = vy[v1[i]];
            } else {
                bx = vx[v1[i]];
                by = vy[v1[i]];
                cx = vx[v0[i]];
                cy = vy[v0[i]];
            }
        }
        abx = bx - ax;
        aby = by - ay;
        acx = cx - ax;
        acy = cy - ay;
        bcx = cx - bx;
        bcy = cy - by;
        if ((abx * acy) - (aby * acx) > 0.0) {
            i1 = 1;
            i2 = 0;
        } else {
            i1 = 0;
            i2 = 1;
        }
        /* find ox, oy, oz */
        y_ind = (unsigned long) (floor((ay - y_min) * inv_step));
        y = yv[y_ind];
        if (aby != 0.0) {
            while (y >= by) {
                x12[i1] = ax - ((ay - y) / acy) * acx;
                x12[i2] = ax - ((ay - y) / aby) * abx;
                x_ind = (unsigned long) (floor((x12[1] - x_min) * inv_step));
                x = xv[x_ind];
                while (x >= x12[0]) {
                    z = (d0[i] - ((x * n0x[i]) + (y * n0y[i]))) / n0z[i];
                    j = x_ind * n_y + y_ind;
                    if (z > oz[j]) {
                        oz[j] = z;
                        ox[j] = x;
                        oy[j] = y;
                        tri[j] = i;
                    }
                    x_ind -= 1;
                    x = xv[x_ind];
                }
                y_ind -= 1;
                y = yv[y_ind];
            }
        }
        if (bcy != 0.0) {
            while (y >= cy) {
                x12[i1] = ax - ((ay - y) / acy) * acx;
                x12[i2] = bx - ((by - y) / bcy) * bcx;
                x_ind = (unsigned long) (floor((x12[1] - x_min) * inv_step));
                x = xv[x_ind];
                while (x >= x12[0]) {
                    z = (d0[i] - ((x * n0x[i]) + (y * n0y[i]))) / n0z[i];
                    j = x_ind * n_y + y_ind;
                    if (z > oz[j]) {
                        oz[j] = z;
                        ox[j] = x;
                        oy[j] = y;
                        tri[j] = i;
                    }
                    x_ind -= 1;
                    x = xv[x_ind];
                }
                y_ind -= 1;
                y = yv[y_ind];
            }
        }
    }
    /* count number of rays & rearrange ox, oy, oz & tri */
    n_ray = 0;
    for (j = 0; j < n_x * n_y; ++j) {
        if (oz[j] > z_inf) {
            ox[n_ray] = ox[j];
            oy[n_ray] = oy[j];
            oz[n_ray] = oz[j];
            tri[n_ray] = tri[j];
            n_ray++;
        }
    }
    ui32[3] = n_ray;
}

/* add function init_field */
void init_field(const unsigned long* ind,
const unsigned long* ui32, const float* fp32, const float* wavenum,
const float* n0x, const float* n0y, const float* n0z, const float* d0,
const float* n1x, const float* n1y, const float* n1z, const float* d1,
const float* n2x, const float* n2y, const float* n2z, const float* d2,
const float* l, const unsigned long* tri_mat,
const unsigned long long* p_te_re, const unsigned long long* p_te_im,
const unsigned long long* p_tm_re, const unsigned long long* p_tm_im,
unsigned long* restrict tri,
float* restrict ox, float* restrict oy, float* restrict oz,
float* restrict ex_x_re, float* restrict ex_x_im,
float* restrict ey_x_re, float* restrict ey_x_im,
float* restrict ex_y_re, float* restrict ex_y_im,
float* restrict ey_y_re, float* restrict ey_y_im) {
    /* fp32 = {0: step, 1: x_min, 2: y_min, 3: t_max,
    4: S/(2*pi), 5: min(cos_th), 6: max(S/(2*pi*cos_th)),
    7: 2*(n_th-1.0)/pi}
    ui32 = {0: n_tri, 1: n_x, 2: n_y, 3: n_ray, 4: n_freq} */
    unsigned long i, n_freq;
    unsigned long mat, row, th_row, f, row_f, th_row_f;
    float sin_q, cos_q, cos_th, inv_len;
    float *p_te_re_m, *p_te_im_m, *p_tm_re_m, *p_tm_im_m;
    float wavenum_area, a, b, TE_re, TE_im, TM_re, TM_im;
    float Eq_re, Eq_im, En_re, En_im;
    float Eq_x_re, Eq_x_im, En_x_re, En_x_im;
    float Eq_y_re, Eq_y_im, En_y_re, En_y_im;
    float dist_i, exp_arg, re_exp, im_exp;

    n_freq = ui32[4];
    /* looping over rays */
    for (i = ind[0]; i < ind[1]; ++i) {
        /* calculate starting index row for 2D arrays */
        row = i * n_freq;
        /* traveled distance for field at r = 1m */
        dist_i = (2.0 * oz[i]) - 1.0;
        /* compute far fields */
        if (tri[i] < tri_mat[0]) {
            /* PEC case. Far fields are easy to compute */
            for (f = 0; f < n_freq; ++f) {
                row_f = row + f;
                exp_arg = wavenum[f] * dist_i;
                re_exp = cos(exp_arg);
                im_exp = sin(exp_arg);
                wavenum_area = wavenum[f] * fp32[4];
                a = im_exp * wavenum_area;
                b = - re_exp * wavenum_area;
                ex_x_re[row_f] = a;
                ex_x_im[row_f] = b;
                ey_x_re[row_f] = 0.0;
                ey_x_im[row_f] = 0.0;
                ex_y_re[row_f] = 0.0;
                ex_y_im[row_f] = 0.0;
                ey_y_re[row_f] = a;
                ey_y_im[row_f] = b;
            }
        } else {
            /* find material index mat */
            mat = 1;
            while (tri[i] >= tri_mat[mat]) { mat++; }
            mat -= 1;
            /* get pointers to Fresnel coefficients of material */
            p_te_re_m = (float*) p_te_re[mat];
            p_te_im_m = (float*) p_te_im[mat];
            p_tm_re_m = (float*) p_tm_re[mat];
            p_tm_im_m = (float*) p_tm_im[mat];
            /* find cos_th & th_row */
            cos_th = fabs(n0z[tri[i]]) * l[tri[i]];
            if (cos_th > 1.0) { cos_th = 1.0; }
            th_row = n_freq * ((unsigned long) round(acos(cos_th) * fp32[7]));
            /* find cos_q & sin_q */
            if ((n0x[tri[i]] != 0.0) || (n0y[tri[i]] != 0.0)) {
                inv_len = 1.0 / pow((n0x[tri[i]] * n0x[tri[i]]) + (n0y[tri[i]] * n0y[tri[i]]), 0.5);
                if (n0z[tri[i]] > 0.0) {
                    cos_q = - n0y[tri[i]] * inv_len;
                    sin_q = n0x[tri[i]] * inv_len;
                } else {
                    cos_q = n0y[tri[i]] * inv_len;
                    sin_q = - n0x[tri[i]] * inv_len;
                }
            } else {
                cos_q = 1.0;
                sin_q = 0.0;
            }
            for (f = 0; f < n_freq; ++f) {
                row_f = row + f;
                th_row_f = th_row + f;
                TE_re = p_te_re_m[th_row_f];
                TE_im = p_te_im_m[th_row_f];
                TM_re = p_tm_re_m[th_row_f];
                TM_im = p_tm_im_m[th_row_f];
                exp_arg = wavenum[f] * dist_i;
                re_exp = cos(exp_arg);
                im_exp = sin(exp_arg);
                wavenum_area = wavenum[f] * fp32[4];
                a = im_exp * wavenum_area;
                b = - re_exp * wavenum_area;
                Eq_re = (- a * TE_re) - (- b * TE_im);
                Eq_im = (- a * TE_im) + (- b * TE_re);
                En_re = (a * TM_re) - (b * TM_im);
                En_im = (a * TM_im) + (b * TM_re);
                Eq_x_re = Eq_re * cos_q;
                Eq_x_im = Eq_im * cos_q;
                Eq_y_re = Eq_re * sin_q;
                Eq_y_im = Eq_im * sin_q;
                En_x_re = En_re * sin_q;
                En_x_im = En_im * sin_q;
                En_y_re = - En_re * cos_q;
                En_y_im = - En_im * cos_q;
                ex_x_re[row_f] = (Eq_x_re * cos_q) + (En_x_re * sin_q);
                ex_x_im[row_f] = (Eq_x_im * cos_q) + (En_x_im * sin_q);
                ey_x_re[row_f] = (Eq_x_re * sin_q) - (En_x_re * cos_q);
                ey_x_im[row_f] = (Eq_x_im * sin_q) - (En_x_im * cos_q);
                ex_y_re[row_f] = (Eq_y_re * cos_q) + (En_y_re * sin_q);
                ex_y_im[row_f] = (Eq_y_im * cos_q) + (En_y_im * sin_q);
                ey_y_re[row_f] = (Eq_y_re * sin_q) - (En_y_re * cos_q);
                ey_y_im[row_f] = (Eq_y_im * sin_q) - (En_y_im * cos_q);
            }
        }
    }
}


/* compute initial far fields & get secondary hit points */
void init_pol_hit(const unsigned long* ind,
const unsigned long* ui32, const float* fp32, const float* wavenum,
const float* n0x, const float* n0y, const float* n0z, const float* d0,
const float* n1x, const float* n1y, const float* n1z, const float* d1,
const float* n2x, const float* n2y, const float* n2z, const float* d2,
const float* l, const unsigned long* tri_mat,
const unsigned long long* p_te_re, const unsigned long long* p_te_im,
const unsigned long long* p_tm_re, const unsigned long long* p_tm_im,
unsigned long* restrict tri,
float* restrict ox, float* restrict oy, float* restrict oz,
float* restrict dx, float* restrict dy, float* restrict dz,
float* restrict qx, float* restrict qy,
unsigned long* restrict hit, unsigned long* restrict hit_count,
float* restrict te_x_re, float* restrict te_x_im,
float* restrict tm_x_re, float* restrict tm_x_im,
float* restrict te_y_re, float* restrict te_y_im,
float* restrict tm_y_re, float* restrict tm_y_im) {
    /* fp32 = {0: step, 1: x_min, 2: y_min, 3: t_max,
    4: S/(2*pi), 5: min(cos_th), 6: max(S/(2*pi*cos_th)),
    7: 2*(n_th-1.0)/pi}
    ui32 = {0: n_tri, 1: n_x, 2: n_y, 3: n_ray, 4: n_freq} */
    unsigned long i, j, i_hit, tri_i, n_freq;
    unsigned long mat, row, th_row, row_count, f, row_f, th_row_f;
    float sin_q, cos_q, cos_th;
    float P1x, P1y, P1z, det, u, det_u, v, t, t_i, k;
    float inv_len_Q;
    float *p_te_re_m, *p_te_im_m, *p_tm_re_m, *p_tm_im_m;
    float Eq_re, Eq_im, En_re, En_im;
    float dist_i, exp_arg, re_exp, im_exp;
    float TE_re, TE_im, TM_re, TM_im;

    n_freq = ui32[4];
    /* looping over rays */
    i_hit = ind[0];
    for (i = ind[0]; i < ind[1]; ++i) {
        /* normal incidence */
        if ((n0x[tri[i]] == 0.0) && (n0y[tri[i]] == 0.0)) { continue; }
        /* find direction (dx[i], dy[i], dz[i]) of reflected ray */
        k = 2.0 * l[tri[i]] * l[tri[i]] * n0z[tri[i]];
        dx[i] = k * n0x[tri[i]];
        dy[i] = k * n0y[tri[i]];
        dz[i] = k * n0z[tri[i]] - 1.0;
        cos_th = fabs(n0z[tri[i]]) * l[tri[i]];
        th_row = n_freq * ((unsigned long) round(acos(cos_th) * fp32[7]));
        /* Q is normal to the plane of incidence */
        inv_len_Q = 1.0 / pow((dx[i] * dx[i]) + (dy[i] * dy[i]), 0.5);
        qx[i] = - dy[i] * inv_len_Q;
        qy[i] = dx[i] * inv_len_Q;
        /* perform ray-tracing (Havel & Herout algorithm) */
        t_i = fp32[3];
        /* looping over triangles skipping tri[i] */
        for (j = 0; j < tri[i]; ++j) {
            det = (dx[i] * n0x[j]) + (dy[i] * n0y[j]) + (dz[i] * n0z[j]);
            if (det == 0.0) {
                continue;
            }
            t = d0[j] - ((ox[i] * n0x[j]) + (oy[i] * n0y[j]) + (oz[i] * n0z[j]));
            if (t * (det * t_i - t) <= 0.0) {
                continue;
            }
            P1x = (det * ox[i]) + (t * dx[i]);
            P1y = (det * oy[i]) + (t * dy[i]);
            P1z = (det * oz[i]) + (t * dz[i]);
            u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (det * d1[j]);
            det_u = det - u;
            if (u * det_u < 0.0) {
                continue;
            }
            v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (det * d2[j]);
            if (v * (det_u - v) < 0.0) {
                continue;
            }
            t_i = t / det;
            tri_i = j;
        }
        for (j = tri[i] + 1; j < ui32[0]; ++j) {
            det = (dx[i] * n0x[j]) + (dy[i] * n0y[j]) + (dz[i] * n0z[j]);
            if (det == 0.0) {
                continue;
            }
            t = d0[j] - ((ox[i] * n0x[j]) + (oy[i] * n0y[j]) + (oz[i] * n0z[j]));
            if (t * (det * t_i - t) <= 0.0) {
                continue;
            }
            P1x = (det * ox[i]) + (t * dx[i]);
            P1y = (det * oy[i]) + (t * dy[i]);
            P1z = (det * oz[i]) + (t * dz[i]);
            u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (det * d1[j]);
            det_u = det - u;
            if (u * det_u < 0.0) {
                continue;
            }
            v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (det * d2[j]);
            if (v * (det_u - v) < 0.0) {
                continue;
            }
            t_i = t / det;
            tri_i = j;
        }
        /* if there is no hit */
        if (t_i == fp32[3]) { continue; }
        dist_i = oz[i] - t_i;
        cos_q = qx[i];
        sin_q = qy[i];
        /* calculate starting index row for 2D arrays */
        row = i * n_freq;
        /* compute TE & TM coefficients */
        if (tri[i] < tri_mat[0]) {
            /* PEC case */
            /* looping over frequencies */
            for (f = 0; f < n_freq; ++f) {
                row_f = row + f;
                exp_arg = wavenum[f] * dist_i;
                re_exp = cos(exp_arg);
                im_exp = sin(exp_arg);
                /* x-polarization */
                te_x_re[row_f] = - re_exp * cos_q;
                te_x_im[row_f] = - im_exp * cos_q;
                tm_x_re[row_f] = re_exp * sin_q;
                tm_x_im[row_f] = im_exp * sin_q;
                /* y-polarization */
                te_y_re[row_f] = - tm_x_re[row_f];
                te_y_im[row_f] = - tm_x_im[row_f];
                tm_y_re[row_f] = te_x_re[row_f];
                tm_y_im[row_f] = te_x_im[row_f];
            }
        } else {
            /* find material index mat */
            mat = 1;
            while (tri[i] >= tri_mat[mat]) { mat++; }
            mat -= 1;
            /* treat as tri_mat[mat] */
            p_te_re_m = (float*) p_te_re[mat];
            p_te_im_m = (float*) p_te_im[mat];
            p_tm_re_m = (float*) p_tm_re[mat];
            p_tm_im_m = (float*) p_tm_im[mat];
            /* looping over frequencies */
            for (f = 0; f < n_freq; ++f) {
                th_row_f = th_row + f;
                row_f = row + f;
                exp_arg = wavenum[f] * dist_i;
                re_exp = cos(exp_arg);
                im_exp = sin(exp_arg);
                TE_re = p_te_re_m[th_row_f];
                TE_im = p_te_im_m[th_row_f];
                TM_re = p_tm_re_m[th_row_f];
                TM_im = p_tm_im_m[th_row_f];
                Eq_re = (re_exp * TE_re) - (im_exp * TE_im);
                Eq_im = (re_exp * TE_im) + (im_exp * TE_re);
                En_re = (re_exp * TM_re) - (im_exp * TM_im);
                En_im = (re_exp * TM_im) + (im_exp * TM_re);
                /* x-polarization */
                te_x_re[row_f] = Eq_re * cos_q;
                te_x_im[row_f] = Eq_im * cos_q;
                tm_x_re[row_f] = En_re * sin_q;
                tm_x_im[row_f] = En_im * sin_q;
                /* y-polarization */
                te_y_re[row_f] = Eq_re * sin_q;
                te_y_im[row_f] = Eq_im * sin_q;
                tm_y_re[row_f] = - En_re * cos_q;
                tm_y_im[row_f] = - En_im * cos_q;
            }
        }

        /* compute values for next ray-tracing */
        hit[i_hit] = i;
        tri[i] = tri_i;
        ox[i] = ox[i] + (dx[i] * t_i);
        oy[i] = oy[i] + (dy[i] * t_i);
        oz[i] = oz[i] + (dz[i] * t_i);
        i_hit++;
    }
    /* number of hits */
    hit_count[0] = i_hit - ind[0];
}


/* compute far fields & get next hit points */
void next_field_hit(const unsigned long* ind,
const unsigned long* ui32, const float* fp32, const float* wavenum,
const float* n0x, const float* n0y, const float* n0z, const float* d0,
const float* n1x, const float* n1y, const float* n1z, const float* d1,
const float* n2x, const float* n2y, const float* n2z, const float* d2,
const float* l, const unsigned long* tri_mat,
const unsigned long long* p_te_re, const unsigned long long* p_te_im,
const unsigned long long* p_tm_re, const unsigned long long* p_tm_im,
unsigned long* restrict tri,
float* restrict ox, float* restrict oy, float* restrict oz,
float* restrict dx, float* restrict dy, float* restrict dz,
float* restrict qx, float* restrict qy, float* restrict qz,
unsigned long* restrict vis, unsigned long* restrict vis_count,
unsigned long* restrict hit, unsigned long* restrict hit_count,
float* restrict te_x_re, float* restrict te_x_im,
float* restrict tm_x_re, float* restrict tm_x_im,
float* restrict te_y_re, float* restrict te_y_im,
float* restrict tm_y_re, float* restrict tm_y_im,
float* restrict ex_x_re, float* restrict ex_x_im,
float* restrict ey_x_re, float* restrict ey_x_im,
float* restrict ex_y_re, float* restrict ex_y_im,
float* restrict ey_y_re, float* restrict ey_y_im) {
    unsigned long i, j, i_vis, i_hit, tri_i, vis_i, n_freq;
    unsigned long mat, row, th_row, f, row_f, th_row_f;
    float dx_i, dy_i, dz_i, sin_q, cos_q;
    float cos_inc, cos_obs, dif_cos_1, dif_cos_2;
    float P1x, P1y, P1z, det, u, det_u, v, t, t_i, k, k_n0x, k_n0y, k_n0z;
    float Qx, Qy, Qz, inv_len_Q, Lx, Ly, Lz, inv_len_L, len_L_2;
    float *p_te_re_m, *p_te_im_m, *p_tm_re_m, *p_tm_im_m;
    float a1, b1, a2, b2, Eq_re, Eq_im, En_re, En_im;
    float area, exp_arg, re_exp, im_exp;
    n_freq = ui32[4];
    float u1_re[n_freq], u1_im[n_freq], u2_re[n_freq], u2_im[n_freq];
    float tt1, tt2, tt3, tt4, nx, ny, nz;

    /* looping over rays */
    i_hit = ind[0];
    i_vis = ind[0];
    for (i = ind[0]; i < ind[1]; ++i) {
        /* calculate starting index row for 2D arrays */
        row = i * n_freq;
        /* find direction (dx_i, dy_i, dz_i) of reflected ray */
        nx = n0x[tri[i]] * l[tri[i]];
        ny = n0y[tri[i]] * l[tri[i]];
        nz = n0z[tri[i]] * l[tri[i]];
        k = 2.0 * l[tri[i]] * l[tri[i]] * ((dx[i] * n0x[tri[i]]) 
        + (dy[i] * n0y[tri[i]]) + (dz[i] * n0z[tri[i]]));
        k_n0x = k * n0x[tri[i]];
        k_n0y = k * n0y[tri[i]];
        k_n0z = k * n0z[tri[i]];
        dx_i = dx[i] - k_n0x;
        dy_i = dy[i] - k_n0y;
        dz_i = dz[i] - k_n0z;
        /* cos for observation point */
        cos_obs = fabs(n0z[tri[i]]) * l[tri[i]];
        /* vis_i = 0 if (ox, oy, oz) is blocked */
        if (k_n0z >= 0.0) {
            vis_i = 0;
        } else {
            vis_i = 1;
            /* cos_obs is negative in the region of incidence */
            cos_obs = - cos_obs;
        }
        /* set vis_i to 1 anyway to check shadow region */
        vis_i = 1;

        t_i = fp32[3];
        /* perform ray-tracing if needed */
        if (ui32[5] == 1) {
            /* looping over triangles skipping tri[i] */
            for (j = 0; j < tri[i]; ++j) {
                t = d0[j] - ((ox[i] * n0x[j]) + (oy[i] * n0y[j]) + (oz[i] * n0z[j]));
                /* visibility check */
                if ((vis_i != 0) && (n0z[j] * t > 0.0)) {
                    P1x = (n0z[j] * ox[i]);
                    P1y = (n0z[j] * oy[i]);
                    P1z = (n0z[j] * oz[i]) + t;
                    u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (n0z[j] * d1[j]);
                    det_u = det - u;
                    if (u * det_u >= 0.0) {
                        v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (n0z[j] * d2[j]);
                        if (v * (det_u - v) >= 0.0) { vis_i = 0; }
                    }
                }
                /* next hit point */
                det = (dx_i * n0x[j]) + (dy_i * n0y[j]) + (dz_i * n0z[j]);
                if ((det == 0.0) || (t * (det * t_i - t) <= 0.0)) { continue; }
                P1x = (det * ox[i]) + (t * dx_i);
                P1y = (det * oy[i]) + (t * dy_i);
                P1z = (det * oz[i]) + (t * dz_i);
                u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (det * d1[j]);
                det_u = det - u;
                if (u * det_u < 0.0) { continue; }
                v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (det * d2[j]);
                if (v * (det_u - v) < 0.0) { continue; }
                t_i = t / det;
                tri_i = j;
            }
            for (j = tri[i] + 1; j < ui32[0]; ++j) {
                t = d0[j] - ((ox[i] * n0x[j]) + (oy[i] * n0y[j]) + (oz[i] * n0z[j]));
                /* visibility check */
                if ((vis_i != 0) && (n0z[j] * t > 0.0)) {
                    P1x = (n0z[j] * ox[i]);
                    P1y = (n0z[j] * oy[i]);
                    P1z = (n0z[j] * oz[i]) + t;
                    u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (n0z[j] * d1[j]);
                    det_u = det - u;
                    if (u * det_u >= 0.0) {
                        v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (n0z[j] * d2[j]);
                        if (v * (det_u - v) >= 0.0) { vis_i = 0; }
                    }
                }
                /* next hit point */
                det = (dx_i * n0x[j]) + (dy_i * n0y[j]) + (dz_i * n0z[j]);
                if ((det == 0.0) || (t * (det * t_i - t) <= 0.0)) { continue; }
                P1x = (det * ox[i]) + (t * dx_i);
                P1y = (det * oy[i]) + (t * dy_i);
                P1z = (det * oz[i]) + (t * dz_i);
                u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (det * d1[j]);
                det_u = det - u;
                if (u * det_u < 0.0) { continue; }
                v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (det * d2[j]);
                if (v * (det_u - v) < 0.0) { continue; }
                t_i = t / det;
                tri_i = j;
            }
        } else if (vis_i == 1) {
            /* perform ray tracing to check visibility only */
            for (j = 0; j < ui32[0]; ++j) {
                if (j == tri[i]) { continue; }
                t = d0[j] - ((ox[i] * n0x[j]) + (oy[i] * n0y[j]) + (oz[i] * n0z[j]));
                /* visibility check */
                if (n0z[j] * t > 0.0) {
                    P1x = (n0z[j] * ox[i]);
                    P1y = (n0z[j] * oy[i]);
                    P1z = (n0z[j] * oz[i]) + t;
                    u = (P1x * n1x[j]) + (P1y * n1y[j]) + (P1z * n1z[j]) + (n0z[j] * d1[j]);
                    det_u = det - u;
                    if (u * det_u >= 0.0) {
                        v = (P1x * n2x[j]) + (P1y * n2y[j]) + (P1z * n2z[j]) + (n0z[j] * d2[j]);
                        if (v * (det_u - v) >= 0.0) {
                            vis_i = 0;
                            break;
                        }
                    }
                }
            }
        }
        /* if there is no need to compute TE & TM fields */
        if ((vis_i == 0) && (t_i == fp32[3])) { continue; }
        /* calculate values needed for computation of TE & TM fields */
        Lx = dx[i] + dx_i;
        Ly = dy[i] + dy_i;
        Lz = dz[i] + dz_i;
        len_L_2 = (Lx * Lx) + (Ly * Ly) + (Lz * Lz);
        /* exclude normal incidence */
        if (len_L_2 > 0.000001) {
            /* Q is normal to the plane of incidence */
            Qx = (dy_i * dz[i]) - (dz_i * dy[i]);
            Qy = (dz_i * dx[i]) - (dx_i * dz[i]);
            Qz = (dx_i * dy[i]) - (dy_i * dx[i]);
            inv_len_Q = 1.0 / pow((Qx * Qx) + (Qy * Qy) + (Qz * Qz), 0.5);
            Qx = Qx * inv_len_Q;
            Qy = Qy * inv_len_Q;
            Qz = Qz * inv_len_Q;
            /* cos_q & sin_q determine angle between current Q & previous Q */
            cos_q = (qx[i] * Qx) + (qy[i] * Qy) + (qz[i] * Qz);
            if (i == 0) {
                printf("cos_q = %f\n", cos_q);
                printf("q x n = %f\n", (qx[i] * nx) + (qy[i] * ny) + (qz[i] * nz));
            }
            sin_q = pow((1.0 - cos_q * cos_q), 0.5);
            if (((qx[i] * Lx) + (qy[i] * Ly) + (qz[i] * Lz)) < 0.0) {
                sin_q = - sin_q;
            }
            qx[i] = Qx;
            qy[i] = Qy;
            qz[i] = Qz;
            /* normalize L & find cos_inc & area */
            if (len_L_2 >= 4.0) {
                cos_inc = 0.0;
            } else {
                cos_inc = pow((1.0 - (0.25 * len_L_2)), 0.5);
            }
            if (cos_inc > fp32[5]) {
                area = fp32[4] / cos_inc;
            } else {
                area = fp32[6];
            }
            inv_len_L = 1.0 / pow(len_L_2, 0.5);
            Lx = Lx * inv_len_L;
            Ly = Ly * inv_len_L;
            /* decompose TE & TM fields according to new Q vector */
            for (f = row; f < row + n_freq; ++f) {
                /* x-polarization */
                Eq_re = te_x_re[f] * cos_q - tm_x_re[f] * sin_q;
                Eq_im = te_x_im[f] * cos_q - tm_x_im[f] * sin_q;
                En_re = te_x_re[f] * sin_q + tm_x_re[f] * cos_q;
                En_im = te_x_im[f] * sin_q + tm_x_im[f] * cos_q;
                te_x_re[f] = Eq_re;
                te_x_im[f] = Eq_im;
                tm_x_re[f] = En_re;
                tm_x_im[f] = En_im;
                /* y-polarization */
                Eq_re = te_y_re[f] * cos_q - tm_y_re[f] * sin_q;
                Eq_im = te_y_im[f] * cos_q - tm_y_im[f] * sin_q;
                En_re = te_y_re[f] * sin_q + tm_y_re[f] * cos_q;
                En_im = te_y_im[f] * sin_q + tm_y_im[f] * cos_q;
                te_y_re[f] = Eq_re;
                te_y_im[f] = Eq_im;
                tm_y_re[f] = En_re;
                tm_y_im[f] = En_im;
            }
        } else {
            cos_inc = 1.0;
            area = fp32[4];
            Lx = (qy[i] * dz_i) - (qz[i] * dy_i);
            Ly = (qz[i] * dx_i) - (qx[i] * dz_i);
        }
        // if (k_n0z < 0.0) {
        //     /* visible region */
        //     dif_cos_1 = (cos_inc - cos_obs) * area / 2.0;
        //     dif_cos_2 = (cos_inc + cos_obs) * area / 2.0;
        // } else {
        //     /* shadow region */
        //     dif_cos_1 = (cos_inc + cos_obs) * area / 2.0;
        //     dif_cos_2 = (cos_inc - cos_obs) * area / 2.0;
        // }
        dif_cos_1 = (cos_inc - cos_obs) * area / 2.0;
        dif_cos_2 = (cos_inc + cos_obs) * area / 2.0;
        /* compute shadow part contributions to far fields */
        for (f = 0; f < n_freq; ++f) {
            exp_arg = wavenum[f] * (oz[i] - 1.0);
            re_exp = cos(exp_arg) * wavenum[f];
            im_exp = sin(exp_arg) * wavenum[f];
            u1_re[f] = - im_exp * dif_cos_1;
            u1_im[f] = re_exp * dif_cos_1;
            u2_re[f] = - im_exp * dif_cos_2;
            u2_im[f] = re_exp * dif_cos_2;
            row_f = row + f;
            /* x-polarization */
            Eq_re = (te_x_re[row_f] * u2_re[f]) - (te_x_im[row_f] * u2_im[f]);
            Eq_im = (te_x_re[row_f] * u2_im[f]) + (te_x_im[row_f] * u2_re[f]);
            En_re = (tm_x_re[row_f] * u2_re[f]) - (tm_x_im[row_f] * u2_im[f]);
            En_im = (tm_x_re[row_f] * u2_im[f]) + (tm_x_im[row_f] * u2_re[f]);
            // if (i == 0) {
            //     printf("En_re = %f\n", En_re);
            //     printf("En_im = %f\n", En_im);
            //     printf("qx = %f\n", qx[i]);
            //     printf("qy = %f\n", qy[i]);
            //     printf("Lx = %f\n", Lx);
            //     printf("Ly = %f\n", Ly);

            // }
            tt1 = (qx[i] - Lx);
            tt2 = (-qx[i] + Lx);
            
            ex_x_re[row_f] = (tt1 * En_re) + (tt2 * Eq_re);
            ex_x_im[row_f] = (tt1 * En_im) + (tt2 * Eq_im);
            // ex_x_re[row_f] = (qy[i] * En_re) + (Ly * Eq_re);
            // ex_x_im[row_f] = (qy[i] * En_im) + (Ly * Eq_im);
            ey_x_re[row_f] = (- qy[i] * Eq_re) + (- Ly * En_re);
            ey_x_im[row_f] = (- qy[i] * Eq_im) + (- Ly * En_im);
            /* y-polarization */
            Eq_re = (te_y_re[row_f] * u2_re[f]) - (te_y_im[row_f] * u2_im[f]);
            Eq_im = (te_y_re[row_f] * u2_im[f]) + (te_y_im[row_f] * u2_re[f]);
            En_re = (tm_y_re[row_f] * u2_re[f]) - (tm_y_im[row_f] * u2_im[f]);
            En_im = (tm_y_re[row_f] * u2_im[f]) + (tm_y_im[row_f] * u2_re[f]);
            ex_y_re[row_f] = (- qx[i] * Eq_re) + (- Lx * En_re);
            ex_y_im[row_f] = (- qx[i] * Eq_im) + (- Lx * En_im);
            // ey_y_re[row_f] = (- qy[i] * Eq_re);
            // ey_y_im[row_f] = (- qy[i] * Eq_im);
            tt1 = (-qx[i] + nx);
            tt2 = (-Lx - nx);
            ey_y_re[row_f] = (0.0 * En_re) + (tt2 * Eq_re);
            ey_y_im[row_f] = (0.0 * En_im) + (tt2 * Eq_im);
            tt3 = En_re / Eq_re;
            tt4 = En_im / Eq_im;
            if (i == 0) {
                printf("en/eq re = %f\n", tt3);
                printf("en/eq im = %f\n", tt4);
            }
        }

        /* in case of covering */
        if (tri[i] >= tri_mat[0]) {
            th_row = n_freq * ((unsigned long) round(acos(cos_inc) * fp32[7]));
            /* find material index mat */
            mat = 1;
            while (tri[i] >= tri_mat[mat]) { mat++; }
            mat -= 1;
            /* treat as tri_mat[mat] */
            p_te_re_m = (float*) p_te_re[mat];
            p_te_im_m = (float*) p_te_im[mat];
            p_tm_re_m = (float*) p_tm_re[mat];
            p_tm_im_m = (float*) p_tm_im[mat];
            for (f = 0; f < n_freq; ++f) {
                th_row_f = th_row + f;
                row_f = row + f;
                /* x-polarization */
                Eq_re = (te_x_re[row_f] * p_te_re_m[th_row_f]) - (te_x_im[row_f] * p_te_im_m[th_row_f]);
                Eq_im = (te_x_re[row_f] * p_te_im_m[th_row_f]) + (te_x_im[row_f] * p_te_re_m[th_row_f]);
                En_re = (tm_x_re[row_f] * p_tm_re_m[th_row_f]) - (tm_x_im[row_f] * p_tm_im_m[th_row_f]);
                En_im = (tm_x_re[row_f] * p_tm_im_m[th_row_f]) + (tm_x_im[row_f] * p_tm_re_m[th_row_f]);
                te_x_re[row_f] = - Eq_re;
                te_x_im[row_f] = - Eq_im;
                tm_x_re[row_f] = En_re;
                tm_x_im[row_f] = En_im;
                /* y-polarization */
                Eq_re = (te_y_re[row_f] * p_te_re_m[th_row_f]) - (te_y_im[row_f] * p_te_im_m[th_row_f]);
                Eq_im = (te_y_re[row_f] * p_te_im_m[th_row_f]) + (te_y_im[row_f] * p_te_re_m[th_row_f]);
                En_re = (tm_y_re[row_f] * p_tm_re_m[th_row_f]) - (tm_y_im[row_f] * p_tm_im_m[th_row_f]);
                En_im = (tm_y_re[row_f] * p_tm_im_m[th_row_f]) + (tm_y_im[row_f] * p_tm_re_m[th_row_f]);
                te_y_re[row_f] = - Eq_re;
                te_y_im[row_f] = - Eq_im;
                tm_y_re[row_f] = En_re;
                tm_y_im[row_f] = En_im;
            }
        }

        /* if there is no hit */
        if (t_i == fp32[3]) { continue; }

        /* compute values for next ray-tracing */
        hit[i_hit] = i;
        tri[i] = tri_i;
        ox[i] = ox[i] + (dx_i * t_i);
        oy[i] = oy[i] + (dy_i * t_i);
        oz[i] = oz[i] + (dz_i * t_i);
        dx[i] = dx_i;
        dy[i] = dy_i;
        dz[i] = dz_i;
        for (f = 0; f < n_freq; ++f) {
            row_f = row + f;
            exp_arg = - wavenum[f] * t_i;
            re_exp = cos(exp_arg);
            im_exp = sin(exp_arg);
            /* x-polarization */
            Eq_re = (te_x_re[row_f] * re_exp) - (te_x_im[row_f] * im_exp);
            Eq_im = (te_x_re[row_f] * im_exp) + (te_x_im[row_f] * re_exp);
            En_re = (tm_x_re[row_f] * re_exp) - (tm_x_im[row_f] * im_exp);
            En_im = (tm_x_re[row_f] * im_exp) + (tm_x_im[row_f] * re_exp);
            te_x_re[row_f] = Eq_re;
            te_x_im[row_f] = Eq_im;
            tm_x_re[row_f] = En_re;
            tm_x_im[row_f] = En_im;
            /* y-polarization */
            Eq_re = (te_y_re[row_f] * re_exp) - (te_y_im[row_f] * im_exp);
            Eq_im = (te_y_re[row_f] * im_exp) + (te_y_im[row_f] * re_exp);
            En_re = (tm_y_re[row_f] * re_exp) - (tm_y_im[row_f] * im_exp);
            En_im = (tm_y_re[row_f] * im_exp) + (tm_y_im[row_f] * re_exp);
            te_y_re[row_f] = Eq_re;
            te_y_im[row_f] = Eq_im;
            tm_y_re[row_f] = En_re;
            tm_y_im[row_f] = En_im;
        }
        i_hit++;
    }
    /* number of hits */
    vis_count[0] = i_vis - ind[0];
    hit_count[0] = i_hit - ind[0];
}

/* sort float array */
void sort_f(const unsigned long n, const unsigned long* i_start_end,
const unsigned long* hit_count, float* restrict fp_array) {
    unsigned long i, j, nr;
    if (n > 1) {
        nr = hit_count[0];
        for (i = 1; i < n; ++i) {
            for (j = 0; j < hit_count[i]; ++j) {
                fp_array[nr] = fp_array[i_start_end[i] + j];
                nr++;
            }
        }
    }
}

/* sort unsigned long array */
void sort_u(const unsigned long n, const unsigned long* i_start_end,
const unsigned long* hit_count, unsigned long* restrict ui_array) {
    unsigned long i, j, nr;
    if (n > 1) {
        nr = hit_count[0];
        for (i = 1; i < n; ++i) {
            for (j = 0; j < hit_count[i]; ++j) {
                ui_array[nr] = ui_array[i_start_end[i] + j];
                nr++;
            }
        }
    }
}

/* sort float 2D array */
void sort_f_2d(const unsigned long n, const unsigned long n_freq,
const unsigned long* i_start_end, const unsigned long* hit_count,
float* restrict fp_array) {
    unsigned long i, j, nr, r, r2, f;
    if (n > 1) {
        nr = hit_count[0];
        for (i = 1; i < n; ++i) {
            for (j = 0; j < hit_count[i]; ++j) {
                r = nr * n_freq;
                r2 = (i_start_end[i] + j) * n_freq;
                for (f = 0; f < n_freq; ++f) {
                    fp_array[r + f] = fp_array[r2 + f];
                }
                nr++;
            }
        }
    }
}

