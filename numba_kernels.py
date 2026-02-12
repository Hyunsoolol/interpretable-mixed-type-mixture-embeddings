
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def soft_thresh(z, g):
    if z > g:
        return z - g
    if z < -g:
        return z + g
    return 0.0

@njit(cache=True, fastmath=True)
def update_two_abs_1d(a, c, s, lam1, lam2):
    if a < 1e-12:
        return 0.0

    def f(b):
        return 0.5 * a * b * b - c * b + lam1 * abs(b) + lam2 * abs(b + s)

    cand0 = 0.0
    cand1 = -s

    best_b = cand0
    best_v = f(best_b)

    v1 = f(cand1)
    if v1 < best_v:
        best_v = v1
        best_b = cand1

    for sgn1 in (-1.0, 1.0):
        for sgn2 in (-1.0, 1.0):
            b = (c - (lam1 * sgn1 + lam2 * sgn2)) / a

            ok1 = (b > 0.0 and sgn1 == 1.0) or (b < 0.0 and sgn1 == -1.0) or abs(b) < 1e-12
            bp = b + s
            ok2 = (bp > 0.0 and sgn2 == 1.0) or (bp < 0.0 and sgn2 == -1.0) or abs(bp) < 1e-12

            if ok1 and ok2:
                vb = f(b)
                if vb < best_v:
                    best_v = vb
                    best_b = b

    return best_b

@njit(cache=True, fastmath=True)
def lasso_cd_weighted_nb(X, y, w, lam, pen_w, b,
                         maxit=400, tol=1e-6, minit=5):
    n, p = X.shape

    # r = y - Xb
    r = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(p):
            s += X[i, k] * b[k]
        r[i] = y[i] - s

    # a_k = sum_i w_i x_ik^2
    a = np.empty(p, dtype=np.float64)
    for k in range(p):
        s = 0.0
        for i in range(n):
            x = X[i, k]
            s += w[i] * x * x
        a[k] = s

    for it in range(1, maxit + 1):
        maxchg = 0.0

        for k in range(p):
            ak = a[k]
            if ak < 1e-12:
                continue

            ck = 0.0
            bk = b[k]
            for i in range(n):
                x = X[i, k]
                ck += w[i] * x * (r[i] + x * bk)

            bnew = soft_thresh(ck, lam * pen_w[k]) / ak
            d = bnew - bk
            if abs(d) > 0.0:
                b[k] = bnew
                if abs(d) > maxchg:
                    maxchg = abs(d)
                for i in range(n):
                    r[i] -= X[i, k] * d

        if it >= minit and maxchg < tol:
            break

    return b

@njit(cache=True, fastmath=True)
def mstep_hp_cd_nb(y, X, tau, beta0, Bfree, rho,
                   lam, pen_w0, pen_wB,
                   max_cd=400, tol_cd=1e-6, min_cd=5, rebuild_every=10):
    n, p = X.shape
    m = tau.shape[1]

    # Bfull = [Bfree, -rowSums(Bfree)]
    Bfull = np.empty((p, m), dtype=np.float64)
    for k in range(p):
        s = 0.0
        for l in range(m - 1):
            Bfull[k, l] = Bfree[k, l]
            s += Bfree[k, l]
        Bfull[k, m - 1] = -s

    # rmat[i,j] = rho[j]*y[i] - sum_k X[i,k]*(beta0[k] + Bfull[k,j])
    rmat = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            s = 0.0
            for k in range(p):
                s += X[i, k] * (beta0[k] + Bfull[k, j])
            rmat[i, j] = rho[j] * y[i] - s

    # rowsumtau[i] = sum_j tau[i,j]
    rowsumtau = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += tau[i, j]
        rowsumtau[i] = s

    for it in range(1, max_cd + 1):
        maxchg = 0.0

        # ---- update beta0 ----
        for k in range(p):
            a = 0.0
            for i in range(n):
                x = X[i, k]
                a += rowsumtau[i] * x * x
            if a < 1e-12:
                continue

            c = 0.0
            b0k = beta0[k]
            for j in range(m):
                for i in range(n):
                    x = X[i, k]
                    c += tau[i, j] * x * (rmat[i, j] + x * b0k)

            bnew = soft_thresh(c, n * lam * pen_w0[k]) / a
            d = bnew - b0k
            if abs(d) > 0.0:
                beta0[k] = bnew
                if abs(d) > maxchg:
                    maxchg = abs(d)
                for j in range(m):
                    for i in range(n):
                        rmat[i, j] -= X[i, k] * d

        # ---- update Bfree ----
        for l in range(m - 1):
            for k in range(p):
                s_minus = 0.0
                for q in range(m - 1):
                    if q != l:
                        s_minus += Bfree[k, q]

                bcur = Bfree[k, l]

                a = 0.0
                for i in range(n):
                    x = X[i, k]
                    a += (tau[i, l] + tau[i, m - 1]) * x * x
                if a < 1e-12:
                    continue

                c = 0.0
                for i in range(n):
                    x = X[i, k]
                    c += tau[i, l] * x * (rmat[i, l] + x * bcur)
                    c -= tau[i, m - 1] * x * (rmat[i, m - 1] - x * bcur)

                lam1 = n * lam * pen_wB[k, l]
                lam2 = n * lam * pen_wB[k, m - 1]
                bnew = update_two_abs_1d(a, c, s_minus, lam1, lam2)

                d = bnew - bcur
                if abs(d) > 0.0:
                    Bfree[k, l] = bnew
                    if abs(d) > maxchg:
                        maxchg = abs(d)
                    for i in range(n):
                        x = X[i, k]
                        rmat[i, l] -= x * d
                        rmat[i, m - 1] += x * d

        if rebuild_every > 0 and (it % rebuild_every) == 0:
            for k in range(p):
                s = 0.0
                for l in range(m - 1):
                    Bfull[k, l] = Bfree[k, l]
                    s += Bfree[k, l]
                Bfull[k, m - 1] = -s

            for j in range(m):
                for i in range(n):
                    s = 0.0
                    for k in range(p):
                        s += X[i, k] * (beta0[k] + Bfull[k, j])
                    rmat[i, j] = rho[j] * y[i] - s

        if it >= min_cd and maxchg < tol_cd:
            break

    for k in range(p):
        s = 0.0
        for l in range(m - 1):
            Bfull[k, l] = Bfree[k, l]
            s += Bfree[k, l]
        Bfull[k, m - 1] = -s

    return beta0, Bfree, Bfull
