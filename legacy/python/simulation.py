
# =========================
# Simulation driver (FAST SETTINGS)
# =========================
# ==========================================
# [Final] 논문 게재용 최종 시뮬레이션 설정
# ==========================================
CFG = dict(
    # [1. 통계적 유의성 확보]
    # 보통 논문에서는 100회를 표준으로 봅니다. (시간이 너무 없으면 50회)
    R = 100,           

    # [2. 데이터 설정 (고정)]
    n = 200,
    p = 60,
    snr = 50,
    m_set = [3],

    # [3. 정밀 탐색 (Grid Search)] ⭐중요
    # 15개는 너무 적습니다. 50개 이상은 써야 "최적의 모델을 찾았다"고 할 수 있습니다.
    nlambda = 50,      # (Recommended: 50 ~ 100)
    lam_ratio = 1e-4,  # 충분히 작은 람다까지 탐색

    # [4. EM 알고리즘 안정성]
    # 수렴 기준을 엄격하게 잡아야 결과의 신뢰도가 올라갑니다.
    max_em = 100,      # 50 -> 100 (충분한 반복)
    tol_em = 1e-5,     # 1e-4 -> 1e-5 (더 정밀하게)

    # [5. Adaptive Lasso 핵심 (Pilot)]
    # Pilot이 멍청하면 본 게임도 망합니다. 30회 이상은 주세요.
    pilot_em = 30,     # 15 -> 30 (안정적인 가중치 계산)
    
    # [6. 초기값 민감도 제거]
    # Local Optima 방지를 위해 여러 번 시도합니다.
    n_start = 20,      # 10 -> 20 (표준적인 횟수)

    # [7. 내부 최적화 엔진 (Coordinate Descent)]
    max_cd = 500,     # 충분히 길게
    tol_cd = 1e-6,     # 아주 정밀하게
    min_cd = 5,
    rebuild_every = 10
)

def run_sim_fast(CFG,
                 show_grid_bar=True,
                 grid_print_best=True,
                 grid_print_progress=True,
                 em_verbose=False,
                 em_trace_every=10):

    methods = ["Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL"]
    res_rows = []

    print(f"[{now_str()}] Running Simulation(FAST): n={CFG['n']}, p={CFG['p']}, SNR={CFG['snr']}, Reps={CFG['R']} | "
          f"m_set={','.join(map(str,CFG['m_set']))} | nlambda={CFG['nlambda']} | n_start={CFG['n_start']} | "
          f"max_em={CFG['max_em']} | max_cd={CFG['max_cd']}")

    t0 = time.time()

    for r in range(1, CFG["R"]+1):
        print(f"\n[{now_str()}] ====================")
        print(f"[{now_str()}] REP {r}/{CFG['R']} START")
        print(f"[{now_str()}] ====================")

        dat = simulate_dataset(n=CFG["n"], p=CFG["p"], snr=CFG["snr"], seed=2026 + r)
        y = dat["y"]
        X = np.asfortranarray(dat["X"], dtype=np.float64)  # Fortran-order for CD kernels

        for meth in methods:
            print(f"\n[{now_str()}] [REP {r}] method={meth} | BIC grid search START")
            t_m0 = time.time()

            fit = fit_by_bic_2stage(
                y, X, meth,
                m_set=CFG["m_set"],
                nlambda=CFG["nlambda"], lam_ratio=CFG["lam_ratio"],
                n_start=CFG["n_start"], seed_base=1,
                pilot_em=CFG["pilot_em"],
                max_em=CFG["max_em"], tol_em=CFG["tol_em"],
                max_cd=CFG["max_cd"], tol_cd=CFG["tol_cd"], min_cd=CFG["min_cd"],
                rebuild_every=CFG["rebuild_every"],
                show_grid_bar=show_grid_bar,
                grid_print_best=grid_print_best,
                grid_print_progress=grid_print_progress,
                em_verbose=em_verbose,
                em_trace_every=em_trace_every
            )

            if fit is None:
                print(f"[{now_str()}] [REP {r}] method={meth} FAILED (no fit)")
                continue

            fit["method"] = meth
            met = compute_metrics(fit, dat)

            elapsed = time.time() - t_m0
            print(f"[{now_str()}] [REP {r}] method={meth} DONE | chosen(m)={fit['m']} lam={fmt_num(fit['lam'],6)} | "
                  f"em_iter={fit.get('em_iter','NA')} ll={fmt_num(fit['ll'],6)} bic={fmt_num(fit['bic'],4)} df={fit['df']} | "
                  f"elapsed={elapsed:.1f}s")

            res_rows.append(dict(
                rep=r, method=meth, snr=CFG["snr"],
                chosen_m=int(fit["m"]), chosen_lam=float(fit["lam"]),
                ll=float(fit["ll"]), bic=float(fit["bic"]), df=int(fit["df"]), em_iter=int(fit.get("em_iter",-1)),
                mse_b=float(met["mse_b"]), mse_s2=float(met["mse_sigma2"]), mse_pi=float(met["mse_pi"]),
                FPR=float(met["FPR"]), TPR=float(met["TPR"]), FHR=float(met["FHR"]), m_hat=int(met["m_hat"])
            ))

        print(f"\n[{now_str()}] REP {r}/{CFG['R']} END | total_elapsed={time.time()-t0:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n[{now_str()}] ALL DONE | elapsed={total_elapsed:.1f}s")

    return res_rows

# Run
res_rows = run_sim_fast(
    CFG=CFG,
    show_grid_bar=True,
    grid_print_best=True,
    grid_print_progress=True,
    em_verbose=False
)

# Convert to DataFrame
import pandas as pd
out = pd.DataFrame(res_rows)
out
