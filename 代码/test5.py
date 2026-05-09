import numpy as np
import pandas as pd
import math
import time as timer 
from pulp import *
from tqdm import tqdm
from collections import defaultdict
import itertools

# 设置初始化常量
G = 9.8; ddsd = 300.0; ymbj = 10.0; ymsc = 20.0; ymcjsd = 3.0
wrj_vmin, wrj_vmax = 70.0, 140.0
jmb = np.array([0.0, 0.0, 0.0]); zmbzx = np.array([0.0, 200.0, 5.0])
mb_bj = 7.0
ddcszb = {'M1': np.array([20000.0, 0.0, 2000.0]), 'M2': np.array([19000.0, 600.0, 2100.0]), 'M3': np.array([18000.0, -600.0, 1900.0])}
wrjcszb = {'FY1': np.array([17800.0, 0.0, 1800.0]), 'FY2': np.array([12000.0, 1400.0, 1400.0]), 'FY3': np.array([6000.0, -3000.0, 700.0]), 'FY4': np.array([11000.0, 2000.0, 1800.0]), 'FY5': np.array([13000.0, -2000.0, 1300.0])}
wrjmz = list(wrjcszb.keys()); ddmz = list(ddcszb.keys())

# 开始进行主程序运行
start_time = timer.time() 
ddxx = {}
for mz, wz in ddcszb.items():
    fx = (jmb - wz) / np.linalg.norm(jmb - wz)
    mz_sj = np.linalg.norm(jmb - wz) / ddsd
    ddxx[mz] = {'cszb': wz, 'sl': ddsd * fx, 'mz_sj': mz_sj}
    print(f"导弹 {mz} 预计在 {mz_sj:.2f} 秒后命中。")    
print("\n阶段一：生成候选战术...")
hxrws = []
z_zhou = np.array([0, 0, 1])
for wrj_m, m_m in tqdm(list(itertools.product(wrjmz, ddmz)), desc="生成候选战术"):
    m_info = ddxx[m_m]
    wrj_h = wrjcszb[wrj_m][2]
    top_rws = []
    for tbao in np.arange(15, m_info['mz_sj'] - 1, 0.3):
        p_dd = m_info['cszb'] + m_info['sl'] * tbao
        sx = zmbzx - p_dd
        if np.linalg.norm(sx) < 1e-6: continue
        sx_norm = sx / np.linalg.norm(sx)
        y_sl = np.cross(sx_norm, z_zhou); y_sl /= (np.linalg.norm(y_sl) + 1e-9)
        s_sl = np.cross(y_sl, sx_norm)
        for alpha in np.linspace(0.05, 0.95, 11):
            p_sx = p_dd + alpha * sx
            for py1 in np.linspace(-ymbj, ymbj, 5):
                for py2 in np.linspace(-ymbj, ymbj, 5):
                    p_bao_hx = p_sx + py1 * y_sl + py2 * s_sl
                    z_bao = p_bao_hx[2]
                    if not (ymbj < z_bao < wrj_h): continue
                    dt2 = 2 * (wrj_h - z_bao) / G
                    if dt2 <= 0: continue
                    dt = math.sqrt(dt2); ttf = tbao - dt
                    if ttf < 1.0: continue
                    fx_jl = np.linalg.norm(p_bao_hx[:2] - wrjcszb[wrj_m][:2])
                    if fx_jl < 1.0: continue
                    sd_req = fx_jl / ttf
                    if wrj_vmin <= sd_req <= wrj_vmax:
                        fx_dir = (p_bao_hx[:2] - wrjcszb[wrj_m][:2]) / fx_jl
                        hxrw = {'wrjm': wrj_m, 'ddm': m_m, 'ttf': ttf, 'tbao': tbao, 'pbao': p_bao_hx, 'sd': sd_req, 'fxdir': fx_dir}
                        yxsc = 0.0
                        for t in np.arange(tbao, tbao + ymsc, 0.1):
                            pym_now = hxrw['pbao'] - np.array([0, 0, ymcjsd * (t - tbao)])
                            pdd_now = m_info['cszb'] + m_info['sl'] * t
                            los_sl = zmbzx - pdd_now; los_len2 = np.dot(los_sl, los_sl)
                            if los_len2 > 1e-9:
                                dd2ym_sl = pym_now - pdd_now
                                t_ty = np.dot(dd2ym_sl, los_sl) / (los_len2 + 1e-9)
                                if 0 <= t_ty <= 1:
                                    dist2 = np.dot(dd2ym_sl, dd2ym_sl) - (t_ty**2) * los_len2
                                    if dist2 < (ymbj + mb_bj)**2: yxsc += 0.1
                        if yxsc > 0.1:
                           hxrw['yxsc'] = yxsc
                           top_rws.append(hxrw); top_rws.sort(key=lambda m: m['yxsc'], reverse=True)
                           if len(top_rws) > 10: top_rws.pop()
    hxrws.extend(top_rws)            
print(f"\n共找到 {len(hxrws)} 个候选战术。")
if hxrws:
    print("\n阶段二：构建并求解MILP模型...")
    dtb = 0.2; jl = 0.001
    max_t = max(info['mz_sj'] for info in ddxx.values())
    sjd_idx = range(int((max_t + ymsc) / dtb))    
    rw_fg = defaultdict(list)
    for i, rw in enumerate(tqdm(hxrws, desc="预计算覆盖范围")):
        dd_mb = rw['ddm']
        for t in np.arange(rw['tbao'], rw['tbao'] + ymsc, dtb):
            pym_now = rw['pbao'] - np.array([0, 0, ymcjsd * (t - rw['tbao'])])
            m_info = ddxx[dd_mb]
            pdd_now = m_info['cszb'] + m_info['sl'] * t
            los_sl = zmbzx - pdd_now; los_len2 = np.dot(los_sl, los_sl)
            if los_len2 > 1e-9:
                dd2ym_sl = pym_now - pdd_now
                t_ty = np.dot(dd2ym_sl, los_sl) / (los_len2 + 1e-9)
                if 0 <= t_ty <= 1:
                    dist2 = np.dot(dd2ym_sl, dd2ym_sl) - (t_ty**2) * los_len2
                    if dist2 < (ymbj + mb_bj)**2:
                        j = int(round(t / dtb))
                        if j in sjd_idx: rw_fg[(j, dd_mb)].append(i)
    wt = LpProblem("UAV_ZSDL_Strategy", LpMaximize)
    x = [LpVariable(f"x_{i}", 0, 1, LpBinary) for i in range(len(hxrws))]
    y = {(j, m): LpVariable(f"y_{j}_{m}", 0, 1, LpBinary) for j in sjd_idx for m in ddmz}
    z = {j: LpVariable(f"z_{j}", 0, 1, LpBinary) for j in sjd_idx}
    wrj_sy = {mz: LpVariable(f"uav_used_{mz}", 0, 1, LpBinary) for mz in wrjmz}
    wt += (lpSum(z.values()) * dtb) + (lpSum(x) * jl), "Maximize_Time_with_Bonus"    
    for u_mz in wrjmz:
        idxs = [i for i, c in enumerate(hxrws) if c['wrjm'] == u_mz]
        wt += lpSum(x[i] for i in idxs) <= 3 * wrj_sy[u_mz]
        wt += wrj_sy[u_mz] <= lpSum(x[i] for i in idxs)
        for i1 in range(len(idxs)):
            for i2 in range(i1 + 1, len(idxs)):
                idx1, idx2 = idxs[i1], idxs[i2]
                if abs(hxrws[idx1]['ttf'] - hxrws[idx2]['ttf']) < 1.0: wt += x[idx1] + x[idx2] <= 1
    for (j, m_mz), rws_idxs in rw_fg.items(): wt += y[(j, m_mz)] <= lpSum(x[i] for i in rws_idxs)
    for j in sjd_idx: wt += z[j] <= lpSum(y[(j, m_mz)] for m_mz in ddmz)
    wt += lpSum(wrj_sy.values()) >= 4
    for m_mz in ddmz:
        idxs = [i for i,c in enumerate(hxrws) if c['ddm'] == m_mz]
        if idxs: wt += lpSum(x[i] for i in idxs) >= 1
    print(" 正在求解MILP模型...")
    wt.solve(PULP_CBC_CMD(msg=0))
    if wt.status == 1:
        zuiyou_rws = [hxrws[i] for i in range(len(hxrws)) if x[i].varValue == 1]
        print(f"\n正在整理最终策略并保存到文件...")
        df = pd.DataFrame([{'无人机编号': u, '烟幕弹编号': i} for u in wrjmz for i in range(1, 4)])
        zuiyou_rws.sort(key=lambda m: (m['wrjm'], m['ttf']))
        wrj_jsq = {u: 1 for u in wrjmz}
        for rw in zuiyou_rws:
            u_mz, bianhao = rw['wrjm'], wrj_jsq[rw['wrjm']]
            if bianhao > 3: continue
            fx_dir = rw['fxdir']; jd = (np.degrees(np.arctan2(fx_dir[1], fx_dir[0])) + 360) % 360
            row_idx = df[(df['无人机编号'] == u_mz) & (df['烟幕弹编号'] == bianhao)].index
            if not row_idx.empty:
                ptf_xy = wrjcszb[u_mz][:2] + fx_dir * rw['sd'] * rw['ttf']
                pbao = rw['pbao']
                df.loc[row_idx, '运动方向'] = f"{jd:.2f}"; df.loc[row_idx, '运动速度(m/s)'] = f"{rw['sd']:.2f}"
                df.loc[row_idx, '投放点x(m)'] = f"{ptf_xy[0]:.2f}"; df.loc[row_idx, '投放点y(m)'] = f"{ptf_xy[1]:.2f}"; df.loc[row_idx, '投放点z(m)'] = f"{wrjcszb[u_mz][2]:.2f}"
                df.loc[row_idx, '起爆点x(m)'] = f"{pbao[0]:.2f}"; df.loc[row_idx, '起爆点y(m)'] = f"{pbao[1]:.2f}"; df.loc[row_idx, '起爆点z(m)'] = f"{pbao[2]:.2f}"
                df.loc[row_idx, '干扰导弹'] = rw['ddm']; df.loc[row_idx, '有效时长(s)'] = f"{rw['yxsc']:.2f}"
            wrj_jsq[u_mz] = bianhao + 1        
        dtb, sjd_map = 0.1, defaultdict(list)
        zb_sjd = {m: np.zeros(int(max_t / dtb) + 2, dtype=bool) for m in ddmz}
        zc_sjd = np.zeros(int(max_t / dtb) + 2, dtype=bool)
        for rw in zuiyou_rws:
            m_mz = rw['ddm']
            for t in np.arange(rw['tbao'], rw['tbao'] + ymsc, dtb):
                p_dd = ddxx[m_mz]['cszb'] + ddxx[m_mz]['sl']*t
                p_ym = rw['pbao'] - np.array([0,0,ymcjsd*(t-rw['tbao'])])
                los = zmbzx-p_dd; len2=np.dot(los,los)
                if len2 > 1e-9:
                    dd2ym = p_ym-p_dd
                    t_ty = np.dot(dd2ym, los)/len2
                    if 0 <= t_ty <= 1 and (np.dot(dd2ym,dd2ym) - t_ty**2 * len2) < (ymbj+mb_bj)**2:
                        idx = int(round(t / dtb))
                        if 0 <= idx < len(zb_sjd[m_mz]):
                            zb_sjd[m_mz][idx] = True; zc_sjd[idx] = True
                        if m_mz not in sjd_map[round(t,2)]: sjd_map[round(t,2)].append(m_mz)
        sc_per_dd = {m: np.sum(sjd) * dtb for m, sjd in zb_sjd.items()}
        z_sc = np.sum(zc_sjd) * dtb        
        print("\n" + "="*50); print("       最终结果汇总"); print("="*50)
        print(f"最优策略动用 {df.dropna(subset=['运动方向'])['无人机编号'].nunique()} 架无人机，共执行 {len(zuiyou_rws)} 次拦截。")
        print(f"\n真实不重复的总时长: {z_sc:.2f} 秒。")
        print("\n各导弹真实有效遮蔽时长分解:")
        for m_mz, m_sc in sc_per_dd.items(): print(f"- 导弹 {m_mz}: {m_sc:.2f} 秒")        
        print("\n--- 详细遮蔽时间轴 (M1:■ M2:▲ M3:▼ | *:多重覆盖) ---")
        sorted_sjs = sorted(sjd_map.keys())
        dd_fh = {'M1': '■', 'M2': '▲', 'M3': '▼'}
        if sorted_sjs:
            for miao, group in itertools.groupby(sorted_sjs, key=lambda t: int(t)):
                hang = f"t = {miao:2d}s |"; fg_map = ['.'] * 10
                for t_val in group:
                    weizhi = int(round((t_val - miao) * 10))
                    if 0 <= weizhi < 10:
                        fgs = sjd_map[t_val]
                        fg_map[weizhi] = '*' if len(fgs) > 1 else dd_fh.get(fgs[0], '?')
                print(hang + "".join(fg_map) + "|")
        print("="*50)
        df.to_excel("result_tactical_efficiency.xlsx", index=False, engine='openpyxl')
        print(f"\n结果已成功保存到 result_tactical_efficiency.xlsx 文件中。")
    else: print("MILP求解失败。")
else: print("未能找到候选战术。")    
end_time = timer.time() 
print(f"\n总计算耗时: {end_time - start_time:.2f} 秒。")