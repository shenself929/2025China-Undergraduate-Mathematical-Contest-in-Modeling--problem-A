import numpy as np
import pandas as pd
import math
import time as timer # 使用别名避免pulp库的命名冲突
from pulp import *
from tqdm import tqdm
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置初始化常量
G = 9.8; ddsd = 300.0; ymbj = 10.0; ymsc = 20.0; ymcjsd = 3.0
wrj_vmin, wrj_vmax = 70.0, 140.0
jmb = np.array([0.0, 0.0, 0.0]); zmbzx = np.array([0.0, 200.0, 5.0])
mb_bj = 7.0
ddcszb = {'M1': np.array([20000.0, 0.0, 2000.0]), 'M2': np.array([19000.0, 600.0, 2100.0]), 'M3': np.array([18000.0, -600.0, 1900.0])}
wrjcszb = {'FY1': np.array([17800.0, 0.0, 1800.0]), 'FY2': np.array([12000.0, 1400.0, 1400.0]), 'FY3': np.array([6000.0, -3000.0, 700.0]), 'FY4': np.array([11000.0, 2000.0, 1800.0]), 'FY5': np.array([13000.0, -2000.0, 1300.0])}
wrjmz = list(wrjcszb.keys()); ddmz = list(ddcszb.keys())

# 开始进行主流程计算
kaishi_sj = timer.time()
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
    m_info = ddxx[m_m]; wrj_h = wrjcszb[wrj_m][2]
    top_rws = []
    for tbao in np.arange(15, m_info['mz_sj'] - 1, 0.3):
        p_dd = m_info['cszb'] + m_info['sl'] * tbao
        sx = zmbzx - p_dd
        if np.linalg.norm(sx) < 1e-6: continue
        sx_norm = sx / np.linalg.norm(sx)
        y_sl = np.cross(sx_norm, z_zhou); y_sl /= (np.linalg.norm(y_sl) + 1e-9)
        s_sl = np.cross(y_sl, sx_norm)
        for alpha in np.linspace(0.05, 0.95, 11):
            for py1 in np.linspace(-ymbj, ymbj, 5):
                for py2 in np.linspace(-ymbj, ymbj, 5):
                    p_bao_hx = (p_dd + alpha * sx) + py1 * y_sl + py2 * s_sl
                    if not (ymbj < p_bao_hx[2] < wrj_h): continue
                    dt2 = 2 * (wrj_h - p_bao_hx[2]) / G
                    if dt2 <= 0: continue
                    dt = math.sqrt(dt2); ttf = tbao - dt
                    if ttf < 1.0: continue
                    fx_jl = np.linalg.norm(p_bao_hx[:2] - wrjcszb[wrj_m][:2])
                    if fx_jl < 1.0: continue
                    sd_req = fx_jl / ttf
                    if wrj_vmin <= sd_req <= wrj_vmax:
                        hxrw = {'wrjm': wrj_m, 'ddm': m_m, 'ttf': ttf, 'tbao': tbao, 'pbao': p_bao_hx, 'sd': sd_req, 'fxdir': (p_bao_hx[:2] - wrjcszb[wrj_m][:2]) / fx_jl}
                        yxsc = 0.0
                        for t in np.arange(tbao, tbao + ymsc, 0.1):
                            pym_now = hxrw['pbao'] - np.array([0, 0, ymcjsd * (t - tbao)])
                            pdd_now = m_info['cszb'] + m_info['sl'] * t
                            los_sl = zmbzx - pdd_now; los_len2 = np.dot(los_sl, los_sl)
                            if los_len2 > 1e-9:
                                t_ty = np.dot(pym_now - pdd_now, los_sl) / (los_len2 + 1e-9)
                                if 0 <= t_ty <= 1 and (np.dot(pym_now - pdd_now, pym_now - pdd_now) - (t_ty**2) * los_len2) < (ymbj + mb_bj)**2:
                                    yxsc += 0.1
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
                t_ty = np.dot(pym_now - pdd_now, los_sl) / (los_len2 + 1e-9)
                if 0 <= t_ty <= 1 and (np.dot(pym_now - pdd_now, pym_now - pdd_now) - (t_ty**2) * los_len2) < (ymbj + mb_bj)**2:
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
                if abs(hxrws[idxs[i1]]['ttf'] - hxrws[idxs[i2]]['ttf']) < 1.0: wt += x[idxs[i1]] + x[idxs[i2]] <= 1
    for (j, m_mz), rws_idxs in rw_fg.items(): wt += y[(j, m_mz)] <= lpSum(x[i] for i in rws_idxs)
    for j in sjd_idx: wt += z[j] <= lpSum(y[(j, m_mz)] for m_mz in ddmz)
    wt += lpSum(wrj_sy.values()) >= 4
    for m_mz in ddmz:
        if any(c['ddm'] == m_mz for c in hxrws): wt += lpSum(x[i] for i,c in enumerate(hxrws) if c['ddm'] == m_mz) >= 1
    print(" 正在求解MILP模型...")
    wt.solve(PULP_CBC_CMD(msg=0))
    if wt.status == 1:
        zuiyou_rws = [hxrws[i] for i in range(len(hxrws)) if x[i].varValue == 1]
        print(f"\n正在整理最终策略...")        
        suoyou_lie = ['无人机编号', '烟幕弹编号', '运动方向', '运动速度(m/s)', '投放点x(m)', '投放点y(m)', '投放点z(m)',
                       '起爆点x(m)', '起爆点y(m)', '起爆点z(m)', '干扰导弹', '有效时长(s)']
        chushi_shuju = [{'无人机编号': u, '烟幕弹编号': i} for u in wrjmz for i in range(1, 4)]
        df = pd.DataFrame(chushi_shuju, columns=suoyou_lie)
        zuiyou_rws.sort(key=lambda m: (m['wrjm'], m['ttf']))       
        for rw in zuiyou_rws:
            u_mz = rw['wrjm']
            yifenpei_shuliang = len(df.loc[(df['无人机编号'] == u_mz) & (df['干扰导弹'].notna())])
            dangqian_bianhao = yifenpei_shuliang + 1
            if dangqian_bianhao > 3: continue
            ptf_xy = wrjcszb[u_mz][:2] + rw['fxdir'] * rw['sd'] * rw['ttf']
            jd = (np.degrees(np.arctan2(rw['fxdir'][1], rw['fxdir'][0])) + 360) % 360
            row_mask = (df['无人机编号'] == u_mz) & (df['烟幕弹编号'] == dangqian_bianhao)
            df.loc[row_mask, ['运动方向', '运动速度(m/s)', '投放点x(m)', '投放点y(m)', '投放点z(m)','起爆点x(m)', '起爆点y(m)', '起爆点z(m)','干扰导弹','有效时长(s)']] = \
                [f"{jd:.2f}", f"{rw['sd']:.2f}", f"{ptf_xy[0]:.2f}", f"{ptf_xy[1]:.2f}", f"{wrjcszb[u_mz][2]:.2f}", f"{rw['pbao'][0]:.2f}", f"{rw['pbao'][1]:.2f}", f"{rw['pbao'][2]:.2f}", rw['ddm'], f"{rw['yxsc']:.2f}"]
        dtb2, sjd_map = 0.1, defaultdict(list)
        zb_sjd = {m: np.zeros(int(max_t / dtb2) + 2, dtype=bool) for m in ddmz}
        zc_sjd = np.zeros(int(max_t / dtb2) + 2, dtype=bool)
        for rw in zuiyou_rws:
            m_mz = rw['ddm']
            for t in np.arange(rw['tbao'], rw['tbao'] + ymsc, dtb2):
                p_dd,p_ym = ddxx[m_mz]['cszb'] + ddxx[m_mz]['sl']*t, rw['pbao'] - np.array([0,0,ymcjsd*(t-rw['tbao'])])
                los = zmbzx-p_dd; len2 = np.dot(los,los)
                if len2 > 1e-9 and 0<=np.dot(p_ym-p_dd,los)/len2<=1 and (np.dot(p_ym-p_dd,p_ym-p_dd) - (np.dot(p_ym-p_dd,los))**2/len2)<(ymbj+mb_bj)**2:
                    idx = int(round(t / dtb2))
                    zb_sjd[m_mz][idx], zc_sjd[idx] = True, True
                    if m_mz not in sjd_map[round(t,2)]: sjd_map[round(t,2)].append(m_mz)
        sc_per_dd, z_sc = {m: np.sum(sjd) * dtb2 for m, sjd in zb_sjd.items()}, np.sum(zc_sjd) * dtb2
        
        print("\n" + "="*50 + "\n       最终结果汇总\n" + "="*50)
        print(f"最优策略动用 {df.dropna(subset=['运动方向'])['无人机编号'].nunique()} 架无人机，共执行 {len(zuiyou_rws)} 次拦截。")
        print(f"\n真实不重复的总时长: {z_sc:.2f} 秒。")
        print("\n各导弹真实有效遮蔽时长分解:")
        for m_mz, m_sc in sc_per_dd.items(): print(f"- 导弹 {m_mz}: {m_sc:.2f} 秒")
        
        print("\n--- 详细遮蔽时间轴 (M1:■ M2:▲ M3:▼ | *:多重覆盖) ---")
        for miao, group in itertools.groupby(sorted(sjd_map.keys()), key=lambda t: int(t)):
            hang = f"t = {miao:2d}s |"; fg_map = ['.'] * 10
            for t_val in group:
                fg_map[int(round((t_val - miao) * 10))] = '*' if len(sjd_map[t_val]) > 1 else {'M1': '■', 'M2': '▲', 'M3': '▼'}.get(sjd_map[t_val][0], '?')
            print(hang + "".join(fg_map) + "|")
        print("="*50)
        df.to_excel("result_tactical_efficiency.xlsx", index=False, engine='openpyxl')
        print(f"\n结果已成功保存到 result_tactical_efficiency.xlsx 文件中。")

        # 可视化结果显示
        print("\n阶段四：正在生成可视化图表...")
        plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
        ys = ['gold', 'orange', 'orangered', 'red', 'darkred']
        ys_map = {mz: ys for mz, ys in zip(wrjmz, ys)}        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)); fig1.suptitle("战役成果汇总", fontsize=24, weight='bold')
        ax1.bar(sc_per_dd.keys(), sc_per_dd.values()); ax1.set_title("各导弹有效遮蔽时长", fontsize=18); ax1.set_ylabel("时长 (秒)", fontsize=14)
        for i, (k, v) in enumerate(sc_per_dd.items()): ax1.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=12)
        rw_counts = pd.Series([rw['wrjm'] for rw in zuiyou_rws]).value_counts()
        pie_ys = [ys_map.get(u, '#CCCCCC') for u in rw_counts.index]
        wedges, texts, autotexts = ax2.pie(rw_counts, labels=rw_counts.index, autopct='%1.1f%%', startangle=140, colors=pie_ys, wedgeprops=dict(width=0.4, edgecolor='w'))
        plt.setp(autotexts, size=12, weight="bold", color="white"); ax2.set_title("各无人机任务分配", fontsize=18)       
        fig2, ax_gantt = plt.subplots(figsize=(16, 6)); fig2.suptitle("作战时间轴", fontsize=20, weight='bold')
        ax_gantt.set_yticks(range(len(ddmz))); ax_gantt.set_yticklabels(ddmz, fontsize=14)
        biaoji = set()
        for i, m_mz in enumerate(ddmz):
            for rw in [r for r in zuiyou_rws if r['ddm'] == m_mz]:
                b = f"由 {rw['wrjm']} 执行"
                ax_gantt.barh(i, ymsc, left=rw['tbao'], height=0.6, color=ys_map.get(rw['wrjm']), edgecolor='black', alpha=0.8, label=b if b not in biaoji else "")
                biaoji.add(b)
        ax_gantt.set_xlabel("时间 (秒)", fontsize=14); ax_gantt.grid(axis='x', linestyle='--', alpha=0.6); ax_gantt.legend(fontsize=12, loc='upper right')

        fig3 = plt.figure(figsize=(12, 10)); ax_3d = fig3.add_subplot(111, projection='3d'); fig3.suptitle('3D战场静态总览', fontsize=20, weight='bold')
        for m_mz, m_info in ddxx.items(): ax_3d.plot([m_info['cszb'][0], jmb[0]], [m_info['cszb'][1], jmb[1]], [m_info['cszb'][2], jmb[2]], '-', label=f'导弹 {m_mz} 轨迹', lw=2)
        plotted_labels = {'uav_paths': set(), 'det_points': False}
        for rw in zuiyou_rws:
            u_mz, p_bao = rw['wrjm'], rw['pbao']; u_start = wrjcszb[u_mz]
            ptf_xy = u_start[:2] + rw['fxdir'] * rw['sd'] * rw['ttf']
            path_label = f'{u_mz} 飞行路径'
            ax_3d.plot([u_start[0], ptf_xy[0]], [u_start[1], ptf_xy[1]], [u_start[2], u_start[2]], '--', color=ys_map.get(u_mz), lw=1.5,
                       label=path_label if u_mz not in plotted_labels['uav_paths'] else "")
            plotted_labels['uav_paths'].add(u_mz)
            det_label = '起爆点'
            ax_3d.scatter(p_bao[0], p_bao[1], p_bao[2], color=ys_map.get(u_mz), marker='*', s=150, edgecolor='black', 
                          label=det_label if not plotted_labels['det_points'] else "")
            plotted_labels['det_points'] = True          
        ax_3d.scatter(zmbzx[0], zmbzx[1], zmbzx[2], c='blue', marker='X', s=200, label='我方目标'); ax_3d.scatter(jmb[0], jmb[1], jmb[2], c='gray', marker='s', s=100, label='预期命中点')
        ax_3d.set_xlabel('X(m)'); ax_3d.set_ylabel('Y(m)'); ax_3d.set_zlabel('Z(m)')
        h, l = ax_3d.get_legend_handles_labels(); ax_3d.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
        print("\n所有图表已生成。")
        plt.show()
    else: print("MILP求解失败。")
else: print("未能找到候选战术。")    
jieshu_sj = timer.time()
print(f"\n总计算耗时: {jieshu_sj - kaishi_sj:.2f} 秒。")