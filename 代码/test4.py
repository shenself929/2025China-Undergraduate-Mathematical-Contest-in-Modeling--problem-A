import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time
import math

# 全局常量和初始条件
G = 9.8
ddsd = 300.0
ymbj = 10.0
ymsc = 20.0
ymcjsd = 3.0
zmbzx = np.array([0.0, 200.0, 5.0])
jmb = np.array([0.0, 0.0, 0.0])
ddcszb = np.array([20000.0, 0.0, 2000.0])
wrjcszb_dict = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}
wrjmz = list(wrjcszb_dict.keys())
wrjcszb = np.array(list(wrjcszb_dict.values()))
ddfx = (jmb - ddcszb) / np.linalg.norm(jmb - ddcszb)
ddsl = ddsd * ddfx
mjsj = np.linalg.norm(ddcszb - jmb) / ddsd
print(f"导弹预计飞行总时长: {mjsj:.2f} 秒")

#目标函数的迭代过程
def mbhs(cs, wrj_zb_arr, dtb):
    cs_arr = cs.reshape(3, 4)
    tbaosj_arr = np.zeros(3); pbaosj_arr = np.zeros((3, 3))
    for i in range(3):
        th, v, tt, yxsj = cs_arr[i]
        tbao = tt + yxsj
        p_wrj0 = wrj_zb_arr[i]
        v_wrj = np.array([v * math.cos(th), v * math.sin(th), 0])
        ptf = p_wrj0 + v_wrj * tt
        pbao = ptf + np.array([v_wrj[0]*yxsj, v_wrj[1]*yxsj, -0.5*G*yxsj**2])        
        # 约束条件设定
        if tbao >= mjsj or pbao[2] <= 0: return 1e9
        ddwz_tbao = ddcszb + ddsl * tbao
        if pbao[0] > ddwz_tbao[0]: return 1e9        
        tbaosj_arr[i] = tbao; pbaosj_arr[i,:] = pbao    
    t_sort = np.sort(tbaosj_arr)
    cf = (t_sort[1] - t_sort[0] - ymsc)**2 + (t_sort[2] - t_sort[1] - ymsc)**2
    t_arr = np.arange(0, mjsj, dtb)
    shifou_zb = np.zeros(len(t_arr), dtype=bool)
    for i in range(3):
        tbao, pbao = tbaosj_arr[i], pbaosj_arr[i]
        ks = max(0, int(tbao / dtb))
        js = min(len(t_arr), int((tbao + ymsc) / dtb) + 1)
        for k in range(ks, js):
            if shifou_zb[k]: continue
            pdd = ddcszb + ddsl * t_arr[k]
            pym = pbao - np.array([0, 0, ymcjsd * (t_arr[k] - tbao)])            
            # 内联点到线段距离计算
            ab = zmbzx - pdd; ap = pym - pdd
            dot_ab_ab = np.dot(ab, ab)
            if dot_ab_ab < 1e-9: dist = np.linalg.norm(ap)
            else:
                ratio = np.clip(np.dot(ap, ab) / dot_ab_ab, 0, 1)
                dist = np.linalg.norm(pym - (pdd + ratio * ab))
            if dist <= ymbj: shifou_zb[k] = True    
    zong_sc = np.sum(shifou_zb) * dtb
    return -(zong_sc - cf * 0.01)

# 开始进行计算
fw1 = [(-math.pi, math.pi), (70, 140), (1.0, 30.0), (0.1, 25.0)]
fw2 = [(-math.pi, math.pi), (70, 140), (10.0, 45.0), (0.1, 25.0)]
fw3 = [(-math.pi, math.pi), (70, 140), (20.0, 60.0), (0.1, 25.0)]
fw = fw1 + fw2 + fw3    
start_t = time.time()

# 优化器调用，显示迭代过程
jg = differential_evolution(
    mbhs, fw, args=(wrjcszb, 0.5), strategy='best1bin', maxiter=1000, popsize=25,
    tol=1e-5, mutation=(0.7, 1.3), recombination=0.9, workers=1, updating='immediate', disp=True
)    
end_t = time.time()
print(f'\n优化完成！总耗时: {(end_t - start_t)/60:.2f} 分钟')
zuiyou_cs = jg.x
    
# 输出最后结果
print("\n正在计算最终的精确遮蔽时间...")
t_arr_jingque = np.arange(0, mjsj, 0.01)

# 进行协同总时长的计算
zong_zb_pts = np.zeros(len(t_arr_jingque), dtype=bool)
for i in range(3):
    th, v, tt, yxsj = zuiyou_cs.reshape(3, 4)[i]
    tbao = tt + yxsj
    v_wrj = np.array([v * math.cos(th), v * math.sin(th), 0])
    ptf = wrjcszb[i] + v_wrj * tt
    pbao = ptf + np.array([v_wrj[0]*yxsj, v_wrj[1]*yxsj, -0.5*G*yxsj**2])
    if tbao >= mjsj or pbao[2] <= 0: continue
    ks = max(0, int(tbao / 0.01))
    js = min(len(t_arr_jingque), int((tbao + ymsc) / 0.01) + 1)
    for k in range(ks, js):
        if zong_zb_pts[k]: continue
        pdd = ddcszb + ddsl * t_arr_jingque[k]
        pym = pbao - np.array([0, 0, ymcjsd * (t_arr_jingque[k] - tbao)])
        ab = zmbzx - pdd; ap = pym - pdd
        dot_ab_ab = np.dot(ab, ab)
        if dot_ab_ab < 1e-9: dist = np.linalg.norm(ap)
        else:
            ratio = np.clip(np.dot(ap, ab) / dot_ab_ab, 0, 1)
            dist = np.linalg.norm(pym - (pdd + ratio * ab))
        if dist <= ymbj: zong_zb_pts[k] = True
zong_sc = np.sum(zong_zb_pts) * 0.01
print('\n--- 最优策略 ---')
print(f'最长总遮蔽时间 (协同): {zong_sc:.4f} s\n')
print("详细策略参数：")
shuchu_data = []    
for i in range(3):
    th, v, tt, yxsj = zuiyou_cs.reshape(3, 4)[i]
    tbao = tt + yxsj
    v_wrj = np.array([v * math.cos(th), v * math.sin(th), 0])
    ptf = wrjcszb[i] + v_wrj * tt
    pbao = ptf + np.array([v_wrj[0]*yxsj, v_wrj[1]*yxsj, -0.5*G*yxsj**2])    
    # 计算单机贡献时长
    danji_zb_pts = np.zeros(len(t_arr_jingque), dtype=bool)
    if tbao < mjsj and pbao[2] > 0:
        ks = max(0, int(tbao / 0.01)); js = min(len(t_arr_jingque), int((tbao + ymsc) / 0.01) + 1)
        for k in range(ks, js):
            pdd = ddcszb + ddsl * t_arr_jingque[k]
            pym = pbao - np.array([0, 0, ymcjsd * (t_arr_jingque[k] - tbao)])
            ab = zmbzx - pdd; ap = pym - pdd
            dot_ab_ab = np.dot(ab, ab)
            if dot_ab_ab < 1e-9: dist = np.linalg.norm(ap)
            else:
                ratio = np.clip(np.dot(ap, ab) / dot_ab_ab, 0, 1)
                dist = np.linalg.norm(pym - (pdd + ratio * ab))
            if dist <= ymbj: danji_zb_pts[k] = True
    danji_sc = np.sum(danji_zb_pts) * 0.01
    print(f"  无人机 {wrjmz[i]}:")
    print(f"    飞行方向 (角度): {np.degrees(th):.2f}°")
    print(f"    飞行速度: {v:.2f} m/s")
    print(f"    投放时间: {tt:.2f} s")
    print(f"    起爆时间: {tbao:.2f} s")
    print(f"    投放点坐标 (x,y,z): ({ptf[0]:.1f}, {ptf[1]:.1f}, {ptf[2]:.1f})")
    print(f"    起爆点坐标 (x,y,z): ({pbao[0]:.1f}, {pbao[1]:.1f}, {pbao[2]:.1f})")
    print(f"    无人机有效干扰时长: {danji_sc:.4f} s ")
    print("-" * 30)
    shuchu_data.append({
        "无人机编号": wrjmz[i], "无人机运动方向": f"{np.degrees(th):.2f}°", "无人机运动速度(m/s)": v,
        "烟幕干扰弹投放点x坐标(m)": ptf[0], "烟幕干扰弹投放点y坐标(m)": ptf[1], "烟幕干扰弹投放点z坐标(m)": ptf[2],
        "烟幕干扰弹起爆点x坐标(m)": pbao[0], "烟幕干扰弹起爆点y坐标(m)": pbao[1], "烟幕干扰弹起爆点z坐标(m)": pbao[2],
        "投放时间(s)": tt, "起爆时间(s)": tbao, "单机有效干扰时长(s)": danji_sc,
    })        
df_excel = pd.DataFrame(shuchu_data)
df_excel["总有效干扰时长(s)"] = zong_sc
df_excel = df_excel[[
    "无人机编号", "无人机运动方向", "无人机运动速度(m/s)", "投放时间(s)", "起爆时间(s)", 
    "单机有效干扰时长(s)", "总有效干扰时长(s)", "烟幕干扰弹投放点x坐标(m)", "烟幕干扰弹投放点y坐标(m)", 
    "烟幕干扰弹投放点z坐标(m)", "烟幕干扰弹起爆点x坐标(m)", "烟幕干扰弹起爆点y坐标(m)", "烟幕干扰弹起爆点z坐标(m)"
]]    
df_excel.to_excel('result2.xlsx', index=False, engine='openpyxl', float_format="%.4f")
print('\n结果已成功保存到 result2.xlsx 文件中。')