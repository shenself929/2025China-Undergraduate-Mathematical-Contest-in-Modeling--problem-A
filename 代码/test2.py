import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#常规变量
G = 9.8
DT = 0.02
mbb = np.array([0.0, 200.0, 0.0])      # 目标的坐标
mbr, mbh = 7.0, 10.0                   # 目标半径和高度
mbzx = mbb + np.array([0.0, 0.0, mbh / 2.0])
wrjp0 = np.array([17800.0, 0.0, 1800.0])# 无人机的初始位置
ddp0 = np.array([20000.0, 0.0, 2000.0]) # 导弹的初始位置
ddv = 300.0
ddfx = (np.zeros(3) - ddp0) / np.linalg.norm(np.zeros(3) - ddp0)
ddvvec = ddv * ddfx
ymr, ymvcj, ymsc = 10.0, 3.0, 20.0

# 2. 核心计算函数
def suan_sc(x, dt_eval, nr, nth, nh):
    th, vv, td, tf = x[0], x[1], x[2], x[3]
    tbao = td + tf     
    vwrj = np.array([vv * np.cos(th), vv * np.sin(th), 0.0])
    ptf = wrjp0 + vwrj * td
    pbao = ptf + vwrj * tf + np.array([0.0, 0.0, -0.5 * G * (tf ** 2)])
    aaa = np.linspace(0, mbr, nr)
    bbb = np.linspace(0, 2*np.pi, nth, endpoint=False)
    ccc = np.linspace(mbb[2], mbb[2] + mbh, nh)
    ddd = [[mbb[0]+r*np.cos(t_), mbb[1]+r*np.sin(t_), z] for z in ccc for r in aaa for t_ in bbb]
    mbpts = np.array(ddd)    
    t0 = max(0.0, tbao)
    t1 = tbao + ymsc
    if t1 <= t0: return 0.0
    zcsc = 0.0 
    for t in np.arange(t0, t1 + 1e-9, dt_eval):
        pdd = ddp0 + ddvvec * t
        pym = pbao + np.array([0., 0., -ymvcj * (t - tbao)])        
        aaa1 = mbpts - pdd
        bbb1 = np.einsum('ij,ij->i', aaa1, aaa1)
        ccc1 = np.where(bbb1 == 0.0, 1.0, bbb1)
        ddd1 = np.einsum('j,ij->i', pym - pdd, aaa1) / ccc1
        ddd1 = np.clip(ddd1, 0.0, 1.0)
        eee1 = pdd + ddd1[:, None] * aaa1
        fff1 = np.linalg.norm(pym - eee1, axis=1)        
        if np.max(fff1) <= ymr:
            zcsc += dt_eval
    return zcsc

# 开始进行PSO
W, C1, C2 = 0.72, 1.49, 1.49
N, maxit, seed = 70, 120, 42
bounds = np.array([[-np.pi, np.pi], [70.0, 140.0], [0.0, 40.0], [0.5, 8.0]])
rng = np.random.default_rng(seed)
pos = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N, 4))
vel = rng.uniform(-np.abs(bounds[:,1]-bounds[:,0]), np.abs(bounds[:,1]-bounds[:,0]), size=(N, 4)) * 0.1
fitness = np.full(N, np.inf)
pbestx, pbestf = pos.copy(), fitness.copy()
gbestx, gbestf = pos[0].copy(), np.inf
lishi = []
print("PSO开始迭代...")
for it in range(maxit):
    for i in range(N):
        th, vv, td, tf = pos[i,:]
        tbao = td + tf
        vwrj = np.array([vv * np.cos(th), vv * np.sin(th), 0.0])
        ptf = wrjp0 + vwrj * td
        pbao = ptf + vwrj * tf + np.array([0.0, 0.0, -0.5 * G * (tf**2)])
        t0, t1 = max(0.0, tbao), tbao + ymsc
        if t1 <= t0:
            fitness[i] = 1e9; continue
        aaa = np.linspace(0,mbr,3); bbb = np.linspace(0,2*np.pi,16,endpoint=False); ccc = np.linspace(mbb[2],mbb[2]+mbh,6)
        ddd = [[mbb[0]+r*np.cos(t_), mbb[1]+r*np.sin(t_), z] for z in ccc for r in aaa for t_ in bbb]
        mbpts = np.array(ddd)
        zcsc1, zcsc2, mind = 0.0, 0.0, 1e9
        for t in np.arange(t0, t1 + 1e-9, 0.06):
            pdd = ddp0 + ddvvec * t
            pym = pbao + np.array([0, 0, -ymvcj * (t - tbao)])
            aaa1 = mbpts - pdd
            bbb1 = np.einsum('ij,ij->i', aaa1, aaa1)
            ccc1 = np.where(bbb1==0, 1, bbb1)
            ddd1 = np.clip(np.einsum('j,ij->i', pym - pdd, aaa1) / ccc1, 0.0, 1.0)
            eee1 = pdd + ddd1[:, None] * aaa1
            zuichad = np.max(np.linalg.norm(pym - eee1, axis=1))
            mind = min(mind, zuichad)
            if zuichad <= ymr: zcsc1 += 0.06
            if zuichad < 300.0: zcsc2 += (1.0 - zuichad / 300.0) * 0.06        
        fitness[i] = -(zcsc1 + 0.02 * zcsc2) + 1e-4 * mind    
    gg = fitness < pbestf
    pbestx[gg] = pos[gg]
    pbestf[gg] = fitness[gg]    
    ggidx = np.argmin(pbestf)
    if pbestf[ggidx] < gbestf:
        gbestf = pbestf[ggidx]
        gbestx = pbestx[ggidx].copy()
    r1, r2 = rng.random((N, 4)), rng.random((N, 4))
    vel = W * vel + C1 * r1 * (pbestx - pos) + C2 * r2 * (gbestx - pos)
    pos += vel
    pos[:, 0] = (pos[:, 0] + np.pi) % (2 * np.pi) - np.pi
    pos[:, 1:] = np.clip(pos[:, 1:], bounds[1:, 0], bounds[1:, 1])
    if (it + 1) % 5 == 0 or it == 0 or it == maxit - 1:
        dqzysc = suan_sc(gbestx, DT, 5, 24, 10)
        lishi.append((dqzysc, gbestx.copy()))
        print(f"迭代 {it+1:3d}/{maxit}：当前最佳严格遮蔽时长 = {dqzysc:.3f} s")

# 开始进行局部搜索
zuiyoux = gbestx.copy()
zuiyouf = suan_sc(zuiyoux, DT, 5, 24, 10)
bujin = np.array([0.25, 6.0, 0.4, 0.25])
for j in range(5):
    gaibian = False
    for k in range(4):
        for fuhao in (+1.0, -1.0):
            xshi = zuiyoux.copy()
            xshi[k] += fuhao * bujin[k]
            xshi[0] = (xshi[0] + np.pi) % (2*np.pi) - np.pi
            xshi[1:] = np.clip(xshi[1:], bounds[1:,0], bounds[1:,1])
            fshi = suan_sc(xshi, DT, 5, 24, 10)
            if fshi > zuiyouf:
                zuiyoux, zuiyouf, gaibian = xshi, fshi, True
    bujin *= 0.5
    if not gaibian and np.all(bujin < np.array([0.02, 1.0, 0.05, 0.05])): break
for k, chuangkou, bjb in [(2, 0.6, 0.06), (3, 0.4, 0.04)]:
    zx = zuiyoux[k]
    for cval in np.arange(zx - chuangkou, zx + chuangkou + 1e-9, bjb):
        xshi = zuiyoux.copy(); xshi[k] = cval
        xshi[0] = (xshi[0] + np.pi) % (2*np.pi) - np.pi
        xshi[1:] = np.clip(xshi[1:], bounds[1:, 0], bounds[1:, 1])
        fshi = suan_sc(xshi, DT, 5, 24, 10)
        if fshi > zuiyouf:
            zuiyouf, zuiyoux = fshi, xshi

# 输出最终的结果
zuiyouth, zuiyouv, zuiyoutd, zuiyoutf = zuiyoux
zuiyout_bao = zuiyoutd + zuiyoutf
zuiyou_vwrj = np.array([zuiyouv * np.cos(zuiyouth), zuiyouv * np.sin(zuiyouth), 0.0])
zuiyou_ptf = wrjp0 + zuiyou_vwrj * zuiyoutd
zuiyou_pbao = zuiyou_ptf+zuiyou_vwrj*zuiyoutf+np.array([0,0,-0.5*G*(zuiyoutf**2)])
zuizhongsc = suan_sc(zuiyoux, DT, 5, 24, 10)
print("\n最优结果：")
print(f"  航向角θ(弧度)={zuiyouth:.6f}（度={np.degrees(zuiyouth):.3f}）")
print(f"  速度(m/s)={zuiyouv:.3f}  投放时刻(s)={zuiyoutd:.3f}  引信延时(s)={zuiyoutf:.3f}")
print(f"  起爆时刻(s)={zuiyout_bao:.3f}  起爆位置(m)={zuiyou_pbao}")
print(f"  投放位置(m)={zuiyou_ptf}")
print(f"  严格遮蔽总时长(s)={zuizhongsc:.4f}")

# 可视化结果显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sc_lishi = [h[0] for h in lishi]
plt.figure(figsize=(8, 3))
plt.title('PSO迭代历史')
plt.step(range(len(sc_lishi)), sc_lishi, where='post')
plt.scatter(len(sc_lishi)-1, sc_lishi[-1], c='orange', zorder=5, label=f'最终 {sc_lishi[-1]:.2f}s')
plt.xlabel('迭代步'); plt.ylabel('遮蔽时长 (s)'); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("最优方案三维示意")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
t_show = np.linspace(0, zuiyout_bao + ymsc, 300)
tc = np.linspace(0, 2*np.pi, 60); zc = np.linspace(mbb[2], mbb[2]+mbh, 2)
aaa, bbb = np.meshgrid(tc, zc)
ax.plot_surface(mbr*np.cos(aaa)+mbb[0], mbr*np.sin(aaa)+mbb[1], bbb, alpha=0.2, color='purple')
ax.plot(*(wrjp0 + zuiyou_vwrj * t_show[:, None]).T, 'b-', label='FY1')
ax.plot(*(ddp0 + ddvvec * t_show[:, None]).T, 'r--', label='导弹M1')
ym_guiji = [zuiyou_pbao+np.array([0,0,-ymvcj*(t-zuiyout_bao)]) for t in t_show if t>=zuiyout_bao]
if ym_guiji: ax.plot(*np.array(ym_guiji).T, 'g-', lw=2.5, label='烟幕中心')
ax.scatter(*zuiyou_ptf, c='b', s=80, label=f'投放 t={zuiyoutd:.2f}s')
ax.scatter(*zuiyou_pbao, c='g', s=90, label=f'起爆 t={zuiyout_bao:.2f}s')
ax.scatter(*mbzx, c='purple', s=65, label='目标中心')
uu,vv=np.mgrid[0:2*np.pi:36j, 0:np.pi:18j]
for i, tl in enumerate(np.linspace(zuiyout_bao, zuiyout_bao + ymsc, 7)):
    pymceng = zuiyou_pbao+np.array([0,0,-ymvcj*(tl-zuiyout_bao)])
    ax.plot_wireframe(pymceng[0]+ymr*np.cos(uu)*np.sin(vv), pymceng[1]+ymr*np.sin(uu)*np.sin(vv), pymceng[2]+ymr*np.cos(vv), color='gray', alpha=0.3 if i==0 else 0.15)
ax.legend(loc='best'); ax.view_init(elev=18, azim=-125)
span=8000; ax.set_xlim(zuiyou_pbao[0]-span,zuiyou_pbao[0]+span); ax.set_ylim(zuiyou_pbao[1]-span,zuiyou_pbao[1]+span); ax.set_zlim(max(0,zuiyou_pbao[2]-2500),zuiyou_pbao[2]+2500)
plt.tight_layout(); plt.show()
t_show2 = np.arange(max(0.0, zuiyout_bao), zuiyout_bao + ymsc + 1e-9, DT)
shifouzhebi = []
mbpts_show = np.array([[mbb[0]+r*np.cos(t_), mbb[1]+r*np.sin(t_), z] for z in np.linspace(mbb[2], mbb[2] + mbh, 10) for r in np.linspace(0, mbr, 5) for t_ in np.linspace(0, 2*np.pi, 24, endpoint=False)])
for t in t_show2:
    pdd = ddp0 + ddvvec * t
    pym = zuiyou_pbao + np.array([0., 0., -ymvcj * (t - zuiyout_bao)])
    aaa1 = mbpts_show - pdd
    bbb1 = np.einsum('ij,ij->i', aaa1, aaa1)
    ccc1 = np.where(bbb1 == 0.0, 1.0, bbb1)
    ddd1 = np.clip(np.einsum('j,ij->i', pym - pdd, aaa1) / ccc1, 0.0, 1.0)
    eee1 = np.linalg.norm(pym - (pdd + ddd1[:, None] * aaa1), axis=1)
    shifouzhebi.append(np.max(eee1) <= ymr)
shifouzhebi = np.array(shifouzhebi)
plt.figure(figsize=(10, 3.5))
plt.title(f"遮蔽时间线 (总计 {zuizhongsc:.2f}s)")
plt.fill_between(t_show2, shifouzhebi.astype(float), step='post', alpha=0.3)
plt.plot(t_show2, shifouzhebi.astype(float), drawstyle='steps-post')
plt.ylim(-0.1, 1.1); plt.yticks([0, 1], ['未遮蔽', '遮蔽'])
plt.xlabel('时间 (s)'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()