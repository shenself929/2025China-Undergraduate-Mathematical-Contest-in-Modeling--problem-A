%% 0. 初始化环境
clear; 
clc; 
close all;

fprintf('======================================================\n');
fprintf('      开始运行问题一：烟幕干扰仿真与可视化\n');
fprintf('======================================================\n\n');

%% 1. 常量与初始参数定义
g = 9.8;               
epsilon = 1e-9;        

fkTgt = [0.0, 0.0, 0.0]; 
rt.c = [0.0, 200.0, 0.0];
rt.r = 7.0;   
rt.h = 10.0;  

uav.p = [17800.0, 0.0, 1800.0]; 
uav.v = 120.0;     
uav.tDrop = 1.5; 
uav.tDet = 3.6;  

sm.r = 10.0;         
sm.sinkV = 3.0; 
sm.validT = 20.0;

msl.p = [20000.0, 0.0, 2000.0];
msl.v = 300.0;   

dt = 0.01; 

%% 2. 核心位置计算
fprintf('步骤1: 正在计算关键点位置...\n');
uxy = uav.p(1:2);
txy = fkTgt(1:2);
dxy = (txy - uxy) / norm(txy - uxy);
dp = [uxy + dxy * uav.v * uav.tDrop, uav.p(3)];

dxyd = (fkTgt(1:2) - dp(1:2)) / norm(fkTgt(1:2) - dp(1:2));
ep = [dp(1:2) + dxyd * uav.v * uav.tDet, dp(3) - 0.5 * g * uav.tDet^2];

fprintf(' - 无人机初始位置: [%.2f, %.2f, %.2f]\n', uav.p);
fprintf(' - 烟幕弹投放点:   [%.2f, %.2f, %.2f]\n', dp);
fprintf(' - 烟幕弹起爆点:   [%.2f, %.2f, %.2f]\n\n', ep);

%% 3. 目标采样与导弹方向
fprintf('步骤2: 正在生成目标采样点和计算导弹轨迹...\n');
ts = genSmp(rt);
mdir = (fkTgt - msl.p) / norm(fkTgt - msl.p);
fprintf(' - 真目标高密度采样点总数: %d\n\n', size(ts, 1));

%% 4. 迭代计算有效时长
fprintf('步骤3: 正在进行高精度迭代仿真...\n');
tDet = uav.tDrop + uav.tDet;
tS = tDet;
tE = tDet + sm.validT;
tv = tS:dt:tE;

vt = 0;
segs = [];
wv = false;
sst = -1;

dd = zeros(length(tv), 1);
rmp = rt.c + [0, 0, rt.h/2];

for i = 1:length(tv)
    t = tv(i);
    mPos = msl.p + mdir * msl.v * t;
    smPos = ep - [0, 0, sm.sinkV * (t - tDet)];
    
    dd(i) = dps(smPos, mPos, rmp);
    
    iv = isTis(mPos, smPos, sm.r, ts, epsilon);
    if iv
        vt = vt + dt; 
    end
    if iv && ~wv
        sst = t; 
    end
    if ~iv && wv && sst > 0
        segs = [segs; struct('s', sst, 'e', t - dt)]; 
    end
    wv = iv;
end
if wv && sst > 0
    segs = [segs; struct('s', sst, 'e', tE)]; 
end
fprintf(' - 仿真完成。\n\n');

%% 5. 打印最终结果
fprintf('==================== 最终计算结果 ====================\n');
fprintf('【总时长】真目标被有效遮蔽的总时长: %.4f 秒\n', vt);
if ~isempty(segs)
    fprintf('【时间段】有效遮蔽时间段: %.4fs ~ %.4fs\n', segs(1).s, segs(1).e);
else
    fprintf('【时间段】无有效遮蔽时间段。\n');
end
fprintf('======================================================\n\n');

%% 6. 可视化
fprintf('步骤4: 正在生成可视化图形...\n\n');

fprintf(' - 正在生成图 1: 3D 物理示意图...\n');
figure('Name', '图 1: 3D 物理示意图', 'Position', [50, 50, 1200, 800]);
hold on; grid on; axis equal; view(45, 25);

plot3(fkTgt(1), fkTgt(2), fkTgt(3), 'kx', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', '假目标');
drawCyl(rt.c, rt.r, rt.h, 'b', 0.3);
plot3(uav.p(1), uav.p(2), uav.p(3), 's', 'Color', [0.1 0.8 0.1], 'MarkerFaceColor', [0.1 0.8 0.1], 'MarkerSize', 10, 'DisplayName', 'FY1 初始');
plot3([uav.p(1), dp(1)], [uav.p(2), dp(2)], [uav.p(3), dp(3)], 'g--', 'LineWidth', 1.5, 'DisplayName', 'FY1 轨迹');
plot3(dp(1), dp(2), dp(3), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8, 'DisplayName', '投放点');
plot3(ep(1), ep(2), ep(3), 'm*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', '起爆点');
plot3([msl.p(1), fkTgt(1)], [msl.p(2), fkTgt(2)], [msl.p(3), fkTgt(3)], 'r-', 'LineWidth', 2, 'DisplayName', 'M1 轨迹');

if ~isempty(segs)
    tp = [segs(1).s, mean([segs(1).s, segs(1).e]), segs(1).e];
    cs = {[0, 0.8, 0.8], [0.9, 0.5, 0], [0.8, 0, 0.8]};
    for i = 1:length(tp)
        t = tp(i);
        mPos = msl.p + mdir * msl.v * t;
        smPos = ep - [0, 0, sm.sinkV * (t - tDet)];
        [sx,sy,sz] = sphere(20);
        surf(sx*sm.r+smPos(1), sy*sm.r+smPos(2), sz*sm.r+smPos(3), 'FaceColor', cs{i}, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', sprintf('烟幕 @ t=%.2fs', t));
        plot3(mPos(1), mPos(2), mPos(3), 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', cs{i}, 'MarkerSize', 12, 'DisplayName', sprintf('导弹 @ t=%.2fs', t));
    end
end

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)'); 
title('图 1: 3D 物理示意图');
legend('show', 'Location', 'bestoutside');

fprintf(' - 正在生成图 2: 遮蔽距离随时间变化图...\n');
figure('Name', '图 2: 遮蔽距离随时间变化', 'Position', [100, 100, 1400, 800]);

subplot(2,1,1);
hold on; grid on;
plot(tv, dd, 'b-', 'LineWidth', 2, 'DisplayName', '烟幕中心到视线距离');
plot(tv, ones(size(tv)) * sm.r, 'r--', 'LineWidth', 2, 'DisplayName', '烟幕半径 (10m)');

if ~isempty(segs)
    for i = 1:length(segs)
        seg = segs(i);
        is = find(tv >= seg.s, 1, 'first');
        ie = find(tv <= seg.e, 1, 'last');
        if ~isempty(is) && ~isempty(ie)
            fill([tv(is:ie), fliplr(tv(is:ie))], [dd(is:ie)', zeros(1, length(is:ie))], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '有效遮蔽区间');
        end
    end
end

xlabel('时间 (s)', 'FontSize', 12);
ylabel('距离 (m)', 'FontSize', 12);
title('烟幕中心到导弹-目标视线的最短距离', 'FontSize', 14, 'FontWeight', 'bold');
legend('show', 'Location', 'best');
ylim([0, max(dd) * 1.2]);

subplot(2,1,2);
hold on; grid on;

if ~isempty(segs)
    tfs = max(tS, segs(1).s - 2);
    tfe = min(tE, segs(1).e + 2);
    
    idf = (tv >= tfs) & (tv <= tfe);
    tf = tv(idf);
    df = dd(idf);
    
    plot(tf, df, 'b-', 'LineWidth', 2, 'DisplayName', '烟幕中心到视线距离');
    plot(tf, ones(size(tf)) * sm.r, 'r--', 'LineWidth', 2, 'DisplayName', '烟幕半径 (10m)');
    
    seg = segs(1);
    ids = (tf >= seg.s) & (tf <= seg.e);
    if sum(ids) > 0
        fill([tf(ids), fliplr(tf(ids))], [df(ids)', zeros(1, sum(ids))], 'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', '有效遮蔽区间');
    end
    
    plot([seg.s, seg.s], [0, max(df)], 'g:', 'LineWidth', 2);
    plot([seg.e, seg.e], [0, max(df)], 'g:', 'LineWidth', 2);
    text(seg.s, max(df)*0.9, sprintf('开始: %.2fs', seg.s), 'FontSize', 10, 'Color', 'g');
    text(seg.e, max(df)*0.9, sprintf('结束: %.2fs', seg.e), 'FontSize', 10, 'Color', 'g');
    
    xlim([tfs, tfe]);
else
    plot(tv, dd, 'b-', 'LineWidth', 2);
    plot(tv, ones(size(tv)) * sm.r, 'r--', 'LineWidth', 2);
end

xlabel('时间 (s)', 'FontSize', 12);
ylabel('距离 (m)', 'FontSize', 12);
title('局部放大图 - 有效遮蔽时间段', 'FontSize', 14, 'FontWeight', 'bold');
legend('show', 'Location', 'best');

fprintf('\n所有计算和可视化任务已完成！\n');

%% ========================= 辅助函数定义区域 =========================
function s = genSmp(rt)
    nc = 30;
    nh = 10;
    s = [];
    c = rt.c;
    r = rt.r;
    h = rt.h;
    mz = c(3);
    Mz = c(3) + h;
    
    theta = linspace(0, 2*pi, nc);
    hs = linspace(mz, Mz, nh);
    
    for z = hs
        for th = theta
            x = c(1) + r * cos(th);
            y = c(2) + r * sin(th);
            s = [s; x, y, z];
        end
    end
    
    s = [s; c; c + [0, 0, h/2]; c + [0, 0, h]];
    s = unique(s, 'rows');
end

function sh = isTis(mPos, smPos, smR, ts, epsilon)
    sh = true;
    for i = 1:size(ts, 1)
        if ~isSis(mPos, ts(i, :), smPos, smR, epsilon)
            sh = false; 
            return;
        end
    end
end

function isect = isSis(M, P, C, r, epsilon)
    MP = P - M; 
    MC = C - M; 
    a = dot(MP, MP);
    if a < epsilon
        isect = norm(MC) <= r + epsilon; 
        return; 
    end
    b = -2 * dot(MP, MC); 
    c = dot(MC, MC) - r^2; 
    dsc = b^2 - 4*a*c;
    if dsc < -epsilon
        isect = false; 
        return; 
    end
    if dsc < 0
        dsc = 0;
    end
    sqrt_d = sqrt(dsc); 
    s1 = (-b - sqrt_d) / (2*a); 
    s2 = (-b + sqrt_d) / (2*a);
    isect = (s1 <= 1 + epsilon) && (s2 >= -epsilon);
end

function drawCyl(c, r, h, clr, alpha)
    [X, Y, Z] = cylinder(r, 30); 
    Z = Z * h;
    X = X + c(1);
    Y = Y + c(2); 
    Z = Z + c(3);
    surf(X, Y, Z, 'FaceColor', clr, 'EdgeColor', 'none', 'FaceAlpha', alpha, 'DisplayName', '真目标');
    fill3(X(1,:), Y(1,:), Z(1,:), clr, 'FaceAlpha', alpha, 'HandleVisibility', 'off');
    fill3(X(2,:), Y(2,:), Z(2,:), clr, 'FaceAlpha', alpha, 'HandleVisibility', 'off');
end

function d = dps(pt, v1, v2)
    v = v2 - v1; 
    w = pt - v1; 
    c1 = dot(w, v);
    if c1 <= 0
        d = norm(pt - v1); 
        return; 
    end
    c2 = dot(v, v);
    if c2 <= c1
        d = norm(pt - v2); 
        return; 
    end
    b = c1 / c2; 
    pb = v1 + b * v;
    d = norm(pt - pb);
end