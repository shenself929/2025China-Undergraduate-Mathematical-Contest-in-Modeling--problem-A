

function rps()
    %% 初始化
    gp = igp();
    lb = [0, 70, 0, 0, 1, 0, 1, 0];
    ub = [2*pi, 140, 30, 20, 12, 20, 12, 20];

    opts = optimoptions('particleswarm','SwarmSize',100,'MaxIterations',200,'UseParallel',true,'Display','iter','FunctionTolerance',1e-8);
    of = @(x) of8(x, gp);

    fprintf('开始优化...\n');
    [x, fv] = particleswarm(of, 8, lb, ub, opts);

    fprintf('\n===== 结果 =====\n');
    fprintf('最佳遮蔽: %.3f s\n', -fv);
    fprintf('航向角(t)= %.2f° | 速度(v)= %.2f m/s\n', rad2deg(x(1)), x(2));
    fprintf('弹1: t=%.2f, d=%.2f\n', x(3), x(4));
    fprintf('弹2: t=%.2f, d=%.2f\n', x(3)+x(5), x(6));
    fprintf('弹3: t=%.2f, d=%.2f\n', x(3)+x(5)+x(7), x(8));

    %% 分析
    [~, tv, m, cm] = of8f(x, gp);
    fprintf('\n===== 坐标与时间段 =====\n');
    
    uv = [x(2)*cos(x(1)), x(2)*sin(x(1)), 0];
    td = [x(3), x(3)+x(5), x(3)+x(5)+x(7)];
    d = [x(4), x(6), x(8)];
    cte = zeros(1,3);

    for i = 1:3
        pd = gp.up + uv * td(i);
        pe = pd + uv * d(i) - [0,0,0.5*gp.g*d(i)^2];
        fprintf('弹 %d: 投放(%.1f,%.1f,%.1f), 起爆(%.1f,%.1f,%.1f)\n', i, pd, pe);
        
        idx = find(cm(:, i));
        tp = tv(idx);
        
        fprintf('  遮蔽时段:\n');
        if isempty(tp)
            fprintf('    无\n');
            cte(i) = 0;
        else
            cte(i) = numel(tp) * gp.dt;
            if numel(tp) == 1
                fprintf('    %.3f - %.3f s\n', tp(1), tp(1));
            else
                ji = find(diff(tp) > gp.dt * 1.5);
                si = [1, ji + 1];
                ei = [ji, numel(tp)];
                for k = 1:length(si)
                    fprintf('    %.3f - %.3f s\n', tp(si(k)), tp(ei(k)));
                end
            end
        end
    end

    fprintf('\n===== 遮蔽统计 =====\n');
    for i = 1:3
        fprintf('弹 %d 贡献时长: %.3f s\n', i, cte(i));
    end
    fprintf('合计(不重叠): %.3f s\n', sum(cte));
    fprintf('总计(重叠): %.3f s\n', sum(m)*gp.dt);
end

function f = of8(x, gp), [f, ~, ~, ~] = sof(x, gp); end
function [f, tv, m, cm] = of8f(x, gp), [f, tv, m, cm] = sof(x, gp); end

function [f, tv, m, cm] = sof(x, gp)
    t=x(1); v=x(2); t1=x(3); d1=x(4); dt2=x(5); d2=x(6); dt3=x(7); d3=x(8);
    t2 = t1+dt2; t3 = t2+dt3;
    uv = [v*cos(t), v*sin(t), 0];
    p1 = gp.up+uv*t1; p2=gp.up+uv*t2; p3=gp.up+uv*t3;
    h1=p1(3)-0.5*gp.g*d1^2; h2=p2(3)-0.5*gp.g*d2^2; h3=p3(3)-0.5*gp.g*d3^2;
    if any([h1,h2,h3]<0.5)
        f=1e5+sum(max(0,0.5-[h1,h2,h3]).^2)*1e6;
        tv=[]; m=[]; cm=[]; return;
    end
    tExp=[t1+d1,t2+d2,t3+d3];
    pExp=[p1+uv*d1-[0,0,0.5*gp.g*d1^2]; p2+uv*d2-[0,0,0.5*gp.g*d2^2]; p3+uv*d3-[0,0,0.5*gp.g*d3^2]];
    smoke=struct('tExp',num2cell(tExp(:)),'pExp',num2cell(pExp, 2));
    [shield, tv, m, cm] = csd(smoke, gp);
    f = -shield;
end

function [tot, tv, m, cm] = csd(smoke, gp)
    maxT = norm(gp.mp - gp.ft) / gp.mv;
    tv = 0:gp.dt:maxT;
    m = false(size(tv));
    cm = zeros(length(tv), length(smoke));
    for i = 1:length(tv)
        tc = tv(i);
        mp = gp.mp + gp.mv * tc * gp.md;
        isCov = false;
        for s = 1:length(smoke)
            b = smoke(s);
            if tc >= b.tExp && tc <= b.tExp + gp.svt
                cp = b.pExp - [0,0,gp.ssv * (tc - b.tExp)];
                isShield = true;
                for sp = 1:size(gp.ts, 1)
                    if ~isis(mp, gp.ts(sp,:), cp, gp.sr, gp.eps)
                        isShield = false; break;
                    end
                end
                if isShield, isCov = true; cm(i, s) = 1; end
            end
        end
        m(i) = isCov;
    end
    tot = sum(m) * gp.dt;
end

function flag = isis(M, P, C, r, eps)
    MP = P - M; MC = C - M;
    a = dot(MP, MP);
    if a < eps, flag = norm(MC) <= r + eps; return; end
    b = -2 * dot(MP, MC);
    c = dot(MC, MC) - r^2;
    dsc = b^2 - 4*a*c;
    if dsc < -eps, flag = false; return; end
    if dsc < 0, dsc = 0; end
    s1 = (-b - sqrt(dsc)) / (2*a); s2 = (-b + sqrt(dsc)) / (2*a);
    flag = (s1 <= 1 + eps) && (s2 >= -eps);
end

function gp = igp()
    gp=struct(); gp.g=9.8; gp.eps=1e-10;
    gp.rtc=[0,200,0]; gp.rtr=7; gp.rth=10;
    gp.ts=gts(gp.rtc,gp.rtr,gp.rth,35,12);
    gp.ft=[0,0,0]; gp.mp=[20000,0,2000]; gp.mv=300;
    gp.md=(gp.ft-gp.mp)/norm(gp.ft-gp.mp);
    gp.sr=10; gp.ssv=3; gp.svt=20;
    gp.up=[17800,0,1800]; gp.uvb=[70,140]; gp.utb=[0, 2*pi]; gp.dt=0.01;
end

function s = gts(c, r, h, nC, nH)
    s = [];
    zMin=c(3); zMax=c(3)+h;
    thV=linspace(0,2*pi,nC); hV=linspace(zMin,zMax,nH);
    for z=hV, for th=thV, s=[s;c(1)+r*cos(th),c(2)+r*sin(th),z]; end, end
    irV=linspace(0,r,3);
    for ri=irV(2:end), for z=hV(2:end-1), for k=1:3:length(thV)
        th = thV(k); s=[s; c(1)+ri*cos(th),c(2)+ri*sin(th),z];
    end, end, end
    s = unique(s, 'rows');
end