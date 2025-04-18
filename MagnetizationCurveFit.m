clear;
mu0 = 4*pi*1e-7;

%% Load BH hysteresis curves
load('./data/init_curves/1s.mat','BH_init'); BH_1 = BH_init;
load('./data/init_curves/2s.mat','BH_init'); BH_2 = BH_init;
load('./data/init_curves/3s.mat','BH_init'); BH_3 = BH_init;
load('./data/init_curves/4s.mat','BH_init'); BH_4 = BH_init;
load('./data/init_curves/5s.mat','BH_init'); BH_5 = BH_init;
load('./data/init_curves/6s.mat','BH_init'); BH_6 = BH_init;
load('./data/init_curves/7s.mat','BH_init'); BH_7 = BH_init;
load('./data/init_curves/8s.mat','BH_init'); BH_8 = BH_init;

%% Vector of all initial curves
% Skip initial curvature
start = 20;
H = [BH_1(start:end,1);BH_2(start:end,1);BH_3(start:end,1);BH_4(start:end,1);BH_5(start:end,1);BH_6(start:end,1);BH_7(start:end,1);BH_8(start:end,1)];
B = [BH_1(start:end,2);BH_2(start:end,2);BH_3(start:end,2);BH_4(start:end,2);BH_5(start:end,2);BH_6(start:end,2);BH_7(start:end,2);BH_8(start:end,2)];
J = B - mu0*H;

% All points
H_all = [BH_1(:,1);BH_2(:,1);BH_3(:,1);BH_4(:,1);BH_5(:,1);BH_6(:,1);BH_7(:,1);BH_8(:,1)];
B_all = [BH_1(:,2);BH_2(:,2);BH_3(:,2);BH_4(:,2);BH_5(:,2);BH_6(:,2);BH_7(:,2);BH_8(:,2)];

%% Fit the magnetization curve model
% Optimized function
f_single = @(w)langevin(w,H,J);
f_double = @(w)double_langevin(w,H,J);

% Number of initial points
N_single = 1000;
N_double = 8;

% Initial values 
init_w1_single = 1.7793; % (w1 = J_max)
init_w2_single = linspace(1e-3,1e3,N_single);

init_w1_double = linspace(1e-3,1.7793,N_double);
init_w2w4 = linspace(1e-3,1e3,N_double);
init_w3_double = 1 - init_w1_double;

[INIT_w1, INIT_w2, INIT_w3, INIT_w4] = ndgrid(init_w1_double,init_w2w4,init_w3_double,init_w2w4);
INIT_w1 = INIT_w1(:); INIT_w2 = INIT_w2(:); INIT_w3 = INIT_w3(:); INIT_w4 = INIT_w4(:);

% Optimize single
options = optimoptions('fminunc','display','off');
err_best = inf;
for i = 1:N_single
    [w,err] = fminunc(f_single,[init_w1_single,init_w2_single(i)],options);
    if err < err_best
        err_best = err;
        fprintf('Iteration: %d, err: %d\n',i,err_best);
        w_single_opt = w;
    end
end

% Optimize double
err_best = inf;
for i = 1:length(INIT_w1)
    [w,err] = fminunc(f_double,[INIT_w1(i),INIT_w2(i),INIT_w3(i),INIT_w4(i)],options);
    if err < err_best
        err_best = err;
        fprintf('Iteration: %d, err: %d\n',i,err_best);
        w_double_opt = w;
    end
end


%% Sample the model
H_sample = linspace(1e-5,5e5,100000);
B_single_sample = w_single_opt(1)*(coth(H_sample./w_single_opt(2)) - w_single_opt(2)./H_sample) + mu0*H_sample;
B_double_sample = w_double_opt(1)*(coth(H_sample./w_double_opt(2)) - w_double_opt(2)./H_sample) + w_double_opt(3)*(coth(H_sample./w_double_opt(4)) - w_double_opt(4)./H_sample) + mu0*H_sample;

%% Coefficient of determination
B_single = w_single_opt(1)*(coth(H_all./w_single_opt(2)) - w_single_opt(2)./H_all) + mu0*H_all;
B_double = w_double_opt(1)*(coth(H_all./w_double_opt(2)) - w_double_opt(2)./H_all) + w_double_opt(3)*(coth(H_all./w_double_opt(4)) - w_double_opt(4)./H_all) + mu0*H_all;

R_single = 1 - sum((B_single - B_all).^2) / sum((B_all - mean(B_all)).^2);
R_double = 1 - sum((B_double - B_all).^2) / sum((B_all - mean(B_all)).^2);

%% Plot curve fit

figure; hold on; grid on;
c = [0.7,0.7,0.7];

plot(BH_1(:,1)/1e3,BH_1(:,2),'color',c);
plot(BH_2(:,1)/1e3,BH_2(:,2),'color',c);
plot(BH_3(:,1)/1e3,BH_3(:,2),'color',c);
plot(BH_4(:,1)/1e3,BH_4(:,2),'color',c);
plot(BH_5(:,1)/1e3,BH_5(:,2),'color',c);
plot(BH_6(:,1)/1e3,BH_6(:,2),'color',c);
plot(BH_7(:,1)/1e3,BH_7(:,2),'color',c);
plot(BH_8(:,1)/1e3,BH_8(:,2),'color',c);
s = plot(H_sample/1e3,B_single_sample,'k-','linewidth',1);
d = plot(H_sample/1e3,B_double_sample,'k--','linewidth',1);

xlabel('H (A/m)');
ylabel('B (T)');


%% Optimized function
function err = langevin(w,x,y_true)
    y = w(1)*(coth(x./w(2)) - w(2)./x);
    err = sum((y-y_true).^2);
end

function err = double_langevin(w,x,y_true)
    y = w(1)*(coth(x./w(2)) - w(2)./x) + w(3)*(coth(x./w(4)) - w(4)./x);
    err = sum((y-y_true).^2);
end
