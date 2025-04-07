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
H_all = [BH_1(:,1);BH_2(:,1);BH_3(:,1);BH_4(:,1);BH_5(:,1);BH_6(:,1);BH_7(:,1);BH_8(:,1)];
B_all = [BH_1(:,2);BH_2(:,2);BH_3(:,2);BH_4(:,2);BH_5(:,2);BH_6(:,2);BH_7(:,2);BH_8(:,2)];

J_all = B_all-mu0*H_all;

%% Fit the magnetization curve model
% Optimized function
f = @(w)langevin(w,H_all,J_all);
% Starting points
N = 1000;
% Vector of optimized parameters
w = zeros(N,2);
% Vector of squared errors
errs = zeros(N,1);
% Initial values (w1 = J_max)
init_w1 = 1.7793;
init_w2 = linspace(1e-1,1e5,N);
% Optimize
options = optimoptions('fminunc','display','off');
for i = 1:N
    [w(i,:),errs(i)] = fminunc(f,[init_w1,init_w2(i)],options);
end
% Select lowest error
[~,i_w] = min(errs);
w_opt = w(i_w,:);

%% Sample the model
H_sample = linspace(1e-5,5e5,100000);
B_sample = w_opt(1)*(coth(H_sample./w_opt(2)) - w_opt(2)./H_sample) + mu0*H_sample;

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
plot(H_sample/1e3,B_sample,'k-','linewidth',1);

xlim([0,150]);
ylim([0,2]);
xlabel('H (A/m)');
ylabel('B (T)');

%% Optimized function
function err = langevin(w,x,y_true)
    y = w(1)*(coth(x./w(2)) - w(2)./x);
    err = sum((y-y_true).^2);
end

