clear;
mu0 = 4*pi*1e-7;

%% Load hysteresis curve data
% Hysteresis curve without initial magnetization
load('./data/test_curve_smoothed.mat','hyst');
% Assuming H starts at -H_max:
H = -hyst(:,1);
J = -hyst(:,2);

%% Anhysteretic magnetization
N = int64(length(H)/2);
J_lo = J(1:N); J_up = flip(J(N+1:end));
H_lo = H(1:N); H_up = flip(H(N+1:end));

J_am = mean([J_lo,J_up],2);
H_am = mean([H_lo,H_up],2);

J_am(H_am == 0) = [];
H_am(H_am == 0) = [];

% figure; hold on;
% plot(H_up,J_up);
% plot(H_lo,J_lo);
% plot(H_am,J_am);
% legend('Upper','Lower','Mean');

%% New H-field grid
% Measured H-field with added initial curve
N_init = 1000;
H_new = [linspace(0,H(1,1),N_init)';H];

%% Test J-A model
% % Parameters
% alpha = 2e-3;
% a = 1000;
% k = 80;
% c = 0.22;
% Ms = 1.4e6;
% 
% J_model_exp = mu0*JA_explicit(H_new,alpha,a,k,c,Ms);
% J_model_rk = mu0*JA_rk(H_new,alpha,a,k,c,Ms);
% 
% figure; hold on;
% plot(H,J);
% plot(H_new,J_model_exp);
% plot(H_new,J_model_rk);
% xlabel('H (A/m)'); ylabel('J (T)');
% grid on;

%% FIT the Langevin function
% Initial parameters
alpha_0 = 2e-3;
a_0 = 1000;
Ms_0 = 1.4e6; 

% Optimize Langevin function: param = [alpha, a, Ms]
ls_fun = @(param)langevin_fit(H_am,J_am/mu0,param);
options = optimset('MaxFunEvals',10000,'Display','iter');
opt_am = fminsearch(ls_fun,[alpha_0,a_0,Ms_0],options);

% Optimized parameters
alpha = opt_am(1);
a = opt_am(2);
Ms = opt_am(3);

%% Sample and plot fitted model
% J_model_am = mu0*calc_Man(J_am/mu0,H_am,alpha,a,Ms);
% 
% figure; hold on;
% plot(H_am,J_am);
% plot(H_am,J_model_am);
% grid on;

%% FIT J-A model
% Initial parameters
k_0 = 100;
c_0 = 0.5;

% Optimize Langevin function: param = [k, c]
ls_fun = @(param)ja_fit(H_up,J_up,alpha,a,Ms,param);
options = optimset('MaxIter',1e4,'MaxFunEvals',1e5,'Display','iter','TolFun',1e-7);
opt_ja = fminsearch(ls_fun,[k_0,c_0],options);
% opt_ja = fminunc(ls_fun,[k_0,c_0],options);

% Optimized parameters
k = opt_ja(1);
c = opt_ja(2);

%% Sample and plot fitted model
J_init = mu0*JA_rk(H_new,alpha,a,Ms,k_0,c_0);
J_model = mu0*JA_rk(H_new,alpha,a,Ms,k,c);

figure; hold on;
plot(H,J);
plot(H_new,J_init);
plot(H_new,J_model);
legend('Data', 'Initial guess', 'Optimized')
grid on;

%% Jiles-Atherton solution (Explicit Euler)
function M = JA_explicit(H,alpha,a,Ms,k,c)
    N = length(H);
    M = zeros(N,1);
    Man = zeros(N,1);
    for i = 1:N-1
        dH = H(i+1)-H(i);
        d = sign(dH);

        Man(i+1) = calc_Man(M(i),H(i+1),alpha,a,Ms);
        M(i+1) = M(i) + dH*(1/(1+c))*(Man(i)-M(i))/(d*k-alpha*(Man(i)-M(i))) + (c/(1+c))*(Man(i+1)-Man(i));
    end    
end

%% Jiles-Atherton solution (Runge-Kutta)
function M = JA_rk(H,alpha,a,Ms,k,c)
    N = length(H);
    M = zeros(N,1);
    for i = 1:N-1
        dH = H(i+1) - H(i);
        d = sign(dH);
       
        k1 = calc_dMdH(M(i),H(i),d,alpha,a,k,c,Ms);
        k2 = calc_dMdH(M(i)+dH*(1/4)*k1,H(i)+(1/4)*dH,d,alpha,a,k,c,Ms);
        k3 = calc_dMdH(M(i)+dH*((3/32)*k1 + 9/32*k2), H(i) + (3/8)*dH, d, alpha, a, k, c, Ms);
        k4 = calc_dMdH(M(i)+dH*((1932/2197)*k1-(7200/2197)*k2+(7296/2197)*k3),H(i)+(12/13)*dH,d,alpha,a,k,c,Ms);
        k5 = calc_dMdH(M(i)+dH*((439/216)*k1-8*k2+(3680/513)*k3-(845/4104)*k4),H(i)+dH,d,alpha,a,k,c,Ms);
        k6 = calc_dMdH(M(i)+dH*(-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5),H(i)+0.5*dH,d,alpha,a,k,c,Ms);

        M(i+1) = M(i) + dH*((16/135)*k1+(6656/12825)*k3+(28561/56430)*k4-(9/50)*k5+(2/55)*k6);
    end
end

function dMdH = calc_dMdH(M,H,d,alpha,a,k,c,Ms)
    Man = calc_Man(M,H,alpha,a,Ms);
    dMdH = (1/(1+c))*(Man-M)/(d*k - alpha*(Man-M));
end

%% Anhysteretic magnetization (Langevin)
function Man = calc_Man(M,H,alpha,a,Ms)
    Heff = H + alpha*M;
    if abs(Heff) < 1e-6
        Man = Ms*sign(Heff);
    else
        Man = Ms*(coth(Heff/a) - a./Heff);
    end
end

%% Least-squares objective function (Langevin)
function err = langevin_fit(H_true,M_true,param)
    % Parameters
    alpha = param(1); a = param(2); Ms = param(3);
    % Predict
    M_model = calc_Man(M_true,H_true,alpha,a,Ms);
    % Compute LS error
    err = (1/length(M_true))*sum((M_model-M_true).^2);
end

%% Least-squares objective function (J-A)
function [err,J_model] = ja_fit(H_true,J_true,alpha,a,Ms,param)
    % Parameters
    k = param(1); c = param(2);
    mu0 = 4*pi*1e-7;
    % Add initial curve points to the H-field
    % (H must start at H_max)
    N_init = 500;
    H = [linspace(0,H_true(1,1),N_init)';H_true];    
    % Predict (Runge-Kutta)
    M_model = JA_rk(H,alpha,a,Ms,k,c);
    J_model = mu0*M_model(N_init+1:end);
    % Compute LS error
    err = (1/length(J_true))*sum((J_model-J_true).^2);
    % err = max(abs(J_model-J_true));

    % % Plot
    % close all;
    % figure; hold on;
    % plot(H_true,J_true);
    % plot(H,mu0*M_model);
    % % plot(H_true,J_model);
    % plot(H_true,(J_model-J_true).^2);
    % % fprintf('Error: %d\n',err);
end

