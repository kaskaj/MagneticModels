clear;
mu0 = 4*pi*1e-7;

%% Load hysteresis curve
% Hysteresis curve without initial magnetization
load('./data/test_curve_smoothed.mat','hyst');
% Assuming H starts at -H_max:
H = -hyst(:,1);
J = -hyst(:,2);

%% Cut a upper part of the curve
% (Enough data for the fitting)
dJ = mean(abs(diff(J)));
i_Hc = find(J<dJ & J>-dJ);
H_cut = H(1:i_Hc(1));
J_cut = J(1:i_Hc(1));

% % Plot result
% figure; hold on;
% plot(H,J);
% plot(H_cut,J_cut);

%% Test J-A model
% % H-field
% H_new = [linspace(0,H(1,1),int64(H(1,1)/(H(1,1)-H(2,1))))';H];
% % Parameters
% Ms = 1.6e6; % (A/m)
% a = 1000; % (A/m)
% k = 700; % (A/m)
% alpha = 1.4e-3;
% c = 0.22;
% 
% J_model = mu0*JA_explicit(H_new,alpha,a,k,c,Ms);
% % J_model = mu0*JA_implicit(H_new,alpha,a,k,c,Ms);
% 
% figure; hold on;
% plot(H_new,J_model);
% grid on;

%% Fit the model
% Initial parameters
alpha = 1.7e-3;
a = 1000; % (A/m)
k = 700; % (A/m)
c = 0.22;
Ms = 1.7e6; % (A/m)
param0 = [alpha,a,k,c,Ms];

% Optimize first with explicit J-A solver
ls_fun = @(param)ls_error(H_cut,J_cut,param,"explicit");
options = optimset('MaxFunEvals',1000);
opt_param_exp = fminsearch(ls_fun,param0,options);

% Use explicit solver optimization as the initial guess
ls_fun = @(param)ls_error(H_cut,J_cut,param,"implicit");
options = optimset('MaxFunEvals',10,'Display','iter');
opt_param = fminsearch(ls_fun,opt_param_exp,options);

%% Sample fitted model
dH = H(1,1)-H(2,1);
N_init = int64(H(1,1)/dH);
H_new = [linspace(0,H(1,1),N_init)';H];
J_model = mu0*JA_explicit(H_new,opt_param(1),opt_param(2),opt_param(3),opt_param(4),opt_param(5));

%% Plot
figure; hold on;
plot(H,J);
plot(H_new,J_model);
grid on;

%% Jiles-Atherton solution (Explicit Euler)
function M = JA_explicit(H,alpha,a,k,c,Ms)
    N = length(H);
    M = zeros(N,1);
    Man = zeros(N,1);
    for i = 1:N-1
        dH = H(i+1)-H(i);
        d = sign(dH);
        
        Heff = H(i+1) + alpha*M(i);
        if abs(Heff) < 1e-6
            Man(i+1) = Ms*sign(Heff);
        else
            Man(i+1) = Ms*(coth(Heff/a) - a/Heff);
        end
        M(i+1) = M(i) + dH*(1/(1+c))*(Man(i)-M(i))/(d*k-alpha*(Man(i)-M(i))) + (c/(1+c))*(Man(i+1)-Man(i));
    end    
end

%% Jiles-Atherton solution (Implicit Euler)
function M = JA_implicit(H, alpha, a, k, c, Ms)
    N = length(H);
    M = zeros(N,1);
    Man = zeros(N,1);    
    for i = 1:N-1
        dH = H(i+1) - H(i);
        d = sign(dH);
        
        Heff = H(i+1) + alpha*M(i);
        if abs(Heff) < 1e-6
            Man(i+1) = Ms*sign(Heff);
        else
            Man(i+1) = Ms*(coth(Heff/a) - a/Heff);
        end
        % Find M(i+1)
        func = @(M_new)(M_new-M(i)) - (dH*(1/(1+c))*(Man(i+1)-M_new)/(d*k-alpha*(Man(i+1)-M_new)) + (c/(1+c))*(Man(i+1)-Man(i)));
        M(i+1) = fsolve(func, M(i), optimset('Display','off','MaxIter',100));
    end    
end

%% Least-squares objective function
function [err,J_model] = ls_error(H_true,J_true,param,solver)
    % J-A Model parameters
    alpha = param(1);
    a = param(2);
    k = param(3);
    c = param(4);
    Ms = param(5);
    mu0 = 4*pi*1e-7;
    % Add initial curve points to the H-field
    % (H must start at H_max)
    dH = H_true(1,1)-H_true(2,1);
    N_init = int64(H_true(1,1)/dH);
    H = [linspace(0,H_true(1,1),N_init)';H_true];
    % Compute polarization using J-A model
    if solver == "explicit"
        M_model = JA_explicit(H,alpha,a,k,c,Ms);
    elseif solver == "implicit"
        M_model = JA_implicit(H,alpha,a,k,c,Ms);
    end
    J_model = mu0*M_model(N_init+1:end);
    % Compute the error
    err = (1/length(J_true))*sum((J_model - J_true).^2);
    % Plot
    % figure; hold on;
    % plot(H_true,J_true);
    % plot(H_true,J_model);
    % plot(H_true,(J_model - J_true).^2);
end

