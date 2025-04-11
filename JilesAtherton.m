clear;
mu0 = 4*pi*1e-7;

%% Load hysteresis curve
% Hysteresis curve without initial magnetization
load('./data/test_curve_smoothed.mat','hyst');
% Assuming H starts at -H_max:
H = -hyst(:,1);
J = -hyst(:,2);

%% New H-field grid
% Measured H-field with added initial curve
N_init = 1000;
H_new = [linspace(0,H(1,1),N_init)';H];

%% Test J-A model
% Parameters
Ms = 1.4e6; % (A/m)
a = 1050; % (A/m)
k = 80; % (A/m)
alpha = 2.1e-3;
c = 0.22;

J_model = mu0*JA_explicit(H_new,alpha,a,k,c,Ms);

figure; hold on;
plot(H,J);
plot(H_new,J_model);
xlabel('H (A/m)'); ylabel('J (T)');
grid on;

%% Cut a upper part of the curve
% Enough data for the fit
H_cut = H(J>=0);
J_cut = J(J>=0);

% Plot result
% figure; hold on;
% plot(H,J);
% plot(H_cut,J_cut);

%% Fit the model
% Initial parameters
Ms = 1.4e6; % (A/m)
a = 1000; % (A/m)
k = 80; % (A/m)
alpha = 2e-3;
c = 0.22;

param0 = [alpha,a,k,c,Ms];

% Optimize first with explicit J-A solver
ls_fun = @(param)ls_error(H_cut,J_cut,param,"explicit");
options = optimset('MaxFunEvals',1000,'Display','iter');
opt_param_exp = fminsearch(ls_fun,param0,options);

% Use explicit solver parameters as the initial guess for implicit solver
% ls_fun = @(param)ls_error(H_cut,J_cut,param,"implicit");
% options = optimset('MaxFunEvals',100,'Display','iter');
% opt_param = fminsearch(ls_fun,opt_param_exp,options);

%% Sample fitted model
% J_model = mu0*JA_explicit(H_new,param0(1),param0(2),param0(3),param0(4),param0(5));
J_model = mu0*JA_explicit(H_new,opt_param_exp(1),opt_param_exp(2),opt_param_exp(3),opt_param_exp(4),opt_param_exp(5));
% J_model = mu0*JA_implicit(H_new,opt_param(1),opt_param(2),opt_param(3),opt_param(4),opt_param(5));

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
            % my_coth = 1/(a/Heff) + (1/3)*(a/Heff) - (1/45)*(a/Heff)^3 + (2/945)*(a/Heff)^5;
            Man(i+1) = Ms*(coth(Heff/a) - a/Heff);
            % Man(i+1) = Ms*(my_coth - a/Heff);
        end
        M(i+1) = M(i) + dH*(1/(1+c))*(Man(i)-M(i))/(d*k-alpha*(Man(i)-M(i))) + (c/(1+c))*(Man(i+1)-Man(i));
    end    
end

%% Jiles-Atherton solution (Implicit Euler)
function M = JA_implicit(H,alpha,a,k,c,Ms)
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
            % my_coth = 1/(a/Heff) + (1/3)*(a/Heff) - (1/45)*(a/Heff)^3 + (2/945)*(a/Heff)^5;
            Man(i+1) = Ms*(coth(Heff/a) - a/Heff);
            % Man(i+1) = Ms*(my_coth - a/Heff);
        end
        % Find M(i+1)
        func = @(M_new)(M_new-M(i)) - (dH*(1/(1+c))*(Man(i+1)-M_new)/(d*k-alpha*(Man(i+1)-M_new)) + (c/(1+c))*(Man(i+1)-Man(i)));
        M(i+1) = fsolve(func, M(i), optimset('Display','off','MaxIter',100));
    end    
end

%% Least-squares objective function
function [err,J_model] = ls_error(H_true,J_true,param,solver)
    mu0 = 4*pi*1e-7;
    % Add initial curve points to the H-field
    % (H must start at H_max)
    N_init = 500;
    H = [linspace(0,H_true(1,1),N_init)';H_true];
    % Compute polarization using J-A model
    if solver == "explicit"
        M_model = JA_explicit(H,param(1),param(2),param(3),param(4),param(5));
    elseif solver == "implicit"
        M_model = JA_implicit(H,param(1),param(2),param(3),param(4),param(5));
    end
    J_model = mu0*M_model(N_init+1:end);
    % Compute the error
    % err = (1/length(J_true))*sum(log(abs(J_model-J_true)));
    err = (1/length(J_true))*sum((J_model-J_true).^2);
    % Plot
    % close all;
    % figure; hold on;
    % plot(H_true,J_true);
    % plot(H_true,J_model);
    % plot(H_true,(J_model - J_true).^2);
    % plot(H_true,log(abs(J_model-J_true)));
    % fprintf('Error: %d\n',err);
end

