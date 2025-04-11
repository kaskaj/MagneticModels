clear;
mu0 = 4*pi*1e-7;

%% Test inverse Langevin function
% Test parameters
Ms = 1.7; a = 2000;
% Applied field
H = linspace(-1e4,1e4,1000);
% Langevin function
M = Ms*(coth(H/a)- a./H);
% Inverse Langevin function
% Source: https://doi.org/10.1016/j.jnnfm.2015.05.007
H_inv = sign(M).*a.*(3*(abs(M)./Ms) + (1/5)*(abs(M)./Ms).^2 .* sin((7/2)*(abs(M)./Ms)) + (1./(1-(abs(M)./Ms))).*(abs(M)./Ms).^3);
% Source: https://doi.org/10.1016/j.jnnfm.2017.09.003
H_inv2 = sign(M).*a.*((abs(M)./Ms).*(3 - (773/768)*(abs(M)./Ms).^2 - (1300/1351)*(abs(M)./Ms).^4 + (501/340)*(abs(M)./Ms).^6 - (678/1385)*(abs(M)./Ms).^8))./((1-(abs(M)./Ms)).*(1+(866/853)*(abs(M)./Ms)));

figure; hold on;
plot(M,(H-H_inv)./H);
plot(M,(H-H_inv2)./H);
grid on; legend('Rel. error 1', 'Rel. error 2');

%% Load hysteresis curve
% % Hysteresis curve without initial magnetization
% load('./data/test_curve_smoothed.mat','hyst');
% % Assuming H starts at -H_max:
% H = -hyst(:,1);
% M = -hyst(:,2)/mu0;

%% Test hysteresis model
% https://doi.org/10.1109/TMAG.2024.3405644
% Parameters
Ms = 1.7e6; % (A/m)
a = 2000; % (m/A)
Hc = 2000; % (A/m)
chi = 100;

% Magnetization vector
N = 1e4;
Mmax = 1.5e6;
M = [linspace(1e-6,Mmax,N), linspace(Mmax,-Mmax,2*N), linspace(-Mmax,Mmax,2*N)];

% Compute model H(M)
H_model = hyst_explicit(M,Ms,a,Hc,chi);

% Plot
figure;
plot(H_model,M);
grid on;

%% Hysteresis solution (Explicit Euler)
function H = hyst_explicit(M,Ms,a,Hc,chi)
    N = length(M);
    H = zeros(N,1);
    Hi = zeros(N,1);
    for k = 1:N-1
        dM = (M(k+1)-M(k));
        e = sign(dM);        
        
        Hi(k+1) = (1 - e*Hi(k)/Hc)*(1/chi)*dM + Hi(k);
        Hr = a*sign(M(k+1))*(3*(abs(M(k+1))/Ms) + (1/5)*(abs(M(k+1))/Ms)^2 * sin((7/2)*(abs(M(k+1))./Ms)) + (1/(1-(abs(M(k+1))/Ms)))*(abs(M(k+1))/Ms)^3);
        H(k+1) = Hr + Hi(k+1);
    end    
end