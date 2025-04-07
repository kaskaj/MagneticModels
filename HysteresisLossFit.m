clear;
rho = 8000; % Volumetric density

%% Load BH hysteresis curves
BH = [];
load('./data/hysteresis_loss/B1.mat','hyst'); BH.B1 = hyst;
load('./data/hysteresis_loss/B2.mat','hyst'); BH.B2 = hyst;
load('./data/hysteresis_loss/B3.mat','hyst'); BH.B3 = hyst;
load('./data/hysteresis_loss/B4.mat','hyst'); BH.B4 = hyst;
load('./data/hysteresis_loss/B5.mat','hyst'); BH.B5 = hyst;
load('./data/hysteresis_loss/B6.mat','hyst'); BH.B6 = hyst;
load('./data/hysteresis_loss/B7.mat','hyst'); BH.B7 = hyst;

names = fieldnames(BH);
N = length(names);

%% Find maximum B values and curve surfaces
B_max = zeros(N,1);
W = zeros(N,1);
pos_max = zeros(N,1);

for i = 1:N
    BH_sample = BH.(names{i});
    [B_max(i), pos_max(i)] = max(BH_sample(:,2));
    W(i) = (1/rho)*trapz(BH_sample(:,1),BH_sample(:,2));
end

%% Saturation polarization treshold
i_lim = B_max < 1.7;
B_max_fit = B_max(i_lim);
W_fit = W(i_lim);

%% Fit Hysteresis loss
% Optimized function
f = @(w)steinmetz(w,B_max_fit,W_fit);
% Starting points
N = 100;
% Vector of optimized parameters
w = zeros(N,2);
% Vector of squared errors
errs = zeros(N,1);
% Initial values
init = linspace(1e-3,1e1,N);
[I1,I2] = meshgrid(init,init); I1 = I1(:); I2 = I2(:);
% Optimize
options = optimoptions('fminunc','display','off');
for i = 1:N^2
    [w(i,:),errs(i)] = fminunc(f,[I1(i),I2(i)],options);
end
% Select lowest error
[~,i_w] = min(errs);
w_opt = w(i_w,:);

%% Sample the model
B_sample = linspace(0.05,1.83,100);
W_sample = w_opt(1).*B_sample.^w_opt(2);

%% Plot
figure; hold on; grid on;
plot(B_sample,W_sample,'color',[0.5,0.5,0.5],'LineWidth',1);
plot([1.83,1.83],[0,2.5],'k--');
scatter(B_max,W,'kX');
xlabel('B (T)');
ylabel('W (J/kg)');

%% Optimized function
function err = steinmetz(w,x,y_true)
    y = w(1).*x.^w(2);
    err = sum((y-y_true).^2);
end
        

