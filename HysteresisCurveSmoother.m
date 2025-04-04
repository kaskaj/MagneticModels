clear;

%% Settings
% Data points span of the smoothing algorithm (moving average)
smooth_span = 5;
% Number of grid interpolation points
interp_points = 1000; % for Hysteresis
% Does the data contain initial curve?
initial = false;
% Filename
filename = strcat('./data/test_curve_smoothed','.mat');
% filename_init = strcat('./data/test_curve_smoothed_init','.mat');

%% Load test curves
% Quasi-stationary measurement (With initial curve)
% curve = table2array(readtable('./data/DC_Sample.csv'));

% Quasi-stationary measurement (Without initial curve)
table_DC = readtable('./data/DC_Toroid.xlsx', 'Sheet', 3);
curve = [table_DC.FieldStrength_A_m_,table_DC.Polarisation_T_];

% Dynamic (50 Hz) measurement
% table_AC = readtable('./data/AC_Toroid_50.xlsx', 'Sheet', 3);
% curve = [table_AC.FieldStrength_A_m_,table_AC.Polarisation_T_];

%% Plot raw hysteresis curve
% figure; grid on;
% plot(curve(:,1),curve(:,2));

%% Separate initial magnetization curve and smooth

if initial
    % Assumes the data starts at [0,0] and constant stepsize
    [~,start] = max(curve(1:length(curve)/4,2));
    hyst = [smooth(curve(start:end,1),smooth_span),smooth(curve(start:end,2),smooth_span)];
    init = [smooth(curve(1:start,1),smooth_span),smooth(curve(1:start,2),smooth_span)];
else
    hyst = [smooth(curve(:,1),smooth_span),smooth(curve(:,2),smooth_span)];
end

%% Pre-horizontal (J/B) centering
H_diff = max(hyst(:,1)) + min(hyst(:,1));
hyst(:,1) = hyst(:,1) - H_diff/2;

%% Pre-vertical (H) centering
B_diff = max(hyst(:,2)) + min(hyst(:,2));
hyst(:,2) = hyst(:,2) - B_diff/2;

%% Split upper and lower parts
[~,i_max] = max(hyst(:,1));
[~,i_min] = min(hyst(:,1));
if i_min > i_max
    hyst = circshift(hyst, -i_min + 1);
    [~, i_min] = min(hyst(:,2));
    [~, i_max] = max(hyst(:,2));
end
hyst_up = hyst(i_min:i_max, :);
hyst_lo = [hyst(i_max:end, :); hyst(1:i_min, :)];

%% Remove redundant points
[~,i_un_up] = unique(hyst_up(:,1));
[~,i_un_lo] = unique(hyst_lo(:,1));
hyst_up = hyst_up(sort(i_un_up),:);
hyst_lo = hyst_lo(sort(i_un_lo),:);

%% Set interpolation grid (nonlinear)
H_max = min(abs([max(hyst_up(:,1)),max(hyst_lo(:,1)),min(hyst_up(:,1)),min(hyst_lo(:,1))]));
coef = 100;
H_new = (H_max/coef)*((coef+1).^((0:interp_points)/interp_points)-1);
H_new = [-flip(H_new),H_new(2:end)];

%% Find H-field offset
% Can sometimes act weird, can be turned off
% f = @(H_offset)offset(H_new,hyst_up,hyst_lo,H_offset);
% [H_offset,~] = fminunc(f,-6000);
% hyst_up(:,1) = hyst_up(:,1)-H_offset/2; 
% hyst_lo(:,1) = hyst_lo(:,1)-H_offset/2;

%% Interpolate data on a new grid
BHint_up = [H_new',interp1(hyst_up(:,1),hyst_up(:,2),H_new,'linear','extrap')'];
BHint_lo = [H_new',interp1(hyst_lo(:,1),hyst_lo(:,2),H_new,'linear','extrap')'];

%% Construct the final curve
% https://www.researchgate.net/publication/251427333_On_the_quantitative_analysis_and_evaluation_of_magnetic_hysteresis_data
B_new_up = (BHint_up(:,2)-flip(BHint_lo(:,2)))/2 - (BHint_up(end,2)-BHint_lo(end,2)+BHint_up(1,2)-BHint_lo(1,2))/4;
B_new_lo = -flip(B_new_up);
hyst_new_up = [H_new',B_new_up];
hyst_new_lo = [H_new',B_new_lo];
hyst_new = [hyst_new_up;flip(hyst_new_lo)];

%% Derive parameters
% Coercivity Hc (A/m)
dy = mean(abs(diff(hyst_new(:,2))));
i_Hc = find(hyst_new_up(:,2) > -dy & hyst_new_up(:,2) < dy);
Hc = mean(hyst_new_up(i_Hc));
% Remanence Br/Jr (T)
i_Jr = find(hyst_new(:,1) == 0);
Jr = hyst_new(i_Jr(1),2);
% Saturation (maximum) Bs/Js (T), Hs (A/m)
[Js,pos_sat] = max(hyst_new(:,2));   
Hs = hyst_new(pos_sat,1);
% Total / Hysteresis loss (J)
P = trapz(hyst_new(:,1),hyst_new(:,2));

%% Plot raw vs new hysteresis curve
figure; hold on; grid on;
p1 = plot(curve(:,1),curve(:,2));
p2 = plot(hyst_new(:,1),hyst_new(:,2),'k--');
xlabel('H (A/m)'); ylabel('J (T)');
% Plot Hc, Jr, Js
scatter([-Hc,Hc],[0,0],'ko','filled');
scatter([0,0],[-Jr,Jr],'ko','filled');
scatter([-Hs,Hs],[-Js,Js],'ko','filled');
legend([p1,p2],{'Raw','Smoothed'},'location','northwest');
hold off;

%% Save curves hysteresis curve
% save(filename_init,'init');
save(filename,'hyst');

%% Offset optimization
function R2 = offset(Hnew,BH_up,BH_lo,H_off)
    BHint_up = [Hnew',interp1(BH_up(:,1),BH_up(:,2),Hnew,'linear','extrap')'];
    BHint_lo = [Hnew',interp1(BH_lo(:,1)-H_off,BH_lo(:,2),Hnew,'linear','extrap')'];
    R2 = sum((BHint_up(:,2)-(-flip(BHint_lo(:,2)))).^2);
end
