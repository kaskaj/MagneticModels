clear;

%% Load loss data
load('./data/data_loss.mat','data_loss');

P = data_loss(:,1); % Specific loss (W/kg)
B = data_loss(:,2); % Magnetic induction (T)
f = data_loss(:,3); % Frequency (Hz)

freq = unique(f); % Vector of unique frequencies
Nf = length(freq);

%% Fit total loss
% P = w1 * B^w2 * f^w3

% Initial parameter guesses
N = 20;
w1_0 = linspace(1e-4, 10, N);
w23_0 = linspace(0.1, 3, N);
[W1_0, W2_0, W3_0] = meshgrid(w1_0, w23_0, w23_0);
W1_0 = W1_0(:);
W2_0 = W2_0(:);
W3_0 = W3_0(:);

% Construct matrix X
% P = w1 * B^w2 * f^w3
% log(P) = log(w1) + w2*log(B) + w3*log(f)
% P = exp(X*w)
X = [ones(length(B),1), log(B), log(f)];

% Anonymous optimized function
fun = @(w)ls_error(w, X, P);

% Optimize for different initial guesses
err_best = inf;
for k = 1:length(W1_0)
    [w, err] = fminunc(fun, [W1_0(k), W2_0(k), W3_0(k)]', optimoptions('fminunc','Display','off'));
    if err < err_best
        err_best = err;
        w_best = w;
        fprintf('Iteration: %d, Error: %d \n', k, err_best);
    end
end

% Simple Least Squares fit
w_ls = (X'*X) \ (X'*log(P));

%% Sample the model
B_sample = linspace(min(B), max(B), 100);
[B_sample, f_sample] = meshgrid(B_sample, freq);

X_new = [ones(length(B_sample(:)),1), log(B_sample(:)), log(f_sample(:))];
P_model = exp(X_new*w_best);
P_model_ls = exp(X_new*w_ls);

%% Coefficient of determination
P_model_0 = exp(X*w_best);
P_model_ls_0 = exp(X*w_ls);

R2 = 1-sum((P_model_0-P).^2)/sum((P-mean(P)).^2);
R2_ls = 1-sum((P_model_ls_0-P).^2)/sum((P-mean(P)).^2);

%% Plot fit
figure; hold on;
for k = 1:Nf
    % Original data
    s = plot(B(f==freq(k)),P(f==freq(k)),'o');
    % fminunc fit
    p(k) = plot(B_sample(f_sample==freq(k)),P_model(f_sample==freq(k)),'Color',s.Color);
    % Simple LS fit
    plot(B_sample(f_sample==freq(k)),P_model_ls(f_sample==freq(k)),'--','Color',s.Color);
end
xlabel('B (T)');
ylabel('P (W/kg)');
set(gca, 'YScale', 'log');
legend(p,{'20 Hz','50 Hz','100 Hz','400 Hz','800 Hz'},'location','southeast');
grid on;

%% Optimized function
function err = ls_error(w, X, y)
    % Predict y
    y_predict = exp(X*w);
    % Compute Error
    err = (1/length(y))*sum((y_predict-y).^2);
end
