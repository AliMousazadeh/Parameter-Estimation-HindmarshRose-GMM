close all; clear; clc;
addpath('./libQ');
%%%%%%%%%%

sInit = [-0.96; -3.67; 3.3];
vInit = sInit + randn(size(sInit)) .* sqrt(0.001);

I = 3.27;
r = 0.007;
IStart = 3.1;
IEnd = 3.3;
rStart = 0;
rEnd = 0.01;
ISearchStep = 1e-3;
rSearchStep = 1e-3;

interpolateStep = 10;
interpolateMethod = 'linear';
opts = odeset('RelTol',1e-6,'AbsTol',1e-7);
optsCost = odeset('RelTol',1e-5,'AbsTol',1e-6);
timeSpanCost = [0 100];
timeSpan = [0 500];

regularizeValue = eps;
gmmTrainMaxIter = 500;
nComponents = 64;
%%%%%%%%%%

s = ode45(@(t, x) hindmarshRose(x, I, r), timeSpan, sInit, opts);
T_s = s.x;
S = s.y;

X_s = S(1, :)';

X = S';

v = ode45(@(t, x) hindmarshRose(x, I, r), timeSpan, vInit, opts);

X_v = v.y(1, :)';
T_v = v.x;

figure;
plot(T_s, X_s, 'b', 'LineWidth', 1);
hold on;
plot(T_v, X_v, '--r', 'LineWidth', 1);
xlabel('t');
ylabel('x');
legend('S', 'V');
hold off;

figure;
plot3(S(1, :), S(2, :), S(3, :), 'b', 'LineWidth', 1);
hold on;
plot3(v.y(1, :), v.y(2, :), v.y(3, :), '--r', 'LineWidth', 1);
legend('S', 'V');
xlabel('X');
ylabel('Y');
zlabel('Z');


options = statset('MaxIter', gmmTrainMaxIter);
tic
GMMModel = fitgmdist(X, nComponents, 'Options', options, ...
    'Regularize', regularizeValue);
toc
dataML = -sum(log(pdf(GMMModel, X) + eps)) / size(X, 1);
fprintf('Data likelihood: %f\n', dataML);

tmp2 = posterior(GMMModel, X);
tmp3 = sum(tmp2, 1);
figure;
histogram(tmp3, 'Normalization','probability');
title('GMM');
xlabel('# of data points');
ylabel('frequency');
legend(strcat('histogram, ', strcat('mu=', ...
    mat2str(sum(tmp3) / nComponents))));

result = costFuncHRGMM(GMMModel, v);

Ivec = IStart:ISearchStep:IEnd;
rvec = rStart:rSearchStep:rEnd;

I_size = size(Ivec, 2);
r_size = size(rvec, 2);
costVals = zeros(I_size, r_size);

V = cell(I_size, r_size);
tic
for i = 1:I_size
    for j = 1:r_size
        V{i, j} = ode45(@(t, x) hindmarshRose(x, Ivec(i), rvec(j)), ...
            timeSpanCost, vInit, optsCost);
    end
end
toc


N = size(X, 1);
tic
for i = 1:1:I_size
    for j = 1:1:r_size
        costVals(i, j) = costFuncHRGMM(GMMModel, V{i, j});
    end
end
toc

[cost, idx] = min(costVals(:));
[I_row, I_col] = ind2sub(size(costVals), idx);
fprintf('Minimum of ML cost function: I = %f r = %f, ML = %f\n', Ivec(I_row), ...
    rvec(I_col), costVals(I_row, I_col));
fprintf('Correct I and r: %f %f\n', I, r);

[p, k] = meshgrid(Ivec, rvec);

iInterp = IStart:ISearchStep / interpolateStep:IEnd;
rInterp = rStart:rSearchStep / interpolateStep:rEnd;
[Xq, Yq] = meshgrid(iInterp, rInterp);
Vq = interp2(p, k, costVals', Xq, Yq, interpolateMethod);


figure;
sur1 = surf(Xq, Yq, Vq);
sur1.EdgeColor = 'none';
xlabel('I');
ylabel('r');
zlabel('J');
colormap jet