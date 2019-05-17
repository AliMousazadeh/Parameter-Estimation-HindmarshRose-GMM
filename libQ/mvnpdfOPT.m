function result = mvnpdfOPT(X, mu, predictedBMUs, R)
result = 0;
nComponents = size(R, 3);

n = size(X, 1);
d = size(X, 2);

K2 = d*log(2*pi)/2;
for iComponent = 1:nComponents
    compR = R(:, :, iComponent);
    compMu = mu(iComponent, :);
    K1 = sum(log(diag(compR)));
    boolCheck = eq(predictedBMUs, iComponent);
    X0 = X(boolCheck, :);
    if ~isempty(X0)
        nd = size(X0, 1);
        compMuBroad = repmat(compMu, nd, 1);
        X0 = X0 - compMuBroad;
        xRinv = X0 / compR;
        quadform = sum(xRinv.^2, 2);
        Z = -0.5*quadform - K1 - K2;
        compPDF = exp(Z);
        result = result - sum(log(compPDF + eps));
    end
end
result = result / n;
end