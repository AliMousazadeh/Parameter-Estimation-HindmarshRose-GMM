function result = costFuncHRGMM(GMMModel, V)
X_new = V.y';

result = -sum(log(pdf(GMMModel, X_new) + eps)) / size(X_new, 1);
end