function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
scale = lambda / m;
h = sigmoid(X * theta);
temp = theta(2:end,:);
J = -((log(h))' * y + (log(1-h))' * (1-y)) / m + scale/2 * temp' * temp;
grad = X' * (h-y) / m + scale * [0; temp];
end
