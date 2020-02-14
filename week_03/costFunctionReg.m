function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
nn=length(theta);
h = sigmoid(X * theta);
scale = lambda / m;
J = -((log(h))' * y + (log(1-h))' * (1-y)) / m + scale/2 * theta(2:nn)' * theta(2:nn);
grad = X' * (h-y) / m + scale * [0; theta(2:nn)];
end
