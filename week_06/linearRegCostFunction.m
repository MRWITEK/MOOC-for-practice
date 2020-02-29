function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);
delta = X * theta - y;
temp = theta(2:end,:);
J = (delta' * delta) ./ (2 .* m) + (lambda ./ (2 .* m)) .* (temp' * temp);
grad = X' * delta ./ m;
grad(2:end,:) += (lambda ./ m) .* temp;
end
