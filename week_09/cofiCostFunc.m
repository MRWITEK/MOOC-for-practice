function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

delta = X * Theta' - Y;
delta = delta .* R;
J = 0.5 * sum(sum(delta .* delta)) + lambda / 2 * sum(sum(X .* X)) + lambda / 2 * sum(sum(Theta .* Theta));
X_grad = delta * Theta + lambda * X;
Theta_grad = delta' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];
end
