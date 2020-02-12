function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

J_history = zeros(num_iters, 1);
scale = alpha / length(y);
for iter = 1:num_iters
    theta = theta - scale * X' * (X * theta - y);
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end
end
