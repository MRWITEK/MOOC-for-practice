function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);

for i=1:num_labels
    Y(i,:) = y==i;
end
Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);
temp = [Theta1_nobias(:); Theta2_nobias(:)];
X = [ones(m,1), X];
z_2 = Theta1 * X';
a_2 = [ones(1, m); sigmoid(z_2)];
z_3 = Theta2 * a_2;
h = sigmoid(z_3);
J = sum(sum(- log(h) .* Y - log(1-h) .* (1-Y)))/m +(lambda/(2*m)) * (temp' * temp);

    function g = sigmoidGradientFromSigmoid(sig)
        g = sig .* (1 .- sig);
    end


Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for i=1:m
    delta_3 = h(:,i) - Y(:,i);
    delta_2 = (Theta2_nobias' * delta_3) .* sigmoidGradientFromSigmoid(a_2(2:end,i));
    Delta_2 = Delta_2 + delta_3 * a_2(:,i)';
    Delta_1 = Delta_1 + delta_2 * X(i,:);
end
Theta1_grad = Delta_1 ./ m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1_nobias;
Theta2_grad = Delta_2 ./ m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2_nobias;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
