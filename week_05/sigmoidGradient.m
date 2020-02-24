function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

%sig = sigmoid(z);
%g = sig .* (1 .- sig);

% d(sigmoid(z))/dz = exp(z)./(exp(z)+1).^2
ex = exp(z);
ex2 = ex .+ 1;
ex2 = ex2 .* ex2;
g = ex ./ ex2;
end
