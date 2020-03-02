function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

[m n] = size(X);
logical = eye(K)(idx,:);
scaling = repelem(1 ./ sum(logical,1)',1,n);
avg_sum = sum(repmat(X,1,K) .* repelem(logical,1,n),1)';
centroids = reshape(avg_sum,n,K)' .* scaling;
end
