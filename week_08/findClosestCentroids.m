function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
m = size(X,1);
distances = sum((repmat(X,K,1) - repelem(centroids,m,1)) .^ 2, 2);
[_, idx] = min(reshape(distances, m, K),[],2);
end
