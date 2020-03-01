function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% word_indices is a vector with indexes into a dictionary list,
% x is a vector of length n, element at position i is 1 when word_indices contains i, 0 otherwise

x = any(eye(n)(word_indices,:),1)';
end
