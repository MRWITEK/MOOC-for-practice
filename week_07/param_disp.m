#!/usr/bin/octave -f
function main()
% plot accuracy, print optimal coefficients

values = 2 .^ [0:12] .* 1e-2;
n = length(values);

filename_template = 'trained_%02i_%02i.mat';
train_mat = zeros(n);
val_mat = zeros(n);
for i=1:n
    for j=1:n
            filename = sprintf(filename_template, i, j);
            load(filename, 'train_pred', 'val_pred');
            train_mat(i,j) = train_pred;
            val_mat(i,j) = val_pred;
    end
end

%hold on;
%surf(log(values), log(values), val_mat);
%mesh(log(values), log(values), train_mat);
%pause;
[i,j] = find(val_mat == max(max(val_mat)),1);
printf('C = %f\nsigma = %f\n', values(i), values(j));

end
main();
