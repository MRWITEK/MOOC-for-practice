#!/usr/bin/octave -f --no-gui-libs
% octave 5.2.0 on Linux: use --no-gui-libs to make fork() and waitpid() work (bug #45625)

function main()
% X y Xval yval
load('ex6data3.mat');

values = 2 .^ [0:12] .* 1e-2;
n = length(values);

% use single-threaded svmTrain to train SVM on the same dataset with different
% coefficients in parallel
cpu_count = nproc();
forks = zeros(cpu_count,1);
fork_idx = 1;
fork_count = 0;
filename_template = 'trained_%02i_%02i.mat';
for i=1:n
    for j=1:n
        if fork_idx > cpu_count
            fork_idx = 1;
        end
        if fork_count == cpu_count
            if forks(fork_idx) != 0
                waitpid(forks(fork_idx));
                fork_count -= 1;
                forks(fork_idx) = 0;
            end
        end
        [pid] = fork();
        if pid == 0 % fork
            model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j))); 
            % accuracy
            train_pred = svmPredict(model, X);
            train_pred = train_pred == y;
            train_pred = mean(train_pred);
            val_pred = svmPredict(model, Xval);
            val_pred = val_pred == yval;
            val_pred = mean(val_pred);
            filename = sprintf(filename_template, i, j);
            save('-binary', filename, 'model', 'train_pred', 'val_pred');
            exit
        else
            forks(fork_idx) = pid;
            fork_count += 1;
            fork_idx += 1;
        end
    end
end

for i = forks(forks != 0)
    waitpid(i);
end

end
main();
