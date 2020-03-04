function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    cv_pred = pval < epsilon;
    true_pos =  sum( yval &  cv_pred);
    false_pos = sum(!yval &  cv_pred);
    false_neg = sum( yval & !cv_pred);
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
end
