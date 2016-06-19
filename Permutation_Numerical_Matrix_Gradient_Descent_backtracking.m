function output = Permutation_Numerical_Matrix_Gradient_Descent_backtracking(filedir)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% T is an integer with the number of iterations
% step is the step that we take for each iteration
% Computes gradient directly without pre-calculating the gradient function

digits(5000);

addpath('/Users/David/Documents/MATLAB/DataPlus'); % add the function preparing to graph
import Permutation_single_query;
import Permutation_multiple_query;
import Permutation_Gradient_Calc;
import Permutation_Single_Gradient_Calc;
import Permutation_Single_Compute_Gradient_Calc

T=1000;
step = 0.1; %initialize the step size
alpha = 0.8; % backtracking constant


[X,Y] = read_letor(filedir);
w = zeros(length(X(1, :)), 1);
param = zeros(length(w),T);
X_divid = cell(length(Y));                      % create a cell array preparing to cal the likelihood
cnt=0;
likelihood_vector = zeros(T,1); %documenting the likelihooda
likelihood_log_vector = zeros(T,1); % documenting the log likelihood for graph
for i = 1 : length(Y)
    [~, index] = sort(Y{i}, 'ascend');     % sort Y in ascending order
    tmpX = zeros(length(Y{i}), length(w));  % initialize temporary value for X as 0 matrix for gradient descent

    for j = 1 : length(Y{i})
        tmpX(j, :) = X(cnt + index(j), :);  % document the X that corresponds to each Y
    end
    X_divid{i} = tmpX;                      % documenting each X for each part of X_divid
    cnt = cnt + length(Y{i});
end


loop = 1;
while loop <= T
    tempW = w;
    wrec = w;
%     parfor (i = 1: length(w), 4) 
%         num_grad = 0;
%         for j = 1:length(Y)
%             for k = 1:length(Y{j})
%                 temp_grad = -Permutation_Single_Compute_Gradient_Calc(X_divid,w,j,k,i);
%                 num_grad = num_grad+temp_grad;
%             end
%         end
%         tempW(i) = w(i) + num_grad*step;
%     end
    for i=1:length(Y)
        X_cnt = X_divid{i};
        Y_cnt = Y{i};
        score_tol = X_cnt * w;
%         difference = zeros (length(Y_cnt));
%         parfor (j=1:length(difference), 4)
%             score_tmp = score_tol - score_tol(j);
%             difference(:,j) = difference(:,j)+score_tmp(:,j:length(score_tmp));
%         end
        exp_score_tol = exp(score_tol);
        exp_score = zeros (length(Y_cnt));
        parfor (j=1:length(exp_score),4)
            exp_score_tmp = exp_score_tol/exp_score_tol(j);
            exp_score_tmp(1:(j-1)) = zeros(j-1,1);
            exp_score(:,j) = exp_score(:,j) + exp_score_tmp;
        end
        parfor (k=1:length(w),4)
            X_cnt_w = X_cnt(:,k);
            difference = zeros(length(Y_cnt));
            for j=1:length(Y_cnt)
                difference_tmp = X_cnt_w - X_cnt_w(j);
                difference_tmp(1:j) = zeros(j,1);
                difference_tmp = difference_tmp.';
                difference(j,:) = difference(j,:)+ difference_tmp;
            end
            upper = difference * exp_score;
            upper = diag(upper);
            upper = upper.';
            lower = sum(exp_score,1);
            grad = upper./lower;
            tempW(k) = w(k) - sum(grad)*step;
        end
    end
    w = tempW;
    param(:,loop) = w;
    display(w);
    display(loop);
%     likelihood = Permutation_multiple_query(w,X_divid); % documenting the likelihood value
%     display(log(likelihood));
%     likelihood_vector(loop)=likelihood;
%     likelihood_log_vector(loop) = log(likelihood);
%     if loop > 1
%         if likelihood_log_vector(loop)-likelihood_log_vector(loop-1) < 0 && adj<2000
%             step = step * alpha;
%             w = wrec;
%             adj = adj+1;
%             continue;
%         elseif adj >= 2000
%             display('too much adjustments');
%             break;
%         end
%     end
%     if loop > 1
%         if abs(likelihood_log_vector(loop)-likelihood_log_vector(loop-1))<0.0000000000001
%             display('too small adjustment');
%             break;
%         end
%     end
    loop = loop + 1;
    adj = 0;
end
output = param(:,loop-1);
display(step);
% plot(1:(loop-1),likelihood_log_vector(1:(loop-1)));
end

function [X, Y] = read_letor(filename)  % X is a feature matrix; Y is a ranking cell array
    f = fopen(filename);
    display(filename)
    X = zeros(2e5, 0);
    qid = '';
    i = 0;
    q = 0;

    while 1 
        l = fgetl(f);
        if ~ischar(l)
            break;
        end

        i = i + 1;
        [lab,  ~, ~, ind] = sscanf(l, '%d qid:', 1); 
        l(1:ind-1) = [];	
        [nqid, ~, ~, ind] = sscanf(l, '%s', 1); 
        l(1:ind-1 )= []; 

        if ~strcmp(nqid, qid)
            q = q + 1;
            qid = nqid;
            Y{q} = lab;
        else
            Y{q} = [Y{q}; lab];
        end

        tmp = sscanf(l, '%d:%f'); 
        X(i, tmp(1 : 2 : end)) = tmp(2 : 2 : end);
    end
    
    X = X(1 : i, :);
    fclose(f);
end

%Permutation_Numerical_Compute_Gradient_Descent_backtracking('/Users/David/Documents/Undergraduate/Data+_Search_Algorithms/Pseudo_tester_data_listML/Fold11/train.txt');