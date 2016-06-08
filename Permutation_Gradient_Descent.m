function output = Permutation_Gradient_Descent(filedir,T,step )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% T is an integer with the number of iterations
% step is the step that we take for each iteration

addpath('/Users/David/Documents/MATLAB/DataPlus'); % add the function preparing to graph
import Permutation_single_query;
import Permutation_multiple_query;

[X,Y] = read_letor(filedir);
w = zeros(length(X(1, :)), 1);
param = zeros(length(w),T);
X_divid = cell(length(Y));                      % create a cell array preparing to cal the likelihood
cnt=0;
likelihood_vector = zeros(T,1); %documenting the likelihooda
for i = 1 : length(Y)
    [~, index] = sort(Y{i}, 'descend');     % sort Y in descending order
    tmpX = zeros(length(Y{i}), length(w));  % initialize temporary value for X as 0 matrix for gradient descent

    for j = 1 : length(Y{i})
        tmpX(j, :) = X(cnt + index(j), :);  % document the X that corresponds to each Y
    end
    X_divid{i} = tmpX;                      % documenting each X for each part of X_divid
    cnt = cnt + length(Y{i});
end

beta = sym('beta_%d',[1 length(w)]);
beta = beta.';
grad = cell(size(w));
permutation = sym(Permutation_multiple_query(beta,X_divid));
for i = 1 : length(w)
    grad{i} = diff(permutation, beta(i));
end

for loop = 1:T
    tempW = w;
    for i = 1: length(w)
        differentiation = sym(grad{i});
        display(differentiation);
        tempW(i) = w(i) + differentiation(w)*step;
    end
    w = tempW;
    param(:,loop) = w;
    likelihood = double(Permutation_multiple_query(w,X_divid)); % documenting the likelihood value
    display(likelihood);
    display(loop);
    likelihood_vector(loop)=likelihood;


end
output = param(:,T);
plot(1:T,likelihood_vector);
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
