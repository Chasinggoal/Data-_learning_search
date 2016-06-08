function output = listMLE_Plotting_Likelihood_nonNDCGbased(dataset)
NDCGTR = zeros(5, 10); % document the NDCG for training data set(a method to measure the ranking quality; known as normalised discounted cumulative gain)
NDCGVA = zeros(5, 10); % document the NDCG for validation data set
NDCGTE = zeros(5, 10); % document the NDCG for testing data set
outfile = 'out.txt';   % the outfile is used to document the performance by using NDCG
T = 500000;               % number of iterations (Note: 500 is just a random number; we can improve by finding an optimal number)
times = 1;             % frequency to document the value of w (beta vector)
rate = 0.001;           % length of the step (Note: 0.01 is a random small step; in the future we need to code something to ensure convergence)
addpath('/Users/David/Documents/MATLAB/DataPlus'); % add the function preparing to graph
import Permutation_single_query;
import Permutation_multiple_query;

likelihood_vector = zeros(T,1); %documenting the likelihooda


% divid the data into five folders, and go through each of them
for fold = 1 : 1
    dname = [dataset '/Fold' num2str(fold) ];    % name of training data
    [X, Y] = read_letor([dname '/train.txt']);      % read training data
    w = zeros(length(X(1, :)), 1);                  % initialize value for w as 0 vector
    param = zeros(length(w), T / times);            % every T/times store w
    in = 1;                                         % index of current parameter
    X_divid = cell(length(Y));                      % create a cell array preparing to cal the likelihood
    % T iterations
    for loop = 1 : T
        cnt = 0;                                    % index corresponding to X
        delta = zeros(1, length(w));                % initialize the increasement as delta as 0 vector
        for i = 1 : length(Y)
            [~, index] = sort(Y{i}, 'descend');     % sort Y in descending order
            tmpX = zeros(length(Y{i}), length(w));  % initialize temporary value for X as 0 matrix for gradient descent

            for j = 1 : length(Y{i})
                tmpX(j, :) = X(cnt + index(j), :);  % document the X that corresponds to each Y
            end
            X_divid{i} = tmpX;                      % documenting each X for each part of X_divid
            product = tmpX * w;                     % 
            totalexp = zeros(1, 3);                 % sum of all exponentials
            tmpexp = exp(product);
            totalexp(1) = sum(tmpexp);              %1,2,3 corresponds to the sum of three different kinds of exp
            totalexp(2) = totalexp(1) - exp(product(1));
            totalexp(3) = totalexp(2) - exp(product(2));

            result = zeros(1, 3);
            tmpresult = tmpexp' * tmpX;   % calculate exp times tmpX according to the formula in the paper

            % calculate the increasement for each item in w
            for inx = 1 : length(w)
                result(1) = tmpresult(inx);
                result(2) = result(1) - exp(product(1)) * tmpX(1, inx);
                result(3) = result(2) - exp(product(2)) * tmpX(2, inx);

                tmpx = [tmpX(1, inx), tmpX(2, inx), tmpX(3, inx)];
                delta(inx) = double(delta(inx)) + double(sum(result / totalexp - tmpx));
            end
            cnt = cnt + length(Y{i});
        end

        w = w - delta' .* rate;     % update value for w
        if (mod(loop, times) == 0)  % document w when needed
            param(:, in) = w;
            in = in + 1;
        end
        likelihood = double(Permutation_multiple_query(w,X_divid)); % documenting the likelihood value
        display(likelihood);
        display(loop);
        likelihood_vector(loop)=likelihood;
    end
    [~,tin]= max(likelihood_vector);
    w = param(:, tin);
    display(tin);
    output=w;
    plot(1:T,likelihood_vector);

    
    
    
    

    
    
end

% outputing the NDCG value into the out.txt file to understand how well the
% model works


% function below is used to read the SVMLight formatting of feature data
% and ranks
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
	


        
        
        
        
        
        
        
        
        
        
        
        
        