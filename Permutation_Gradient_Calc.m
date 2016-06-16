function output = Permutation_Gradient_Calc( x, beta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% This function calculate the gradient for the permutation likelihood

digits(1000);
addpath('/Users/David/Documents/MATLAB/DataPlus');
import Permutation_Single_Gradient_Calc;
s1 = size(x);
s2 = size(x{1});
i = 1;
output = cell(s1(1), s2(1), length(beta));
while i <= s1
    j = 1;
    while j<= s2(1)
        parfor (k=1:length(beta), 4)
            output{i,j,k} = Permutation_Single_Gradient_Calc(x, beta, i, j, k);
        end
        j = j+1;
        display(j);
    end
    i = i+1;
end

    


end

