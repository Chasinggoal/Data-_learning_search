function p = Log_Permutation_multiple_query( b,x )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
% b is a col vector of weights; x is a cell col vector consisted of outputs
% of multiple queries

digits(10000);
addpath('/Users/David/Documents/MATLAB/DataPlus');
import Log_Permutation_single_query;
si = size(x);
s = si(1);
p = 0;
i = 1;
while i <= s
    current = x{i};
    p = p+(Log_Permutation_single_query(b, current));
    i =  i+1;
end


end

