function y = Log_Permutation_single_query( b,x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% x is a matrix of features with ordered rank, b is the weight set col
% vector

digits(10000);
g = x*b;
e = exp(1);
i = 1;
s = size(g);
y = 0; 
while i<=s(1)
    denom = 0;
    indic = 1;
    while indic <= s(1)-i+1
        denom = denom + e^(g(s(1)-indic+1,1));
        indic = indic+1;
    end
    res = log((e^(g(i,1)))/denom);
    y = y + res;
    i = i+1;
end


end

