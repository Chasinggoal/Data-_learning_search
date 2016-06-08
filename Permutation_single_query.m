function y = Permutation_single_query( b,x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
% x is a matrix of features with ordered rank, b is the weight set col
% vector
g = x*b;
e = exp(1);
i = 1;
s = size(g);
y = 1; 
while i<=s(1)
    y = double(y)*double(e^(g(i,1)));
    denom = 0;
    indic = 1;
    while indic <= s(1)-i+1
        denom = denom + double(e^(g(s(1)-indic+1,1)));
        indic = indic+1;
    end
    y = y/denom;
    i = i+1;
end


end

