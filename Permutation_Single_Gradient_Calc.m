function output = Permutation_Single_Gradient_Calc(x, b, qid, oid, beta_id)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% provides a specifc part of the gradient

query = x{qid};
s = size(query);
item_number = s(1);
e = exp(1);

under = 0;
upper = 0;

if oid == item_number
    output = 1;
else
    for loop = oid+1 : item_number
        count = query(loop,:)-query(oid,:);
        under = under + e^(count*b);
        upper = upper + e^(count*b)*count(beta_id);
    end
    if upper == 0
        output = 0;
    else
        output = upper/(under+1) + dot(b,ones(1,length(b)));
        output = matlabFunction(output);
    end
end
end

