function [X, Y] = read_letor(filename)
    f = fopen(filename);
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
        display(tmp);
        display(tmp(1 : 2 : end));
        display(tmp(2 : 2 : end));
        X(i, tmp(1 : 2 : end)) = tmp(2 : 2 : end);
    end
    
    
    X = X(1 : i, :);
    fclose(f);