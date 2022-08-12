function [theta] = highlightZone_test(tip)
[ind_x, ind_y] = find(tip~=0);

[lenu,lenv] = size(tip);
theta = zeros(lenu,lenv);
ttt = zeros(lenu,lenv);
for l=1:length(ind_x)
    max_x = ind_x(l);
    max_y = ind_y(l);
    for i = max_y-2:max_y+2
        for j = max_x-2:max_x+2
                ttt(i,j) = 1;
        end
    end
    theta = theta+ttt;    
end
theta(abs(theta)>0) = 1;
theta = theta.';