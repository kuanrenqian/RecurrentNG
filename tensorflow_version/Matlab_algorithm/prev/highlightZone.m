function [theta] = highlightZone(lenu,lenv,Max_x,Max_y,size_Max)
theta = zeros(lenu,lenv);
ttt = zeros(lenu,lenv);
for l=1:size_Max
    max_x = Max_x(l);
    max_y = Max_y(l);
    for i = max_y-1:max_y+1
        for j = max_x-1:max_x+1
            try
                ttt(i,j) = 1;
            catch
            end
            %ttt(i,j) = 1;
        end
    end
    try
        theta = theta+ttt; 
    catch
    end   
end
theta(abs(theta)>0) = 1;