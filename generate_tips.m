clear all;
close all;
clc;

cases = dir('./case76*');

iter_stage2_begin = 500;
iter_stage3_begin = iter_stage2_begin+10000;
iter_stage45_begin = iter_stage3_begin+18000;

state = 0;
for j=1:length(cases)
    caseNum = regexp(cases(j).name, '\d+', 'match');
    caseNum = caseNum{1};
    fileList = dir(strcat('./case',caseNum,'/phi*.mat'));
    out = strcat('./case',caseNum,'/tips_%2d');
    for i=1:length(fileList)-1
        filename = strcat(fileList(i).folder,'/',fileList(i).name);
        phi = load(filename).phi_plot;
        [lenu,lenv] = size(phi);
        iter = str2double(regexp(fileList(i).name, '\d+', 'match'));

        cue = zeros(lenu,lenv);
        for l = 1:lenu
            for m = 1:lenv
                cue(l,m) = sqrt((l-lenu/2)^2+(m-lenv/2)^2);
            end
        end

        if(iter<=iter_stage2_begin)
            theta_ori = ones(lenu,lenv);
            state = 1;
        elseif (iter < iter_stage3_begin) || (iter >=iter_stage45_begin)
            tips = sum_filter(phi,0);
            regionalMaxima = imregionalmax(full(tips));
            [Max_y,Max_x] = find(regionalMaxima);
            size_Max = length(Max_x);
            [theta_ori] = highlightZone(lenu,lenv,Max_x,Max_y,size_Max);
            state = 2;
        else
            phi_id = round(phi);
            dist = bwdistgeodesic(logical(phi_id),round(lenu/2),round(lenv/2));
            dist(isinf(dist))=0;
            dist(isnan(dist))=0;

            tip = sum_filter(phi_id,1);
            regionalMaxima = imregionalmax(full(tip));
            L_tip = bwconncomp(regionalMaxima,4);
            S_tip = regionprops(L_tip,'Centroid');
            centroids_tip = floor(cat(1,S_tip.Centroid));

            tips = zeros(1,L_tip.NumObjects);
            for ii = 1:L_tip.NumObjects
                tips(ii) = cue(centroids_tip(ii,1),centroids_tip(ii,2));
            end
            [~,tip_want_ind] = max(tips);

            max_x = centroids_tip(tip_want_ind,1);
            max_y = centroids_tip(tip_want_ind,2);

            [theta_ori] = highlightZone(lenu,lenv,max_x,max_y,1);
            state = 3;
        end

%         fprintf('Iter %d using %d\n',iter,state);
        save(sprintf(out,iter),'theta_ori');
% 
%         subplot(1,2,1);
%         imshow(phi);
%         subplot(1,2,2);
%         imshow(theta_ori);
%         drawnow;
    end
    
    disp(strcat('Finished case ',caseNum));
end

%%
% figure;
% subplot(1,2,1);
% imshow(phi);
% subplot(1,2,2);
% imshow(theta_ori);