clear;clc;
gth = double(imread('Houston_map.tif'));

% index = load('index.mat').data;
% pred = load('pred.mat').data;
% [pred,labelp] = max(pred,[],2);
% label = zeros(349,1905);
% for i = 1:12197
%     x = index(1,i)+1;
%     y = index(2,i)+1;
%     label(x,y) = labelp(i);
% end
label = load('result.mat').houston2013_result;
for i = 1:349
    for j = 1:1905
        if((gth(i,j)~=0)&&(label(i,j)==0))
            label(i,j) = gth(i,j);
        end
    end
end

map = label2color(label,'houston2013');
% map = label2color(gth,'houston2013');
figure;
imshow(map);
imwrite(map,'p.png')
