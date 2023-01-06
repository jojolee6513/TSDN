clear;clc;
%载入超像素分割图
sup_map = load('C:\Users\8\Desktop\matlab code\multiscale_superpixel_segmentation\Houston_suppixel2.mat').a;

%载入预测矩阵以及索引矩阵，生成结果图
cnn_map = zeros(349,1905);
gcn_map = zeros(349,1905);
%绘制结果图
index = load('index.mat').data;
pcnn = load('pred_cnn.mat').data;
pgcn = load('pred_gcn.mat').data;
gth = imread('Houston_test.tif');
gth_tr = imread('Houston_test.tif');

for s=1:12197
    [max1,label1] = max((pcnn'));
    [max2,label2] = max((pgcn'));
    %Matlab的索引是从1开始的，但numpy是从0
    %所以要+1
    x = index(1,s)+1;
    y = index(2,s)+1;
    cnn_map(x,y) = label1(s);
    gcn_map(x,y) = label2(s);
end
% figure;
imagesc(cnn_map);title('cnn result');
% figure;
imagesc(gcn_map);title('gcn result');
% figure;
imagesc(gth);title('gth');
% figure;
error_map = double(gth)-cnn_map;
%error_map = gcn_map-cnn_map;
imagesc(error_map);title('error');
%加载掩码
mask = load('mask2.mat').area;
% figure;
err_mask = cnn_map.*mask;
imagesc(cnn_map.*mask);title('select');

% 为每个像素点按照超像素的方法投票
cnn = cnn_map;
gcn = gcn_map;
% cnn = cnn_map;gcn = gcn_map;figure;imagesc(cnn);title('cor');

figure;imagesc(cnn);title('cnn');
for b = 1:max(max(sup_map))
    arr = [];
    loc = [];
    for i = 1:349
        for j = 1:1905
            if (sup_map(i,j)==b)&&(cnn(i,j)~=0)
                arr = [arr;cnn(i,j)];
                if (cnn(i,j)~=gcn(i,j))  
                    loc = [loc;i,j];
                end
                nums = size(loc);
                if(nums~=0)
                    cor = mode(arr);
                    for n = 1:nums
                        x = loc(n,1);
                        y = loc(n,2);
                        cnn(x,y) = cor;
                    end
                end
            end
        end
    end
end

% re = double(gth).*mask;

%这段代码说明超像素直接做后处理的提高
% counter = 0;
% for i = 1:349
%     for j = 1:1905
%         if(cnn(i,j)~= gth(i,j))
%             counter = counter + 1;
%         end
%     end
% end

figure;imagesc(cnn);title('post cnn');
cnn_1 = zeros(349,1905);
for i = 1:349
    for j = 1:1905
        if(cnn(i,j)~= cnn_map(i,j))
            cnn_1(i,j) = cnn(i,j);
        end
    end
end
figure;imagesc(cnn_1);title('minus');