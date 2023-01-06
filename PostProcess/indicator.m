%根据混淆矩阵计算OA,AA和Kappa值
%使用： 改变data变量的路径

clear;clc;

data = load('C:\Users\8\Desktop\mx_ppf.mat').data;
data = double(data);
[r,c] = size(data);
class_acc = [];
class_samples = sum(data);
diag = 0;
kappa_sum = 0;
for i = 1:r
    class_acc = [class_acc;data(i,i)/class_samples(i)];
    diag = diag + data(i,i);
    kappa_sum = kappa_sum + sum(data(i,:))*sum(data(:,i));
end

class_acc
OA = diag/sum(class_samples)
AA = mean(class_acc)

po = OA;
pe = kappa_sum/(sum(class_samples)*sum(class_samples));

Kappa = (po-pe)/(1-pe)