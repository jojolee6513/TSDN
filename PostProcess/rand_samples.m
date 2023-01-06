function out = rand_samples(data,rate)

[r,c] = size(data);
out = zeros(r*c,1);
index = find(data);
num = size(index);num = num(1);
order = randperm(num);
index = index(order);
samples = floor(num*rate);

for i = 1:samples
    out(index(i)) = data(index(i));
end
out = reshape(out,r,c);


