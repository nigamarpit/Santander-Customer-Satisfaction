function dataset = preProcessing(dataset)

[row, col] = size(dataset);

%replace the 999... and -999... with the mean of column
for j = 1:col
    count = 0;
    sumVal = 0;
    
    q1 = quantile(dataset(:,j), 0.25);
    q3 = quantile(dataset(:,j), 0.75);
    range = q3 - q1;
    k = 1.5;
    
    for i = 1:row
        if dataset(i,j) < q3+k*range || dataset(i,j) > q1-k*range
            count = count+1;
            sumVal = sumVal+dataset(i,j);
        end
    end
    
    m = sumVal/count;    
    
    for i = 1:row
        if dataset(i,j) == -999999 || dataset(i,j) == 999999
            dataset(i,j) = m;
        end
    end
end