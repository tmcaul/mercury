function zi_one_hot_encode(data)
    # assumes zero-indexed categories
    one_hot=zeros(Float64, maximum(data)+1, size(data, 1));
    for i in 1:size(data, 1)
        label=data[i]+1
        one_hot[label, i]=1
    end;

    return one_hot

end;
