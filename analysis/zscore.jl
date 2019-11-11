# Normalize all columns from data frame or array (not in-place)
function zscore(x)

    if typeof(x) == DataFrame
        for c = 1:length(x)
            x[:,c] = zscore(x[:,c])
        end
    else
        x = (x.-mean(skipmissing(x))) / std(skipmissing(x))
        x = convert(Array{Union{Float64, Missings.Missing},1},x)        # Because for some reason, the type of x becomes Any sometimes
    end
    x
end
