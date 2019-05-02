function read_data(experiment)

    #= Read data from one of the experiments (exp={SPR,ET,EEG}) and do preprocessing:
    - Log-transform RTs
    - Add columns for previous-word covariates
    - Remove rejected data. Note that log_freq_prev is missing for words following punctuation and clitics, so these
      items will not be analysed even though they are not explicitly rejected.
    - Normalize quantitative covariates. Surprisal values will be normalized upon use
    =#
    print("Reading "* experiment * " data files\n")
    data = CSV.read("data_" * experiment * ".csv", delim='\t')

    if experiment=="SPR"
        data[:logRT]         = log.(data[:RT])
        data[:log_freq_prev] = [missing; data[1:end-1,:log_freq]]
        data[:nr_char_prev]  = [missing; data[1:end-1,:nr_char]]
        data[:logRT_prev]    = [missing; data[1:end-1,:logRT]]

        data[[:log_freq, :nr_char, :word_pos, :log_freq_prev, :nr_char_prev, :logRT_prev]] = zscore(data[[:log_freq, :nr_char, :word_pos, :log_freq_prev, :nr_char_prev, :logRT_prev]])
    elseif experiment=="ET"
        data[:logRTfirstpass] = log.(data[:RTfirstpass])
        data[:log_freq_prev]  = [missing; data[1:end-1,:log_freq]]
        data[:nr_char_prev]   = [missing; data[1:end-1,:nr_char]]

        data[[:log_freq, :nr_char, :word_pos, :log_freq_prev, :nr_char_prev]] = zscore(data[[:log_freq, :nr_char, :word_pos, :log_freq_prev, :nr_char_prev]])
    else
        data[[:log_freq, :nr_char, :word_pos, :baseline]] = zscore(data[[:log_freq, :nr_char, :word_pos, :baseline]])
    end

    data = data[.!data[:reject_data],:]
    data = data[.!data[:reject_word],:]
    for s = subjects[(subjects[:experiment].==experiment) .& (subjects[:reject]),:subj_nr]
        data = data[.!(data[:subj_nr].==s),:]
    end

    data
end
