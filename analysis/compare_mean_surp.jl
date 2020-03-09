# Do pairwise tests for effects of one model's surprisal estimates over and above the other,
# comparing SRN, GRU and LSTM after averaging surprisal over training repetitions (fully trained only)

using Distributions, CategoricalArrays

function add_mean_surp(new_col_name, surp_col_names)
    surp = surprisal[surp_col_names]
    mean_surp = Array{Union{Missing, Float64}}(undef, size(surp,1), 1)
    for i = collect(1:1:size(surp,1))
        mean_surp[i,:] .= mean(surp[i,:])
    end
    surprisal[new_col_name] = vec(mean_surp)
    surprisal
end

function benjamini(pvals,fdr=.05)
    #=Benjamini & Hochberg's (J. Royal Statistical Society B,1995) method for
    controlling the false discovery rate for multiple comparisons.

    pvals  : a list of (uncorrected) p-values for the individual comparisons
    fdr    : the desired false discovery rate. Default: .05

    Returns the indices of "pvals" corresponding to the null hypotheses that 
    should be rejected (i.e., significant effects).
    =#
    nrcomp = length(pvals)
    ind    = sortperm(pvals)    # Sort indices of p-values from lowest to highest p-value
    pvals  = pvals[ind]         # Sort p-values 
    sort(ind[1:findall(pvals .<= fdr*collect(1:nrcomp)/nrcomp)[end]])
end

function create_data_surp(data)   
    # Create data frame with human data and average surprisal of each RNN type
    
    # Add average surprisals
    surprisal = add_mean_surp(:srn, [ :SRN_6470000_0_surprisal, :SRN_6470000_1_surprisal, :SRN_6470000_2_surprisal, :SRN_6470000_3_surprisal, :SRN_6470000_4_surprisal, :SRN_6470000_5_surprisal])
    surprisal = add_mean_surp(:gru, [ :GRU_6470000_0_surprisal, :GRU_6470000_1_surprisal, :GRU_6470000_2_surprisal, :GRU_6470000_3_surprisal, :GRU_6470000_4_surprisal, :GRU_6470000_5_surprisal])
    surprisal = add_mean_surp(:lstm,[:LSTM_6470000_0_surprisal,:LSTM_6470000_1_surprisal,:LSTM_6470000_2_surprisal,:LSTM_6470000_3_surprisal,:LSTM_6470000_4_surprisal,:LSTM_6470000_5_surprisal])
    surprisal = add_mean_surp(:srn_prev, [ :SRN_6470000_0_surprisal_prev, :SRN_6470000_1_surprisal_prev, :SRN_6470000_2_surprisal_prev, :SRN_6470000_3_surprisal_prev, :SRN_6470000_4_surprisal_prev, :SRN_6470000_5_surprisal_prev])
    surprisal = add_mean_surp(:gru_prev, [ :GRU_6470000_0_surprisal_prev, :GRU_6470000_1_surprisal_prev, :GRU_6470000_2_surprisal_prev, :GRU_6470000_3_surprisal_prev, :GRU_6470000_4_surprisal_prev, :GRU_6470000_5_surprisal_prev])
    surprisal = add_mean_surp(:lstm_prev,[:LSTM_6470000_0_surprisal_prev,:LSTM_6470000_1_surprisal_prev,:LSTM_6470000_2_surprisal_prev,:LSTM_6470000_3_surprisal_prev,:LSTM_6470000_4_surprisal_prev,:LSTM_6470000_5_surprisal_prev])
    
    # Join human data with surprisal values and normalize suprisals
    surp_df = surprisal[[:srn, :gru, :lstm, :srn_prev, :gru_prev, :lstm_prev, :item]]
    data_surp = join(data, surp_df, kind=:left, on=:item, makeunique=true)
    data_surp[[:srn, :gru, :lstm, :srn_prev, :gru_prev, :lstm_prev]] = zscore(data_surp[[:srn, :gru, :lstm, :srn_prev, :gru_prev, :lstm_prev]])
    data_surp[!,1] = CategoricalArray(data_surp[!,1]) # subject
    data_surp[!,4] = CategoricalArray(data_surp[!,1]) # item
    data_surp
end

function do_model_comparisons(lmm_srn,lmm_gru,lmm_lstm,lmm_srn_gru,lmm_srn_lstm,lmm_gru_lstm,df)
    #=
    Compare deviance of models with one versus two sets of surprisal estimates. Returns 
    χ²-statistics and uncorrected p-values
    =#
    chi2_srn_gru   = deviance(lmm_gru)  - deviance(lmm_srn_gru)  
    chi2_srn_lstm  = deviance(lmm_lstm) - deviance(lmm_srn_lstm) 
    chi2_gru_srn   = deviance(lmm_srn)  - deviance(lmm_srn_gru)  
    chi2_gru_lstm  = deviance(lmm_lstm) - deviance(lmm_gru_lstm) 
    chi2_lstm_srn  = deviance(lmm_srn)  - deviance(lmm_srn_lstm) 
    chi2_lstm_gru  = deviance(lmm_gru)  - deviance(lmm_gru_lstm) 

    p_srn_gru  = 1-cdf(Chisq(df), chi2_srn_gru)
    p_srn_lstm = 1-cdf(Chisq(df), chi2_srn_lstm)
    p_gru_srn  = 1-cdf(Chisq(df), chi2_gru_srn)
    p_gru_lstm = 1-cdf(Chisq(df), chi2_gru_lstm)
    p_lstm_srn = 1-cdf(Chisq(df), chi2_lstm_srn)
    p_lstm_gru = 1-cdf(Chisq(df), chi2_lstm_gru)

    (Array([missing      chi2_srn_gru   chi2_srn_lstm;
           chi2_gru_srn  missing        chi2_gru_lstm;
           chi2_lstm_srn chi2_lstm_gru missing]),
     Array([missing    p_srn_gru  p_srn_lstm;
            p_gru_srn  missing    p_gru_lstm;
            p_lstm_srn p_lstm_gru missing]))
end

function print_model_comparisons(chi2,df,sign)
    # Print results table of pairwise comparisons (χ²-statistics and asterisks indicating corrected significance levels)
    
    function asterisks(sign,idx)
        if idx in sign[1]
            stars = "***"
        elseif idx in sign[2]
            stars = "**"
        elseif idx in sign[3]
            stars = "*"
        else
            stars = ""
        end
    end

    chi2text = " χ²(" * repr(df) * ") = "
    println("SRN over GRU :" * chi2text * repr(round(chi2[1,2],digits=3)) * asterisks(sign,3))
    println("SRN over LSTM:" * chi2text * repr(round(chi2[1,3],digits=3)) * asterisks(sign,5))
    println("GRU over SRN :" * chi2text * repr(round(chi2[2,1],digits=3)) * asterisks(sign,1))
    println("GRU over LSTM:" * chi2text * repr(round(chi2[2,3],digits=3)) * asterisks(sign,6))
    println("LSTM over SRN:" * chi2text * repr(round(chi2[3,1],digits=3)) * asterisks(sign,2))
    println("LSTM over GRU:" * chi2text * repr(round(chi2[3,2],digits=3)) * asterisks(sign,4))
end

function do_fit(data_type, surp_from)
    # Fit LMM with all factors named in 'surp_from' and other factors depending on data_type
    # ("SPR", "ET" or "EEG)
    
    function fixef(surp_from)
        fixef_str = ""
        for surp = surp_from
            fixef_str *= surp * "+"
        end
        fixef_str
    end

    function ranef(surp_from)
        ranef_str = ""
        for surp = surp_from
            ranef_str *= "(0+" * surp * "|subj_nr) +"
        end
        ranef_str
    end

    if data_type == "SPR"
        formula_str = "(logRT ~" * fixef(surp_from) * 
                      "logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos + (1|subj_nr) +" * ranef(surp_from) * 
                      "(0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr) + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item))"
    elseif data_type == "ET"
        formula_str = "(logRTfirstpass ~" * fixef(surp_from) * 
                      "log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix + (1|subj_nr) +" * ranef(surp_from) *
                      "(0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr) + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item))"
    elseif data_type == "EEG"
        formula_str = "(N400 ~" * fixef(surp_from) * 
                      "baseline + log_freq * nr_char * word_pos+ (1|subj_nr) +" * ranef(surp_from) *
                      "(0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr) + (1|item))"
    else error("Unknown data type"*data_type)
    end

    fit!(LinearMixedModel(eval(Meta.parse("@formula" * formula_str)), data_surp))
end
#-------------------------------------------------------------------------------------------------------------------
println("Doing pairwise model comparisons")

#=
data_SPR = read_data("SPR")
data_ET  = read_data("ET")
data_EEG = read_data("EEG")
=#

# Fit regression models with average surprisals from each RNN and from each pair
println("Running SPR analyses")
data_surp = create_data_surp(data_SPR)

lmm_srn_SPR  = do_fit("SPR", ["srn", "srn_prev"])
lmm_gru_SPR  = do_fit("SPR", ["gru", "gru_prev"])
lmm_lstm_SPR = do_fit("SPR", ["lstm","lstm_prev"])

lmm_srn_gru_SPR  = do_fit("SPR", ["srn", "srn_prev", "gru", "gru_prev"])
lmm_srn_lstm_SPR = do_fit("SPR", ["srn", "srn_prev", "lstm", "lstm_prev"])
lmm_gru_lstm_SPR = do_fit("SPR", ["gru", "gru_prev", "lstm", "lstm_prev"])

println("Running ET analyses")
data_surp = create_data_surp(data_ET)

lmm_srn_ET  = do_fit("ET", ["srn", "srn_prev"])
lmm_gru_ET  = do_fit("ET", ["gru", "gru_prev"])
lmm_lstm_ET = do_fit("ET", ["lstm","lstm_prev"])

lmm_srn_gru_ET  = do_fit("ET", ["srn", "srn_prev", "gru", "gru_prev"])
lmm_srn_lstm_ET = do_fit("ET", ["srn", "srn_prev", "lstm", "lstm_prev"])
lmm_gru_lstm_ET = do_fit("ET", ["gru", "gru_prev", "lstm", "lstm_prev"])

println("Running EEG analyses")
data_surp = create_data_surp(data_EEG)

lmm_srn_EEG = do_fit("EEG", ["srn"])
lmm_gru_EEG = do_fit("EEG", ["gru"])
lmm_lstm_EEG = do_fit("EEG", ["lstm"])

lmm_srn_gru_EEG = do_fit("EEG", ["srn",  "gru"])
lmm_srn_lstm_EEG = do_fit("EEG", ["srn", "lstm"])
lmm_gru_lstm_EEG = do_fit("EEG", ["gru", "lstm"])

# Do the model comparisons
chi2_SPR, p_SPR = do_model_comparisons(lmm_srn_SPR, lmm_gru_SPR, lmm_lstm_SPR, lmm_srn_gru_SPR, lmm_srn_lstm_SPR, lmm_gru_lstm_SPR, 4)
chi2_ET, p_ET   = do_model_comparisons(lmm_srn_ET,  lmm_gru_ET,  lmm_lstm_ET,  lmm_srn_gru_ET,  lmm_srn_lstm_ET,  lmm_gru_lstm_ET,  4)
chi2_EEG, p_EEG = do_model_comparisons(lmm_srn_EEG, lmm_gru_EEG, lmm_lstm_EEG, lmm_srn_gru_EEG, lmm_srn_lstm_EEG, lmm_gru_lstm_EEG, 2)

#=
lmm_srn_SPR  = fit(LinearMixedModel, @formula(logRT ~ srn + srn_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                         + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                         + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
lmm_gru_SPR  = fit(LinearMixedModel, @formula(logRT ~ gru + gru_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                         + (1|subj_nr) + (0+gru|subj_nr) + (0+gru_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                         + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)                                             
lmm_lstm_SPR = fit(LinearMixedModel, @formula(logRT ~ lstm + lstm_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                         + (1|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                         + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)

lmm_srn_gru_SPR  = fit(LinearMixedModel, @formula(logRT ~ srn + srn_prev + gru  + gru_prev  + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                             + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+gru|subj_nr)  + (0+gru_prev|subj_nr)  + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                             + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
lmm_srn_lstm_SPR = fit(LinearMixedModel, @formula(logRT ~ srn + srn_prev + lstm + lstm_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                             + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                             + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
lmm_gru_lstm_SPR = fit(LinearMixedModel, @formula(logRT ~ gru + gru_prev + lstm + lstm_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                             + (1|subj_nr) + (0+gru|subj_nr) + (0+gru_prev|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                             + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)

#-------------------------------------------------------------------------------------------------------------------
println("Running ET analyses")
data_surp = create_data_surp(data_ET)

lmm_srn_ET  = fit(LinearMixedModel, @formula(logRTfirstpass ~ srn + srn_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                          + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                          + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)
lmm_gru_ET  = fit(LinearMixedModel, @formula(logRTfirstpass ~ gru + gru_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                          + (1|subj_nr) + (0+gru|subj_nr) + (0+gru_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                          + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)
lmm_lstm_ET = fit(LinearMixedModel, @formula(logRTfirstpass ~ lstm + lstm_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                          + (1|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                          + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)

lmm_srn_gru_ET  = fit(LinearMixedModel, @formula(logRTfirstpass ~ srn + srn_prev + gru + gru_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                              + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+gru|subj_nr) + (0+gru_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                              + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)
lmm_srn_lstm_ET = fit(LinearMixedModel, @formula(logRTfirstpass ~ srn + srn_prev + lstm + lstm_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                              + (1|subj_nr) + (0+srn|subj_nr) + (0+srn_prev|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                              + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)
lmm_gru_lstm_ET = fit(LinearMixedModel, @formula(logRTfirstpass ~ gru + gru_prev + lstm + lstm_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                              + (1|subj_nr) + (0+gru|subj_nr) + (0+gru_prev|subj_nr) + (0+lstm|subj_nr) + (0+lstm_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                              + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)

#-------------------------------------------------------------------------------------------------------------------
println("Running EEG analyses")
data_surp = create_data_surp(data_EEG)

lmm_srn_EEG  = fit(LinearMixedModel, @formula(N400 ~ srn + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+srn|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
lmm_gru_EEG  = fit(LinearMixedModel, @formula(N400 ~ gru + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+gru|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
lmm_lstm_EEG = fit(LinearMixedModel, @formula(N400 ~ lstm + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+lstm|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)

lmm_srn_gru_EEG  = fit(LinearMixedModel, @formula(N400 ~ srn + gru + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+srn|subj_nr) + (0+gru|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr)
                                       + (1|item)), data_surp)
lmm_srn_lstm_EEG = fit(LinearMixedModel, @formula(N400 ~ srn + lstm + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+srn|subj_nr) + (0+lstm|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr)
                                       + (1|item)), data_surp)
lmm_gru_lstm_EEG = fit(LinearMixedModel, @formula(N400 ~ gru + lstm + baseline + log_freq * nr_char * word_pos
                                       + (1|subj_nr) + (0+gru|subj_nr) + (0+lstm|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr)
                                       + (1|item)), data_surp)
=#
#-----------------------------------------



#---------------------------------------------------
# Perform FDR correction at three different α levels
#--------------------------------------------------

p_vals = [p_SPR[:]; p_ET[:]; p_EEG[:]]   # concatenate all p-values column wise
p_vals = p_vals[.~ismissing.(p_vals)]    # remove missing values

sign001 = benjamini(p_vals,.001)                             # indices of p_vals with corrected p<.001
sign01  = setdiff(benjamini(p_vals,.01), sign001)            # indices of p_vals with corrected .001<p<.01
sign05  = setdiff(benjamini(p_vals,.05), [sign01;sign001])   # indices of p_vals with corrected .01<p<.05

# Split by data type
sign_SPR = (sign001[sign001.<=6],                      sign01[ sign01 .<=6],                      sign05[ sign05 .<=6])
sign_ET  = (sign001[(sign001.>6) .& (sign001.<=12)].-6, sign01[ (sign01 .>6) .& (sign01 .<=12)].-6, sign05[ (sign05 .>6) .& (sign05 .<=12)].-6)
sign_EEG = (sign001[sign001.>12].-12,                   sign01[ sign01 .>12].-12,                   sign05[ sign05 .>12].-12)

println("\nSelf-paced reading")
println("------------------")
print_model_comparisons(chi2_SPR,4,sign_SPR)

println("\nEye tracking")
println("------------------")
print_model_comparisons(chi2_ET, 4, sign_ET)

println("\nEEG")
println("------------------")
print_model_comparisons(chi2_EEG,4,sign_EEG)