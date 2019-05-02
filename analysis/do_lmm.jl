# Fit linear mixed-effects models to each of the three human data sets

function fit_1surp(data, basedev, col, prevcol)
    # Fit to a single set of surprisal estimates by including them in the baseline model

    # Select current and previous-word surprisal values and combine with item numbers in a temporary data frame
    surp_df = [surprisal[[col, prevcol]] surprisal[[:item]]]
    rename!(surp_df, x=>n for (x,n) = zip(names(surp_df)[1:2], [:surp, :surp_prev]))

    # Join data with current surprisal values, compute weighted average surprisal, and normalize suprisals
    data_surp = join(data, surp_df, kind=:left, on=:item, makeunique=true)
    avsurp    = mean(skipmissing(data_surp[:surp]))
    data_surp[:surp]      = zscore(data_surp[:surp])
    data_surp[:surp_prev] = zscore(data_surp[:surp_prev])

    # Fit mixed-effects model (with surprisal factor(s)) to human data.
    # Correlation terms are not included because these mess with the log-likelihoods: They come to
    # diverge from the z-scores, which are more reliable because they are hardly affected by the correlation terms
    if :logRT in names(data)
        model = fit(LinearMixedModel, @formula(logRT ~ surp + surp_prev + logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                               + (1|subj_nr) + (0+surp|subj_nr) + (0+surp_prev|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                               + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_surp)
    elseif :logRTfirstpass in names(data)
        model = fit(LinearMixedModel, @formula(logRTfirstpass ~ surp + surp_prev + log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                               + (1|subj_nr) + (0+surp|subj_nr) + (0+surp_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                               + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_surp)
    else
        model = fit(LinearMixedModel, @formula(N400 ~ surp + baseline + log_freq * nr_char * word_pos
                                               + (1|subj_nr) + (0+surp|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr)
                                               + (1|item)), data_surp)
    end

    # Return deviance decrease (chi-squared), coefficient(s), z-score(s), and average surprisal
    if :N400 in names(data)
        chi2       = basedev - deviance(model)
        b_surp     = coeftable(model).cols[1][2]
        z_surp     = coeftable(model).cols[3][2]

        [chi2 b_surp z_surp avsurp]
    else
        chi2       = basedev - deviance(model)
        b_surp     = coeftable(model).cols[1][2]
        z_surp     = coeftable(model).cols[3][2]
        b_prevsurp = coeftable(model).cols[1][3]
        z_prevsurp = coeftable(model).cols[3][3]

        [chi2 b_surp z_surp b_prevsurp z_prevsurp avsurp]
    end
end

function fit_surps(data, basedev)
    # Loop over the different estimates for surprisal (i.e., over networks) and fit regression models.
    col       = surp_col
    prevcol   = prevsurp_col
    countdown = nrrep*nrtest

    for rep = 1:nrrep
        for test = 1:nrtest
            print(repr(countdown)*' ')      # Countdown to show how much left to do

            outfit_srn  = fit_1surp(data, basedev, col, prevcol)
            outfit_gru  = fit_1surp(data, basedev, col+nrtest,  prevcol+nrtest)
            outfit_lstm = fit_1surp(data, basedev, col+2nrtest, prevcol+2nrtest)

            push!(fit_results,[rep test "SRN"  outfit_srn])
            push!(fit_results,[rep test "GRU"  outfit_gru])
            push!(fit_results,[rep test "LSTM" outfit_lstm])

            col       += 1
            prevcol   += 1
            countdown -= 1
        end

        col     += 2nrtest
        prevcol += 2nrtest
    end
    print('\n')
end

#---------------------------------------------------------------------
# Self-paced reading
#---------------------------------------------------------------------
data_SPR = read_data("SPR")
print("Fitting LMMs to SPR data\n")

basedev_SPR = deviance(fit(LinearMixedModel, @formula(logRT ~ logRT_prev * log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos
                                                      + (1|subj_nr) + (0+logRT_prev|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                                      + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) + (1|item)), data_SPR))

fit_results = DataFrame(rep=Int64[], epoch=Int64[], rnntype=String[], chi2=Float64[], b_surp=Float64[], z_surp=Float64[], b_prevsurp=Float64[], z_prevsurp=Float64[], avsurp=Float64[])
fit_surps(data_SPR, basedev_SPR)

CSV.write("lmm_SPR.csv", fit_results; delim='\t')

#---------------------------------------------------------------------
# Eye tracking
#---------------------------------------------------------------------
data_ET = read_data("ET")
print("Fitting LMMs to ET data\n")

basedev_ET = deviance(fit(LinearMixedModel, @formula(logRTfirstpass ~ log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix
                                                     + (1|subj_nr) + (0+log_freq|subj_nr) + (0+log_freq_prev|subj_nr)
                                                     + (0+nr_char|subj_nr) + (0+nr_char_prev|subj_nr) + (0+word_pos|subj_nr) +(0+prevfix|subj_nr) + (1|item) + (0+prevfix|item)), data_ET))

#= To get a Gamma GLMM, do the following (Somehow, I can't get GLMMs to work with more than one effect per random variable)
using GLM
fit!(GeneralizedLinearMixedModel(@formula(RTfirstpass ~ log_freq * log_freq_prev * nr_char * nr_char_prev * word_pos * prevfix + (1|subj_nr) + (1|item)), data, Gamma(), IdentityLink()), fast=true)
=#
fit_results = DataFrame(rep=Int64[], epoch=Int64[], rnntype=String[], chi2=Float64[], b_surp=Float64[], z_surp=Float64[], b_prevsurp=Float64[], z_prevsurp=Float64[], avsurp=Float64[])
fit_surps(data_ET, basedev_ET)

CSV.write("lmm_ET.csv", fit_results; delim='\t')

#---------------------------------------------------------------------
# Electroencephalography
#---------------------------------------------------------------------
data_EEG = read_data("EEG")
print("Fitting LMMs to EEG data\n")

basedev_EEG = deviance(fit(LinearMixedModel, @formula(N400 ~ baseline + log_freq * nr_char * word_pos
                                                      + (1|subj_nr) + (0+baseline|subj_nr) + (0+log_freq|subj_nr) + (0+nr_char|subj_nr) + (0+word_pos|subj_nr)
                                                      + (1|item)), data_EEG))

fit_results = DataFrame(rep=Int64[], epoch=Int64[], rnntype=String[], chi2=Float64[], b_surp=Float64[], z_surp=Float64[], avsurp=Float64[])
fit_surps(data_EEG, basedev_EEG)

CSV.write("lmm_EEG.csv", fit_results; delim='\t')
