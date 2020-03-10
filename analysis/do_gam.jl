#=---------------------------------------------------------------
Analysis Stage 2: For each RNN type, fit a GAM to predict
the log-likelihood ratio (and effect direction) from the weighted
average surprisals. Create data frames with all values for plotting.
---------------------------------------------------------------=#

using RCall
print("Fitting GAMs")

function do_gam(lmm_rnn)
    # Fit GAM in R, generate a plot (without actually plotting it), and return plot values

    model = R"""m = gam(effect ~ s(logprob)+s(logprob,rep,bs="re"), data=$lmm_rnn)"""
    plot  = R"plot($model, select=0)"

    intercept = model[:coefficients][1]
    fit = convert(Array{Float32,1}, plot[1][:fit])
    se  = convert(Array{Float32,1}, plot[1][:se])
    x   = convert(Array{Float32,1}, plot[1][:x])
    (intercept, fit, se, x)
end

function get_gam_curve(lmm_results)
    lmm_results[!,:effect]  = lmm_results[!,:chi2] .* sign.(lmm_results[!,:b_surp])                  # reverse effect direction if coefficient negative. What if b_surp and b_prevsurp of unequal sign?!
    lmm_results[!,:logprob] = -lmm_results[!,:avsurp]

    lmm_srn  = lmm_results[lmm_results[!,:rnntype].=="SRN",:]
    lmm_gru  = lmm_results[lmm_results[!,:rnntype].=="GRU",:]
    lmm_lstm = lmm_results[lmm_results[!,:rnntype].=="LSTM",:]

    (intercept_srn,  fit_srn,  se_srn,  x_srn)  = do_gam(lmm_srn)
    (intercept_gru,  fit_gru,  se_gru,  x_gru)  = do_gam(lmm_gru)
    (intercept_lstm, fit_lstm, se_lstm, x_lstm) = do_gam(lmm_lstm)
 
    gam_curve_srn  = DataFrame(rnntype="SRN",  fit=intercept_srn .+ fit_srn,  se=se_srn,  x=x_srn)
    gam_curve_gru  = DataFrame(rnntype="GRU",  fit=intercept_gru .+ fit_gru,  se=se_gru,  x=x_gru)
    gam_curve_lstm = DataFrame(rnntype="LSTM", fit=intercept_lstm .+ fit_lstm, se=se_lstm, x=x_lstm)

    [gam_curve_srn; gam_curve_gru; gam_curve_lstm]
end

R"library(mgcv)"

gam_curve_SPR = get_gam_curve(CSV.read("lmm_SPR.csv", delim='\t'))
CSV.write("gam_SPR.csv", gam_curve_SPR; delim='\t')

gam_curve_ET = get_gam_curve(CSV.read("lmm_ET.csv", delim='\t'))
CSV.write("gam_ET.csv", gam_curve_ET; delim='\t')

gam_curve_EEG = get_gam_curve(CSV.read("lmm_EEG.csv", delim='\t'))
CSV.write("gam_EEG.csv", gam_curve_EEG; delim='\t')
