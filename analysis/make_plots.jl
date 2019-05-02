ENV["MPLBACKEND"] = "qt4agg"   # To fix a bug when connecting to Anaconda Python
using PyPlot, CSV

function make_scatterplot(lmm_results)
    fit_srn  = lmm_results[lmm_results[:rnntype].=="SRN", :]
    fit_gru  = lmm_results[lmm_results[:rnntype].=="GRU", :]
    fit_lstm = lmm_results[lmm_results[:rnntype].=="LSTM",:]

    scatter(-fit_srn[:avsurp], fit_srn[:chi2] .*sign.(fit_srn[:b_surp]),  s=10, c=:blue,  marker="x", label="SRN")
    scatter(-fit_gru[:avsurp], fit_gru[:chi2] .*sign.(fit_gru[:b_surp]),  s=10, c=:red,   marker="o", label="GRU")
    scatter(-fit_lstm[:avsurp],fit_lstm[:chi2].*sign.(fit_lstm[:b_surp]), s=10, c=:green, marker="^", label="LSTM")

    grid("on")
end

function make_gamplot(gam_curve)
    gam_srn  = gam_curve[gam_curve[:rnntype].=="SRN", :]
    gam_gru  = gam_curve[gam_curve[:rnntype].=="GRU", :]
    gam_lstm = gam_curve[gam_curve[:rnntype].=="LSTM", :]

    plot(gam_srn[:x],  gam_srn[:fit],  "b-",  label="SRN")
    plot(gam_gru[:x],  gam_gru[:fit],  "r:",  label="GRU")
    plot(gam_lstm[:x], gam_lstm[:fit], "g--", label="LSTM")

    fill_between(gam_srn[:x],  gam_srn[:fit] -2gam_srn[:se],  gam_srn[:fit] +2gam_srn[:se],  color=:blue,  alpha=0.3, linewidth=0)
    fill_between(gam_gru[:x],  gam_gru[:fit] -2gam_gru[:se],  gam_gru[:fit] +2gam_gru[:se],  color=:red,   alpha=0.3, linewidth=0)
    fill_between(gam_lstm[:x], gam_lstm[:fit]-2gam_lstm[:se], gam_lstm[:fit]+2gam_lstm[:se], color=:green, alpha=0.3, linewidth=0)

    grid("on")
end

#-------------------------------------------------------------------
lmm_SPR = CSV.read("lmm_SPR.csv", delim='\t')
lmm_ET  = CSV.read("lmm_ET.csv",  delim='\t')
lmm_EEG = CSV.read("lmm_EEG.csv", delim='\t')
gam_SPR = CSV.read("gam_SPR.csv", delim='\t')
gam_ET  = CSV.read("gam_ET.csv",  delim='\t')
gam_EEG = CSV.read("gam_EEG.csv", delim='\t')

lmm_EEG[:chi2] = -lmm_EEG[:chi2]
gam_EEG[:fit]  = -gam_EEG[:fit]

figure()

subplot(231)
make_scatterplot(lmm_SPR)
ylabel("Goodness-of-fit")
title("Self-paced reading time")

subplot(232)
make_scatterplot(lmm_ET)
title("Gaze duration")
legend(loc="upper left")

subplot(233)
make_scatterplot(lmm_EEG)
title("N400 size")
#legend(loc="upper right")

subplot(234)
make_gamplot(gam_SPR)
xlabel("Weighted average log-prob")
ylabel("Goodness-of-fit")

subplot(235)
make_gamplot(gam_ET)
xlabel("Weighted average log-prob")
legend(loc="upper left")

subplot(236)
make_gamplot(gam_EEG)
xlabel("Weighted average log-prob")
#legend(loc="upper right")
