# Run all analyses and make all plots

using CSV, MixedModels, DataFrames, Statistics
include("zscore.jl")
include("read_data.jl")

# Set constants
surp_col_name = :SRN_1000_0_surprisal    # Name of the first data frame column that contains surprisal values
nrrep  = 6                               # Number of networks trained per network type
nrtest = 9                               # Number of test points over incremental training

# Read subject and surprisal data, and add previous-word surprisal values
print("Reading subject data and surprisal files\n")
subjects  = CSV.read("subjects.csv", delim='\t')
surprisal = CSV.read("surprisal.csv",delim='\t')

surp_col     = findall(names(surprisal).==surp_col_name)[1]            # Find column index of first network's surprisal value
prevsurp_col = size(surprisal,2)+1                                     # Columns for previous-word surprisal will start here

for c = surp_col:size(surprisal,2)
    col_name = String(names(surprisal)[c])
    surprisal[Symbol(col_name*"_prev")] = [missing; surprisal[1:end-1,c]]
end

include("do_lmm.jl")                      # Fit LMMs to explain human data from RNN surprisal estimates
include("compare_mean_surp.jl")           # Pairwise comparisons between network types on surprisal averaged over repetitions
include("do_gam.jl")                      # Fit GAMs to explain LMM-fit from RNN language model accuracy
include("make_plots.jl")                  # Make scatterplots and GAM curve plots for each of the human data sets
