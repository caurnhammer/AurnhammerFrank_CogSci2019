This folder contains all data and code for running the analysis of [1].

DATA FILES
----------
* data_SPR.csv, data_ET.csv, data_EEG.csv are tab-separated text files containing the self-paced reading (SPR), eye-tracking (ET), and electroencephalography (EEG) data, respectively. SPR and ET data come from [2], EEG data from [3].
The first line contains column headers. All subsequent lines contain sentence-reading data, sorted by participant number and within each participant in the order of presentation during the experiment.
The following headers are shared by all three data files:
- subj_nr  : Participant ID-number. Note that the data sets use the same ID-numbers even though these are not the same participants.
- sent_nr  : Stimuli sentence ID-number. Identical sentences across the three data sets have identical ID-numbers.
- word     : Stimulus word as presented during experiment, with any punctuation attached.
- word_pos : Position of word in the sentence
- item     : Word-token ID-number; equals 100sent_nr+word_pos
- log_freq : Log-transformed frequency of word in the ENCOW corpus used for RNN training. Words with punctuation and clitics have no frequency because
             these were split into two tokens in the corpus.
- nr_char  : Number of characters in the word
- reject_data : Is "true" only if the current data point is excluded from analysis, because the RT is not between 50 and 3000 ms (for SPR and ET) or because
                there as an artefact (e.g. a blink) in the EEG signal.
- reject_word : Is "true" only if the current word is excluded from analysis, because it is a sentence-initial or -final word or because it is a clitic.

The following headers are unique for each data set:
- RT (in data_SPR) : Reading time on the word, in ms
- RT_firstpass (in data_ET) : First-pass reading time (gaze duration) on the word, in ms. Is 0 if the word was not fixated.
- prevfix (in data_ET)      : Is "true" only if the previous word was not fixated.
- N400 (in data_EEG)     : Size of the N400 ERP component in response to the word (see [3] for definition).
- baseline (in data_EEG) : Baseline level, i.e., average electrode potential in the 100 ms leading up to word onset.

* subjects.csv is a tab-separated text file containing participant information. Column headers are:
- experiment : Which of the three studies (SPR, ET, or EEG) the participant took part in.
- subj_nr    : Participant ID-number, corresponding to those in data_SPR/ET/EEG.csv
- age        : Participant age
- age_en     : Self-reported age at which the participant started learning English (0 for native speakers)
- gender     : Participant gender (f or m)
- hand       : Participant's dominant hand (r or l)
- correct    : Fraction of stimuli comprehension question that was answered correctly
- reject     : Is "true" only if the participant was excluded from analysis, because of being a non-native speaker or having a large error rate

* surprisal.csv is a tab-separated text file containing word-surprisal estimates from recurrent neural network (RNN) models.
The first three columns are sent_nr, word_pos, and item, as for data_SPR/ET/EEG.csv (see above). All subsequent columns contain surprisal values.
Each surprisal column header is [rnntype]_[train]_[rep]_surprisal; where [rnntype] denotes the RNN architecture (SRN, GRU, or GRU), [train] is the number of sentences trained on when the surprisal values were generated, and [rep] is the training repetition number (0 to 5).

ANALYSIS CODE
-------------
Analyses were run in Julia (v0.6.2) with the MixedModels package (v0.18.0).
- do_all.jl runs all the analyses and creates plots by calling do_lmm.jl, do_model_comparisons.jl, do_gam.jl, and make_plots.jl
- do_lmm.jl fits linear mixed-effects models to each of the three data sets using each set of surprisal values.
- compare_mean_surp.jl performs pairwise comparisons between fully trained RNN models' surprisal estimates, averaged over training repetitions.
- do_gam.jl fits the GAM curves to the output of do_lmm_SPR/ET/EEG, by calling R (package mcgv).
- make_plots.jl creates the paper's figure, by calling Python (package matplotlib).

ANALYSIS OUTPUT
---------------
* lmm_SPR/ET/EEG.csv are the outputs of do_lmm.jl. They are tab-separated text files with the following column headers
- rep     : Training repetition number (1 to 6)
- epoch   : Moment during RNN training at which surprisal values were generated (1 to 9, corresponding to 1K, 3K, 10K, 30K, 100K, 300K, 1M, 3M, and 6.47M sentences).
- rnntype : RNN architecture (SRN, GRU, or LSTM)
- chi2    : Decrease in regression model deviance due to including surprisal as a predictor.
- b       : Coefficient of surprisal predictor in fitted regression model
- b_prev  : Coefficient of previous-word surprisal predictor (SPR and ET only)
- z       : z-statistic for coefficient of surprisal predictor
- z_prev  : z-statistic for coefficient of previous-word surprisal predictor (SPR and ET only)
- avsurp  : Average surprisal over all items included in the analysis (i.e., weighted by the number of times a word token is included)

* gam_SPR/ET/EEG.csv are the outputs of do_gam.jl. They are tab-separated text files with the following column headers
- rnntype : RNN architecture (SRN, GRU, or LSTM)
- fit     : Fitted decrease in regression model deviance due to included surprisal as a predictor, with a negative sign for negative-going surprisal effect.
- se      : Standard Error of fit.
- x       : Negative average surprisals at which GAM fits (and SEs) are computed.

REFERENCES
----------
[1] Aurnhammer, C. & Frank, S.L. (2019). Comparing gated and simple recurrent neural network architectures as models of human sentence processing.
[2] Frank, S.L., Monsalve, I.F., Thompson, R.L., & Vigliocco, G. (2013). Reading-time data for evaluating broad-coverage models of English sentence processing. Behavior Research Methods, 45, 1182-1190.
[3] Frank, S.L., Otten, L.J., Galli, G., & Vigliocco, G. (2015). The ERP response to the amount of information conveyed by words in sentences. Brain and Language, 140, 1-11.
