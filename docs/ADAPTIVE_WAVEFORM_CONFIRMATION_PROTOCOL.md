# Development confirmation after the first adaptive screen

Fixed before confirmation fitting, 2026-11-08. The initial 21-model screen found
better aligned shape/derivatives from hybrid morphology training, but little
native-phase improvement and weak raw-RGB-only candidates. It is an exploratory,
single-seed, six-real-subject selection result. Confirmation remains development.

Use all 26 original UBFC Train subjects in the same five subject folds as the
linear baseline. Fit on the other folds and all eight **original SCAMPS Train**
examples. The two synthetic development clips from the first screen are now
reused for fitting in this real-source cross-validation study; no new synthetic
validation/generalization claim is made. No official Val/Test group changes.

Eight hypotheses per fold, 300 fixed local-GPU optimizer steps each: residual
morphology with 0/20/50% synthetic sampling; native loss at 20%; morphology with
past-only context; a wider 32-channel morphology model; spectral warmup followed
by morphology; and the dual-band morphology model. All other optimizer/input
contracts match the first screen. The alternative curriculum starts with spectral
distance and introduces lag-tolerant waveform/derivative terms after one quarter
of the budget; it does not end by forcing unverified native contact phase.

Compare fixed CHROM_WIN and fit-polarity, input-quality hybrids on the same folds.
Also test predeclared mean/median ensembles of the base, context and dual-band
morphology outputs, standardized using each output alone. No reference-selected
alignment, polarity correction, member selection or ensemble weights.

After all 40 fits, repeat the initial screen's residual/dual-band morphology
conditions on its original split with seeds 20261125 and 20261147 at 120 steps.
Together with the original seed this checks initialization sensitivity. It is
not a substitute for more participants or a third-domain evaluation.

Time cap: 15 minutes for this local compact confirmation campaign. Missing
configurations/folds are recorded, and no incomplete-cohort method is called
the winner. Fixed final steps only; no validation-best checkpoint selection.
Report native and aligned shape/derivative metrics, HR error, all failures and
initialization results separately. Do not call the resulting development leader
the final best model or promote it into physiological timing/BP without evidence.
