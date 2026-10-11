---
title: "Detecting faults in a chemical plant without examples of faults"
geometry: "margin=1.5cm"
fontsize: 10pt
header-includes:
  - \usepackage{graphicx}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
  - \interlinepenalty=10000
  - \makeatletter\renewcommand{\maketitle}{\begin{center}\large\@title\end{center}\vspace{0.5em}}\makeatother
---

## 1. The plant and the task

The Tennessee Eastman process (TEP) is a simulated chemical plant published by Eastman Chemical Company as a benchmark for process control and monitoring (Downs and Vogel, 1993). Reactants A, C, D and E and an inert B are fed to a reactor, where they form products G and H and a by-product F. The reactor effluent is cooled in a condenser and split in a vapour-liquid separator. The vapour returns to the reactor through a compressor, with part of it purged, and the liquid goes to a stripper that removes the remaining reactants. A plant-wide control system holds the plant at its operating point. The data have 52 channels sampled every 3 minutes: 41 measurements of flows, temperatures, pressures, levels and compositions (xmeas_1 to xmeas_41) and 11 manipulated variables (xmv_1 to xmv_11).

Faults are rare and a new fault is often of a type not recorded before, so a classifier trained on past faults is of limited use. We build two detectors that learn only normal operation and raise an alarm when the plant departs from it. Both are trained on fault-free runs 1–300; runs 301–400 set thresholds, and runs 401–500 measure false-alarm rates. Each run contains 500 samples over 25 hours. We test faults IDV 1–20 using runs 1–20 of each fault. Samples 1–20 are normal and the first faulty observation is sample 21 (Rieth et al., 2017). We report the share of each faulty period in alarm, the time to the first alarm, and the channels that drive the alarm.

## 2. The two detectors

Each channel is standardised with its mean and standard deviation (ddof=1) over training runs 1–300. Preprocessing and model parameters are fitted only on these runs. Each threshold is numpy.quantile(validation scores, 0.99), using default linear interpolation over runs 301–400. A sample is in alarm when it and the previous two samples all strictly exceed the threshold within the same run.

PCA monitor. PCA finds the orthogonal directions of largest variance in the standardised training data, the eigenvectors of its correlation matrix. We keep the smallest number of components whose eigenvalues sum to at least 90% of the total: k=31 (90.2%; 30 components give 88.9%). For a standardised sample $z$, retained loading vectors $P$ and eigenvalues $\lambda_i$, $t=P^\top z$, $T^2=\sum_{i=1}^{k}t_i^2/\lambda_i$ and $\mathrm{SPE}=\lVert z-PP^\top z\rVert^2$. $T^2$ measures distance within the retained subspace, scaled by each direction's variance. SPE measures the part that the retained directions cannot reproduce. The thresholds are 51.7971 for $T^2$ and 11.7745 for SPE.

Forecast monitor. A single Ridge(alpha=1.0), fitted on 149,400 training rows, predicts all 52 standardised channels at sample $t$ from the 104 values at $t-1$ and $t-2$ within the same run. The first two samples of every run have no forecast score. Each residual $r_j$ is divided by its training residual standard deviation $s_j$ (ddof=1), and the score is $\sum_j(r_j/s_j)^2$. Each squared scaled residual has training mean approximately one, so a normal total score is close to 52 (52.1 on validation); the threshold is 112.4423. For predictable channels, small changes can produce large scaled residuals: the training residual standard deviation for stripper temperature xmeas_18 is only 0.044 in standardised channel units.

\begin{minipage}{\linewidth}
\centering
\includegraphics[width=0.53\linewidth]{figures/detectors_fault4_run1.png}
\par\smallskip
{\small Figure 1: $T^2$, SPE and ridge scores for IDV(4), run 1, with thresholds (dashed) and fault onset (dotted).\par}
\end{minipage}

Figure 1 shows a step in reactor cooling-water inlet temperature. $T^2$ reaches 209 at sample 21, then varies around its threshold (post-fault median 47; 32% of samples above threshold). SPE and ridge exceed their thresholds on 99.6% and 99.2% of post-fault samples. These are single-score exceedance rates; the detection rates below apply the three-consecutive-sample alarm rule.

\newpage

## 3. Results

**Table 1.** Detection rates (%) and median delays (minutes). Detection rate is the fraction of samples 21–500 in alarm for each faulty run, averaged over 20 runs. Delay is $3\times(\text{first post-fault alarm sample}-20)$; its median includes only runs that alarmed. Parentheses show missed-run counts; omitted counts are zero. A dash indicates an unavailable or inapplicable delay. Row 0 gives false-alarm rates over scored normal-test samples. Extra precision is retained for small nonzero rates.

| IDV | fault (Chiang et al., 2000) | rate T² | rate SPE | rate ridge | delay T² | delay SPE | delay ridge |
|---:|:-----------------------------------|------:|------:|------:|-------:|-------:|-------:|
| 0 | none: false-alarm rate, test runs | 0.052 | 0.00 | 0.00 | – | – | – |
| 1 | A/C feed ratio, step | 98.5 | 99.2 | 99.4 | 25.5 | 15 | 12 |
| 2 | B composition, step | 97.3 | 95.9 | 96.5 | 39 | 57 | 51 |
| 3 | D feed temperature, step | 0.0208 | 0.0 | 0.0 | 231 (19) | – (20) | – (20) |
| 4 | reactor cooling water inlet temp., step | 9.1 | 99.4 | 96.9 | 64.5 | 12 | 9 |
| 5 | condenser cooling water inlet temp., step | 33.2 | 12.5 | 99.6 | 9 | 15 | 9 |
| 6 | A feed loss, step | 97.9 | 99.6 | 99.6 | 33 | 9 | 9 |
| 7 | C header pressure loss, step | 99.6 | 99.6 | 99.6 | 9 | 9 | 9 |
| 8 | A, B, C feed composition, random | 94.2 | 82.2 | 95.4 | 70.5 | 67.5 | 52.5 |
| 9 | D feed temperature, random | 0.0104 | 0.0 | 0.0 | 231 (19) | – (20) | – (20) |
| 10 | C feed temperature, random | 8.5 | 17.7 | 69.8 | 471 | 177 | 67.5 |
| 11 | reactor cooling water inlet temp., random | 25.8 | 43.4 | 45.4 | 43.5 | 40.5 | 33 |
| 12 | condenser cooling water inlet temp., random | 94.4 | 86.7 | 95.7 | 37.5 | 48 | 30 |
| 13 | reaction kinetics, slow drift | 88.4 | 89.3 | 90.2 | 159 | 136.5 | 138 |
| 14 | reactor cooling water valve, sticking | 97.8 | 78.4 | 99.5 | 12 | 13.5 | 9 |
| 15 | condenser cooling water valve, sticking | 0.0208 | 0.0 | 0.0 | 231 (19) | – (20) | – (20) |
| 16 | unknown | 1.9 | 10.1 | 71.2 | 690 (3) | 156 | 45 |
| 17 | unknown | 66.6 | 86.8 | 86.7 | 138 | 118.5 | 118.5 |
| 18 | unknown | 87.1 | 88.4 | 89.3 | 175.5 | 151.5 | 141 |
| 19 | unknown | 1.0 | 0.3 | 2.9 | 471 | 610.5 (6) | 121.5 |
| 20 | unknown | 13.6 | 43.2 | 63.5 | 186 | 165 | 151.5 |

About 1% of individual normal-test scores exceed each threshold (1.09%, 1.01% and 1.00% for $T^2$, SPE and ridge). After the persistence rule, false-alarm rates are 0.052%, 0% and 0%. Excluding IDV 3, 9 and 15, mean detection rates across the remaining 17 faults are 59.7%, 66.6% and 82.4%. Ridge is within 0.1 percentage points of the highest rate on 15 of these faults (all except IDV 2 and 4), and has the shortest or equal shortest median delay on 15 (all except IDV 2 and 13).

## 4. Comparison

IDV 5: compensation and changed relationships. After the condenser cooling-water inlet-temperature step, PCA alarm rates decline. Over samples 201–500, $T^2$ and SPE alarm on 0.93% and 0.42% of samples, while ridge alarms on 100%. In this interval, condenser cooling-water flow xmv_11 remains about 1.9 training standard deviations above normal. In training, xmv_11 and stripper underflow xmeas_17 correlate at -0.999, and the ridge prediction of A feed xmeas_1 gives both channels weights of about 3.2 at lag 2 and 2.5 at lag 1. Their effects largely cancel under normal conditions. The changed relationship disrupts this cancellation, producing an average absolute A-feed prediction residual of about 82 residual standard deviations in the late interval. For PCA, the residual component of the full post-fault mean shift gives an SPE of 1.76, below 11.8. Russell et al. (2000) likewise report that PCA-based residual statistics can fall below threshold while this fault persists.

\newpage

IDV 10 and 16: temporal variation. Ridge detects 69.8% and 71.2% of post-fault samples, compared with SPE's 17.7% and 10.1%. In IDV 16, within-run three-minute changes in stripper temperature xmeas_18 and steam flow xmeas_19 have about 2.5 times their normal-test standard deviation. Their training residual standard deviations are only 0.044 and 0.12 in standardised channel units, making ridge sensitive to these changes. Static PCA uses the current observation without explicitly modelling the previous samples.

IDV 4: a broken correlation. All three statistics alarm in every run, but SPE and ridge have much higher sustained coverage (99.4% and 96.9%) than $T^2$ (9.1%). Reactor cooling-water flow xmv_10 rises by about 7 standard deviations while reactor temperature xmeas_9, correlated with it at 0.63 in training, stays near normal. Only 47% of the squared mean shift lies in the retained directions. The mean shift gives $T^2=18.6$, below its threshold, and SPE=25.5, above it. This agrees qualitatively with the greater residual-space sensitivity reported for this fault by Russell et al. (2000).

IDV 2: where $T^2$ performs best. The B-composition step moves the plant mainly along the retained directions: 99% of the squared mean shift lies in that subspace. $T^2$ has the highest detection rate (97.3%) and shortest median delay (39 minutes). Ridge is therefore not uniformly superior.

## 5. Faults nobody catches

All three statistics effectively miss IDV 3, 9 and 15. SPE and ridge never alarm after onset in any of their 20 runs per fault. $T^2$ alarms in only one run per fault, with detection rates of 0.0104–0.0208%, below its normal-test false-alarm rate of 0.052%. These isolated alarms do not demonstrate useful fault sensitivity.

Across the post-fault samples, channel means differ from normal-test means by at most about 0.24 training standard deviations, and channel standard deviations differ by less than 14%. IDV 3 and 9 disturb the unmeasured D-feed temperature, while IDV 15 produces no clear response in the recorded channels. [Russell, Chiang and Braatz (2000)](https://web.mit.edu/braatzgroup/36_Fault_detection_in_industrial_processes_using_canonical_variate_analysis_and_dynamic_principal_component_analysis.pdf) similarly reported high missed-detection rates and no visually observable mean or variance changes for these faults, excluding them from their comparison. Our results agree qualitatively, although the evaluation protocols differ. The weak observable departures are difficult for the specified monitors to detect; they do not prove that every alternative model must fail. Additional measurements could provide more direct evidence.

## 6. Diagnosis

SPE and ridge scores split into per-channel contributions: $(z_j-(PP^\top z)_j)^2$ for SPE and $(r_j/s_j)^2$ for ridge. For each fault and detector, we average contributions over all post-fault alarm samples pooled across the 20 runs, retaining the top five in results/contributions.csv. Figure 2 shows each channel's share of the full 52-channel mean statistic. Faults 3, 9 and 15 have no contribution rows because SPE and ridge produce no post-fault alarms.

\begin{minipage}{\linewidth}
\centering
\includegraphics[width=0.55\linewidth]{figures/contributions.png}
\par\smallskip
{\small Figure 2: Top five channels for IDV 4, 6 and 10, as shares of the full mean statistic over post-fault alarm samples; stream numbers in brackets.\par}
\end{minipage}

\newpage

IDV 4, reactor cooling-water inlet temperature, step. Both detectors identify the cooling loop. Over SPE alarm samples, cooling-water flow xmv_10 averages 6.95 training standard deviations above normal, while reactor temperature xmeas_9 averages -0.025. Ridge assigns 68% to xmv_10; SPE assigns 44% to xmv_10 and 38% to xmeas_9. Their positive training correlation causes PCA to reconstruct a higher reactor temperature when cooling flow rises. A residual therefore appears on xmeas_9 despite its near-normal observed value. This illustrates contribution smearing (Westerhuis et al., 2000).

IDV 6, A feed loss, step. The A-feed measurement xmeas_1 falls to zero while its valve xmv_3 opens fully (post-fault median 100%, versus 24.6% normally). Before shutdown, these channels contribute about 58% and 38% of ridge, respectively. In all 20 runs, pressure reaches approximately 3,000 kPa and measurements xmeas_1–22 freeze 5.70–6.05 hours after onset, consistent with the simulator's shutdown condition. These samples form 76% of SPE alarm samples. Consequently, pooled SPE ranks recycle valve xmv_5 first (38%), reflecting the stopped simulation rather than the initiating feed fault. Ridge retains xmeas_1 as its top contributor. IDV 18 exhibits a frozen shutdown segment in 15 of 20 runs.

IDV 10, C-feed temperature, random variation. The stream-4 inlet temperature is not recorded directly. Both detectors rank stripper temperature xmeas_18 first (29% of SPE and 71% of ridge), consistent with a downstream response to the disturbance. Its post-fault standard deviation is 2.67 times training, its mean shifts by only 0.009 training standard deviations, and its within-run three-minute changes have 2.10 times the training standard deviation. Ridge alarms on 6,697 samples, versus 1,695 for SPE.

Across the 17 faults with SPE/ridge alarms, the detectors agree on the top channel for six. Ridge ranks A feed xmeas_1 first for seven faults (1, 5, 6, 8, 12, 13, 18), although only IDV 6 is an A-feed fault. Contributions identify where an alarm appears, which is not necessarily where the fault originated.

## 7. Limits

- All runs come from one simulator, control structure and operating point. Real plants experience changing throughput, equipment condition, sensor drift, missing data and maintenance. Fixed thresholds may not remain valid, and zero observed false alarms do not guarantee zero future false alarms.
- Each run contains one programmed fault with a known onset and lasts 25 hours. Overlapping, unlisted and slower faults are not tested. Twenty runs per fault provide limited evidence about generalisation.
- Both detectors are linear. PCA is static and the forecaster uses only two lags. Contributions show where deviations appear, not necessarily their root cause; feedback and frozen shutdown data can dominate them. Detection delays describe only runs that alarmed and must be read alongside missed-run counts.

## AI use

We used generative AI to help write the code, and to draft the report. All detection numbers come from the files in results.

\newpage

## Appendix: contributions

- **Pingfan Bu (Andrew ID: pbu):** Developed the forecast-residual monitor and shared data-loading and standardisation code. Generated ridge scores and monitoring plots, contributed the forecast-monitor explanation and residual analysis, and prepared the initial combined report and evidence PDF.

- **Justin Huh (Andrew ID: junwooh):** Developed and validated a PCA implementation, including component selection and T²/SPE calculations. Implemented detector evaluation, including false-alarm rates, detection rates, delays and missed-run counts. Revised and consolidated the final report, including the detector explanations, evaluation methodology, fault comparisons, diagnosis discussion and visualisations.

- **Wenwen Wang (Andrew ID: wenwenw):** Developed the channel-contribution analysis for SPE and ridge, generated the top-five contribution table and comparison plots, and contributed the diagnosis section. Interpreted the alarm-driving channels against the selected faults' physical descriptions and explained the distinction between an alarm's location and its initiating cause.

- **Bolan Zheng (Andrew ID: bolanz):** Developed and validated a PCA implementation, including component selection and T²/SPE calculations, and integrated it into the shared pipeline. Implemented detector evaluation, including thresholds, persistent alarms, false-alarm rates, detection rates, delays and missed-run counts, providing the quantitative results for the report's Results and Comparison sections.
