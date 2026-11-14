# Voice Arena — video walkthrough script

**Target runtime:** about 1 minute 45 seconds. Present `Voice_Arena_Q1_Slides.pdf` full screen and record the slides with your voice. The script follows the six slides in that PDF.

## Slide 1 — The question · 0:00–0:10

“Hi, I’m Shaurya Tomer. I designed Voice Arena as a way to choose a speaker diarization system for the audio you actually have—not just pick the model with the smallest number on one benchmark.”

## Slide 2 — Who uses it · 0:10–0:27

“Engineers need to know what will work on their recordings. Researchers want to know whether a method beats a baseline. Product and privacy teams care about cost, deployment, and where the audio goes. So the leaderboard shows overlap, missed speakers, speed, cost, and privacy alongside accuracy.”

## Slide 3 — How I would evaluate systems · 0:27–0:48

“For the full benchmark, I propose about one hundred hours across meetings, in-the-wild speech, difficult domains, and clean controls. I would keep speakers and sessions separate in the hidden test set. References would be annotated from audio, with overlap retained, at least twenty percent double-labelled and adjudicated, and difficult sections audited. Every system gets the same mono waveform; the primary track gets no reference speaker count.”

## Slide 4 — Reading the leaderboard · 0:48–1:12

“This is the pilot view, not the full hundred-hour benchmark. DER uses a quarter-second collar and includes overlap. The two pyannote systems scored 14.3 and 14.8 percent, with confidence intervals that overlap. The table also shows JER, overlap DER, speaker-count error, speed, and cost. The lower rows are sanity-check baselines, not competing products. My paired difference says no clear winner.”

## Slide 5 — What the pilot taught me · 1:12–1:32

“The pilot covers three AMI meetings—about fifty-seven minutes. On TS3003a, pyannote 3.1 found only two of four speakers: DER was 15.6 percent, while JER was 63 percent. Overall overlap DER was around 36 percent, much worse than overall DER. The collar also changed community-1 from 18.4 percent without it to 14.3 with it. These are useful warnings, but three correlated meetings cannot establish a broad ranking; NeMo has no score because it did not run successfully here.”

## Slide 6 — Reflection and next steps · 1:32–1:52

“AMI influenced me most because using its pilot data showed how annotation conversion and meeting-to-meeting variation affect the result. DIHARD III pushed me to report domain slices. I’m least sure about the quarter-second collar and the one-point practical threshold; I’d calibrate both against annotator disagreement and product needs. A single DER can hide a missed speaker, so I pair it with JER and count error. Next I’d build the audited reference set and run all twelve candidates.”

## Recording notes

- The scorecard’s two model rows and two baselines are real pilot results; the mockup is labeled as a pilot view. The proposed ~100-hour benchmark has not yet been run.
- Keep the baseline distinction in the narration: the grey/lower rows are sanity checks, not evaluated vendor systems.
- Rehearse once with a timer. If you exceed two minutes, shorten the system-user examples on slide 2 and the caveats on slide 5; keep the pilot result and your reflection.
- Record in Loom or another screen recorder, upload as an unlisted link, then verify it opens in a private browser window.
