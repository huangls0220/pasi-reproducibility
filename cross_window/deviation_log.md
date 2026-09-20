# R73 setup deviation before formal execution

The first preparation used the six preregistered input windows and produced
restricted caches and episodes. Its smoke test did not pass: the draft runner
called the historical E20 `build_config`, which gave the frozen-point profile
`D_bar=10` and `budget_ratio=0.7`, whereas the R56 formal comparison fixed both
profiles at `D_bar=50` and `budget_ratio=3.5`. The draft runner rejected the
frozen-point cell before simulation. One fixed-envelope stable cell ran and
reported assignment coverage 0.514396887159533 and zero QoS violations. No
formal matrix ran, and no window, seed, scenario, envelope, or outcome metric
was selected or changed on the basis of this result.

The runner now loads the archived R56 configurations directly and changes only
the legally reconstructed episode path and its observed provider count. The
draft restricted output is retained outside the distributable package for
audit, not used in the formal analysis. A new restricted output directory and
new frozen runner hash are required before execution.

The preregistration calls the six chosen months "consecutive qualifying
calendar months." That description is imprecise: the frozen list is October,
November, December, January, March, and April; February was not selected
because the original comparison corpus already used February 14--21. The six
explicit date pairs, not that phrase, define the sample. None was replaced
after preprocessing or outcome inspection. This correction does not turn the
convenience-selected windows into a random population sample.
