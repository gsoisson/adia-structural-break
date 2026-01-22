~~~text
preprocess_*.parquet  — Long-format (id,time) table used by all feature blocks.

Columns:
- period               : {0,1} (0=before, 1=after)
- original             : raw value
- standardized         : robust z-score using BEFORE median/MAD
- clipped              : winsorized standardized (cutoffs from BEFORE)
- detrended            : detrended & mean-centered version of `clipped`
- diff_standardized    : first difference of `standardized`
- diff_detrended       : detrended & mean-centered version of `diff_standardized`
- absdiff_detrended    : |first difference| → normalized + detrended/mean-centered
- absval_detrended     : |standardized| → normalized + detrended/mean-centered
- squared_detrended    : (standardized)^2 → winsorized + detrended/mean-centered
- diff2_standardized   : second difference (curvature) of `standardized`
- diff2_detrended      : detrended & mean-centered version of `diff2_standardized`

Usage guide:
- moments                   → uses `original`, `standardized`, `clipped`
- quantiles/tests           → uses `standardized`
- rates                     → uses `original`, `standardized`
- autocorrelation/AR        → uses `detrended`
- frequency                 → uses `standardized` and `detrended`
- differences               → uses `diff_standardized`, `diff_detrended`, `absdiff_detrended`
- curvature                 → uses `diff_standardized`, `diff2_standardized`, `diff2_detrended`
- absolute/squared          → uses `absval_detrended`, `squared_detrended`
- boundary/boundary_edge    → uses `standardized`, `clipped`, `detrended`, `diff_standardized`, `diff2_standardized`, `diff2_detrended`
- rolling                   → uses `clipped`, `diff_detrended`
~~~