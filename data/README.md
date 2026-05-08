# Data Policy and Schema

The measured monitoring dataset used in the manuscript is not distributed in
this repository. It contains project-specific structural monitoring records and
is controlled by the data owner.

The code expects a private aligned table with:

- a `datetime` column;
- crack response columns named `crack_*`;
- rebar-stress response columns named `rebar_*`;
- optional thermal, hydraulic, rainfall, and other environmental drivers.

The file `sample/sample_monitoring.csv` is a tiny toy schema example. It is not
research data, is not used in the manuscript, and should not be used to compare
model performance.

