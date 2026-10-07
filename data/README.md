# Data

`raw/day.csv` is the daily file from the UCI Bike Sharing dataset. It contains daily rental counts for the Capital Bikeshare system in 2011–2012, plus calendar and weather fields.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894)
- Dataset author: Hadi Fanaee-T
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The project models `cnt`, the total daily count. It intentionally does not use same-day weather fields for future predictions because those values would not be known with certainty at planning time.

