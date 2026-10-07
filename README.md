# Height & weight estimator

A Streamlit frontend for exploring the relationship between weight and height in
`HeightWeight1.csv` and estimating height with a simple linear regression.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The interface assumes the dataset's measurements are in kilograms and centimeters,
based on their observed scale. Estimates outside the dataset's weight range are
marked as extrapolations; the model is for educational use and is not a medical tool.
