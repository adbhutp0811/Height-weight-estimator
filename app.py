from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().with_name("HeightWeight1.csv")
VALIDATION_SEED = 42
VALIDATION_FRACTION = 0.2

st.set_page_config(
    page_title="Height & weight",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    required_columns = {"Weight", "Height"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {', '.join(sorted(missing_columns))}."
        )

    data = data.loc[:, ["Weight", "Height"]].copy()
    for column in ("Weight", "Height"):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"The {column} column must contain numeric values.") from error

    values = data[["Weight", "Height"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The dataset contains missing or non-finite measurements.")
    if (values <= 0).any():
        raise ValueError("All height and weight measurements must be greater than zero.")
    if len(data) < 5:
        raise ValueError("At least five measurements are required to fit and validate the model.")

    return data


def fit_model(data):
    weights = data["Weight"].to_numpy(dtype=float)
    heights = data["Height"].to_numpy(dtype=float)

    shuffled_indices = np.random.RandomState(VALIDATION_SEED).permutation(len(data))
    test_count = int(np.ceil(len(data) * VALIDATION_FRACTION))
    test_indices = shuffled_indices[:test_count]
    train_indices = shuffled_indices[test_count:]

    validation_slope, validation_intercept = np.polyfit(
        weights[train_indices], heights[train_indices], deg=1
    )
    validation_predictions = (
        validation_slope * weights[test_indices] + validation_intercept
    )
    validation_errors = heights[test_indices] - validation_predictions
    mae = float(np.mean(np.abs(validation_errors)))
    total_variance = float(np.sum((heights[test_indices] - heights[test_indices].mean()) ** 2))
    r_squared = (
        1 - float(np.sum(validation_errors**2)) / total_variance
        if total_variance > 0
        else float("nan")
    )

    slope, intercept = np.polyfit(weights, heights, deg=1)
    return float(slope), float(intercept), mae, r_squared


data = load_data()
slope, intercept, validation_mae, validation_r_squared = fit_model(data)
min_weight = float(data["Weight"].min())
max_weight = float(data["Weight"].max())
mean_height = float(data["Height"].mean())

with st.sidebar:
    st.subheader("About this model")
    st.write(
        "A simple linear regression trained on the supplied height-and-weight dataset."
    )
    st.caption(
        "Measurements are shown as kg and cm, inferred from the scale of the dataset."
    )
    st.divider()
    st.write(f"**Training records**  \n{len(data):,}")
    st.write(f"**Observed weight range**  \n{min_weight:.1f}–{max_weight:.1f} kg")
    st.caption(
        "This is an educational estimate. Height cannot be reliably inferred from weight alone."
    )

st.caption("HEIGHT & WEIGHT  /  LINEAR REGRESSION")
st.title("A quick estimate, grounded in your data")
st.write(
    "Explore the relationship between weight and height, then see what the model estimates "
    "for a given weight."
)

input_column, result_column = st.columns([0.85, 1.15], gap="large")
with input_column:
    with st.container(border=True):
        st.subheader("Try an estimate")
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.1,
            max_value=300.0,
            value=round(float(data["Weight"].median()), 1),
            step=0.5,
            help="For the most meaningful estimate, enter a weight within the observed data range.",
        )
        st.caption(f"Model trained on weights from {min_weight:.1f} to {max_weight:.1f} kg.")

with result_column:
    estimated_height = slope * weight + intercept
    with st.container(border=True):
        st.subheader("Estimated height")
        st.metric(
            "Height",
            f"{estimated_height:.1f} cm",
            delta=f"{estimated_height - mean_height:+.1f} cm",
            delta_color="off",
            help="Difference from the dataset's average recorded height.",
        )
        st.caption(
            "Compared with the dataset average. The estimate uses the best-fit line across all records."
        )

if weight < min_weight or weight > max_weight:
    st.warning(
        "This weight is outside the range represented in the dataset, so the estimate is an extrapolation."
    )

with st.container(horizontal=True):
    st.metric("Measurements", f"{len(data):,}", border=True)
    st.metric(
        "Validation MAE",
        f"{validation_mae:.2f} cm",
        help="Mean absolute error on a held-out 20% sample. Lower is better.",
        border=True,
    )
    st.metric(
        "Validation R²",
        f"{validation_r_squared:.3f}",
        help="R² on the held-out 20% sample. Closer to 1 indicates a stronger fit.",
        border=True,
    )

st.subheader("The relationship in your dataset")
st.write(
    "Each dot is a recorded measurement. The line shows the model's average trend."
)

display_data = data.sample(n=min(len(data), 3500), random_state=VALIDATION_SEED)
points = (
    alt.Chart(display_data)
    .mark_circle(opacity=0.2, size=22, color="#5B7FA6")
    .encode(
        x=alt.X("Weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)),
        y=alt.Y("Height:Q", title="Height (cm)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Weight:Q", title="Weight (kg)", format=".1f"),
            alt.Tooltip("Height:Q", title="Height (cm)", format=".1f"),
        ],
    )
)

line_data = pd.DataFrame(
    {
        "Weight": np.linspace(min_weight, max_weight, num=100),
    }
)
line_data["Height"] = slope * line_data["Weight"] + intercept
trend_line = (
    alt.Chart(line_data)
    .mark_line(color="#243E70", strokeWidth=3)
    .encode(
        x=alt.X("Weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)),
        y=alt.Y("Height:Q", title="Height (cm)", scale=alt.Scale(zero=False)),
    )
)

with st.container(border=True):
    st.altair_chart(
        (points + trend_line).properties(height=360).interactive(),
        alt="Recorded weights and heights with a fitted linear regression trend line",
        width="stretch",
    )
    st.caption(
        f"Trend line: height = {slope:.2f} × weight {intercept:+.2f}. "
        "The chart displays a representative sample; the model uses every record."
    )

with st.expander("View sample records"):
    st.dataframe(
        data.head(10).round(2),
        hide_index=True,
        alt="First ten height and weight records from the dataset",
    )
