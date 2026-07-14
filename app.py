from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Nassau Factory Optimizer",
    page_icon="🍬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nassau_candy_distributor.csv"
CONFIG_PATH = BASE_DIR / "factory_config.json"

REGION_CENTROIDS = {
    "Atlantic": (39.0, -75.0),
    "Interior": (39.5, -90.0),
    "Pacific": (37.5, -120.0),
    "Gulf": (31.0, -90.0),
    "Northeast": (42.5, -72.0),
    "Southeast": (33.0, -83.0),
    "Central": (39.0, -97.0),
    "West": (40.0, -112.0),
}

SHIP_MODE_FACTOR = {
    "Same Day": 0.72,
    "First Class": 0.82,
    "Second Class": 0.92,
    "Standard Class": 1.00,
}

@st.cache_data
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Lead Time Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin %"] = np.where(
        df["Sales"].ne(0), (df["Gross Profit"] / df["Sales"]) * 100, 0
    )
    config = load_config()
    df["Current Factory"] = df["Product Name"].map(config["product_factory"])
    df = df.dropna(subset=["Lead Time Days", "Current Factory", "Region", "Ship Mode"])
    q1, q99 = df["Lead Time Days"].quantile([0.01, 0.99])
    df = df[df["Lead Time Days"].between(q1, q99)].copy()
    return df

def haversine(lat1, lon1, lat2, lon2):
    radius = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda/2)**2
    return 2 * radius * math.asin(math.sqrt(a))

def region_coords(region):
    if region in REGION_CENTROIDS:
        return REGION_CENTROIDS[region]
    return (39.5, -98.35)

def add_distance(df):
    config = load_config()
    factories = config["factories"]
    distances = []
    for _, row in df.iterrows():
        factory = factories[row["Current Factory"]]
        rlat, rlon = region_coords(row["Region"])
        distances.append(
            haversine(factory["latitude"], factory["longitude"], rlat, rlon)
        )
    result = df.copy()
    result["Estimated Distance Miles"] = distances
    return result

@st.cache_resource
def train_models(df):
    data = add_distance(df)
    features = [
        "Product Name", "Current Factory", "Region", "Ship Mode",
        "Estimated Distance Miles", "Units", "Sales", "Cost", "Profit Margin %"
    ]
    target = "Lead Time Days"
    X = data[features]
    y = data[target]

    categorical = ["Product Name", "Current Factory", "Region", "Ship Mode"]
    numerical = ["Estimated Distance Miles", "Units", "Sales", "Cost", "Profit Margin %"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, random_state=42, n_jobs=-1, max_depth=14
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    fitted = {}
    results = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred),
        })

    results_df = pd.DataFrame(results).sort_values(["RMSE", "MAE"])
    best_name = results_df.iloc[0]["Model"]
    return fitted, results_df, best_name

def simulate(product, region, ship_mode, priority):
    df = load_data()
    config = load_config()
    factories = config["factories"]
    models, _, best_name = train_models(df)
    model = models[best_name]

    product_rows = df[df["Product Name"] == product]
    current_factory = config["product_factory"][product]

    if product_rows.empty:
        raise ValueError("No historical rows are available for the selected product.")

    baseline = product_rows[
        (product_rows["Region"] == region) &
        (product_rows["Ship Mode"] == ship_mode)
    ]
    if baseline.empty:
        baseline = product_rows

    typical_units = float(baseline["Units"].median())
    typical_sales = float(baseline["Sales"].median())
    typical_cost = float(baseline["Cost"].median())
    typical_margin = float(baseline["Profit Margin %"].median())

    rlat, rlon = region_coords(region)
    rows = []
    for factory_name, coords in factories.items():
        distance = haversine(
            coords["latitude"], coords["longitude"], rlat, rlon
        )
        scenario = pd.DataFrame([{
            "Product Name": product,
            "Current Factory": factory_name,
            "Region": region,
            "Ship Mode": ship_mode,
            "Estimated Distance Miles": distance,
            "Units": typical_units,
            "Sales": typical_sales,
            "Cost": typical_cost,
            "Profit Margin %": typical_margin,
        }])
        model_prediction = float(model.predict(scenario)[0])

        # Distance adjustment makes alternate-factory scenarios responsive even
        # when historical data contains only the current product-factory pairing.
        current_coords = factories[current_factory]
        current_distance = haversine(
            current_coords["latitude"], current_coords["longitude"], rlat, rlon
        )
        adjusted_lead = model_prediction + ((distance - current_distance) / 650.0)
        adjusted_lead *= SHIP_MODE_FACTOR.get(ship_mode, 1.0)
        adjusted_lead = max(1.0, adjusted_lead)

        logistics_cost = distance * 0.055 * max(1.0, typical_units)
        estimated_profit = typical_sales - typical_cost - logistics_cost
        profit_stability = np.clip(
            100 - abs(logistics_cost) / max(typical_sales, 1) * 100, 0, 100
        )
        rows.append({
            "Factory": factory_name,
            "Predicted Lead Time": adjusted_lead,
            "Distance (miles)": distance,
            "Estimated Logistics Cost": logistics_cost,
            "Estimated Profit After Logistics": estimated_profit,
            "Profit Stability": profit_stability,
        })

    result = pd.DataFrame(rows)
    current_lead = float(
        result.loc[result["Factory"] == current_factory, "Predicted Lead Time"].iloc[0]
    )
    current_profit = float(
        result.loc[result["Factory"] == current_factory, "Estimated Profit After Logistics"].iloc[0]
    )

    result["Lead Time Reduction %"] = (
        (current_lead - result["Predicted Lead Time"]) / max(current_lead, 1)
    ) * 100
    result["Profit Impact"] = (
        result["Estimated Profit After Logistics"] - current_profit
    )
    result["Risk Score"] = np.clip(
        50 - result["Lead Time Reduction %"] + np.maximum(-result["Profit Impact"], 0) * 4,
        0, 100
    )
    result["Confidence Score"] = np.clip(
        88 - (result["Distance (miles)"] / 1000) * 4, 55, 95
    )

    speed_weight = priority / 100
    profit_weight = 1 - speed_weight
    speed_component = np.clip(result["Lead Time Reduction %"], -100, 100)
    profit_component = np.clip(result["Profit Stability"], 0, 100)
    result["Recommendation Score"] = (
        speed_weight * speed_component +
        profit_weight * profit_component -
        0.20 * result["Risk Score"]
    )
    result["Current Assignment"] = result["Factory"].eq(current_factory)
    return result.sort_values("Recommendation Score", ascending=False), current_factory, best_name

st.title("🍬 Factory Reallocation & Shipping Optimization")
st.caption("Nassau Candy Distributor — predictive decision intelligence prototype")

df = load_data()
models, model_results, best_model = train_models(df)

with st.sidebar:
    st.header("Simulation Controls")
    product = st.selectbox("Product", sorted(df["Product Name"].unique()))
    region = st.selectbox("Destination region", sorted(df["Region"].unique()))
    ship_mode = st.selectbox("Ship mode", sorted(df["Ship Mode"].unique()))
    priority = st.slider(
        "Optimization priority: profit ← → speed",
        min_value=0, max_value=100, value=60
    )
    run = st.button("Generate Recommendation", type="primary", use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview", "Optimization Simulator",
    "Model Performance", "Data Explorer"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{df['Order ID'].nunique():,}")
    c2.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    c3.metric("Gross Profit", f"${df['Gross Profit'].sum():,.0f}")
    c4.metric("Average Lead Time", f"{df['Lead Time Days'].mean():.1f} days")

    factory_perf = df.groupby("Current Factory", as_index=False).agg(
        Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Lead Time Days", "mean"),
    )
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            factory_perf, x="Current Factory", y="Avg_Lead_Time",
            title="Average Lead Time by Factory"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            factory_perf, x="Sales", y="Gross_Profit",
            size="Orders", hover_name="Current Factory",
            title="Factory Sales vs Gross Profit"
        )
        st.plotly_chart(fig, use_container_width=True)

    route_perf = df.groupby(["Region", "Ship Mode"], as_index=False).agg(
        Avg_Lead_Time=("Lead Time Days", "mean"),
        Gross_Profit=("Gross Profit", "sum"),
        Orders=("Order ID", "nunique"),
    )
    fig = px.bar(
        route_perf.sort_values("Avg_Lead_Time", ascending=False),
        x="Region", y="Avg_Lead_Time", color="Ship Mode",
        barmode="group", title="Regional Shipping Performance"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if run:
        result, current_factory, model_name = simulate(
            product, region, ship_mode, priority
        )
        best = result.iloc[0]
        current = result[result["Factory"] == current_factory].iloc[0]

        st.subheader("Recommendation")
        a, b, c, d = st.columns(4)
        a.metric("Current Factory", current_factory)
        b.metric("Recommended Factory", best["Factory"])
        c.metric(
            "Expected Lead-Time Change",
            f"{best['Lead Time Reduction %']:.1f}%"
        )
        d.metric(
            "Estimated Profit Impact",
            f"${best['Profit Impact']:.2f}"
        )

        if best["Factory"] == current_factory:
            st.success(
                "The current factory remains the strongest option under the selected priority."
            )
        elif best["Profit Impact"] < 0:
            st.warning(
                "The recommended reassignment improves the combined score but may reduce estimated profit."
            )
        else:
            st.success(
                "The recommended factory improves the selected speed-profit objective."
            )

        display = result[[
            "Factory", "Current Assignment", "Predicted Lead Time",
            "Distance (miles)", "Lead Time Reduction %", "Profit Impact",
            "Risk Score", "Confidence Score", "Recommendation Score"
        ]].copy()
        st.dataframe(
            display.style.format({
                "Predicted Lead Time": "{:.2f}",
                "Distance (miles)": "{:.0f}",
                "Lead Time Reduction %": "{:.1f}%",
                "Profit Impact": "${:.2f}",
                "Risk Score": "{:.1f}",
                "Confidence Score": "{:.1f}%",
                "Recommendation Score": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            result.sort_values("Predicted Lead Time"),
            x="Factory", y="Predicted Lead Time",
            color="Recommendation Score",
            title="Predicted Performance Across Factories"
        )
        st.plotly_chart(fig, use_container_width=True)

        config = load_config()
        map_rows = []
        for _, row in result.iterrows():
            f = config["factories"][row["Factory"]]
            map_rows.append({
                "Factory": row["Factory"],
                "lat": f["latitude"],
                "lon": f["longitude"],
                "Score": row["Recommendation Score"],
            })
        st.map(pd.DataFrame(map_rows), latitude="lat", longitude="lon")

        st.download_button(
            "Download Scenario Results",
            result.to_csv(index=False).encode("utf-8"),
            file_name="factory_reallocation_recommendation.csv",
            mime="text/csv",
        )
        st.caption(f"Prediction model used: {model_name}")
    else:
        st.info("Choose the filters in the sidebar and click Generate Recommendation.")

with tab3:
    st.subheader("Model Evaluation")
    st.dataframe(
        model_results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.success(f"Selected model: {best_model}")
    fig = px.bar(
        model_results, x="Model", y=["MAE", "RMSE"],
        barmode="group", title="Prediction Error Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Important: alternate-factory recommendations are scenario estimates. "
        "The source dataset does not contain actual historical reassignments or freight charges."
    )

with tab4:
    st.subheader("Filtered Operational Data")
    selected_factory = st.multiselect(
        "Factory filter", sorted(df["Current Factory"].unique()),
        default=sorted(df["Current Factory"].unique())
    )
    selected_region = st.multiselect(
        "Region filter", sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique())
    )
    filtered = df[
        df["Current Factory"].isin(selected_factory) &
        df["Region"].isin(selected_region)
    ]
    st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Filtered Data",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_nassau_data.csv",
        mime="text/csv",
    )
