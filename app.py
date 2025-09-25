import numpy as np, pandas as pd, streamlit as st, joblib, tensorflow as tf

# --- Page setup & single-line title ---
st.set_page_config(page_title="San Mateo Home Price Estimator", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .app-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0 0 .5rem 0; }
    </style>
    <h1 class="app-title">🏠 San Mateo Home Price Estimator</h1>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    pre = joblib.load("preprocessor.joblib")              # fitted ColumnTransformer (trained with ZIP as string/object)
    model = tf.keras.models.load_model("model.keras")     # trained TF model
    return pre, model

pre, model = load_artifacts()

# Allowed ZIPs (fixed pull-down)
ALLOWED_ZIPS = ["94401", "94402", "94403", "94404", "94010"]

def prep_for_model(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    if "ZIP" in df.columns:
        # keep ZIP as string/object with zero-padding to 5 chars
        df["ZIP"] = df["ZIP"].astype("string").str.strip().str.zfill(5)
    X = pre.transform(df)
    if hasattr(X, "toarray"):  # handle sparse matrices
        X = X.toarray()
    return X.astype("float32")

# --- Single property form ---
st.subheader("Single Family Home")
c1, c2 = st.columns(2)
with c1:
    beds  = st.number_input("Beds", min_value=0, value=3, step=1)
    baths = st.number_input("Baths", min_value=0.0, value=2.0, step=0.5)
    sqft  = st.number_input("Living Area (sq ft)", min_value=0, value=1600, step=50)
with c2:
    lot   = st.number_input("Lot Size (sq ft)", min_value=0, value=5000, step=100)
    year  = st.number_input("Year Built", min_value=1800, max_value=2100, value=1975, step=1)
    zipc  = st.selectbox("ZIP code", ALLOWED_ZIPS, index=0)

if st.button("Estimate Price"):
    row = pd.DataFrame([{
        "ZIP": zipc,  # keep as string for OHE
        "BEDS": beds, "BATHS": baths, "SQUARE_FEET": sqft,
        "LOT_SIZE": lot, "YEAR_BUILT": year
    }])
    X1 = prep_for_model(row)
    pred = float(model.predict(X1, verbose=0).ravel()[0])  # dollars
    st.success(f"Estimated Price: **${pred:,.0f}**")

# --- Batch CSV scoring in sidebar (restricted to allowed ZIPs) ---
st.sidebar.header("Batch predict (CSV)")
csv = st.sidebar.file_uploader(
    "Upload CSV with columns: ZIP, BEDS, BATHS, SQUARE_FEET, LOT_SIZE, YEAR_BUILT",
    type=["csv"]
)
if csv is not None:
    df_in = pd.read_csv(csv)
    need = ["ZIP","BEDS","BATHS","SQUARE_FEET","LOT_SIZE","YEAR_BUILT"]
    miss = [c for c in need if c not in df_in.columns]
    if miss:
        st.sidebar.error(f"Missing columns: {miss}")
    else:
        # enforce string ZIPs and restrict to allowed set
        df_in["ZIP"] = df_in["ZIP"].astype("string").str.strip().str.zfill(5)
        bad_mask = ~df_in["ZIP"].isin(ALLOWED_ZIPS)
        if bad_mask.any():
            bad_zips = sorted(df_in.loc[bad_mask, "ZIP"].unique().tolist())
            st.sidebar.error(
                f"{bad_mask.sum()} row(s) have disallowed ZIPs: {', '.join(bad_zips)}. "
                f"Allowed ZIPs: {', '.join(ALLOWED_ZIPS)}"
            )
        else:
            Xb = prep_for_model(df_in[need])
            preds = model.predict(Xb, verbose=0).ravel()
            out = df_in.copy()
            out["Predicted_PRICE"] = preds
            st.sidebar.success(f"Scored {len(out):,} rows")
            st.sidebar.download_button(
                "Download predictions",
                out.to_csv(index=False).encode(),
                "predictions.csv",
                "text/csv"
            )
