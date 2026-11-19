import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Stats Regression Challenge", layout="wide")

st.title("📊 Class Data & Linear Regression Challenge")
st.markdown(
    "Collect data from your classmates on **two quantitative variables**, run the statistical analysis, "
    "and complete the **5-question Regression Challenge (100 pts)** based on your results!"
)

# --- STEP 1: STUDENT INFO & VARIABLE SETUP ---
st.header("Step 1: Define Your Variables & Enter Class Data")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    student_names = st.text_input("Student / Group Name(s)", placeholder="e.g., Alex & Jordan")
with col_info2:
    x_name = st.text_input("Explanatory Variable (x)", value="Hours of Sleep")
    x_unit = st.text_input("Units for x", value="hours")
with col_info3:
    y_name = st.text_input("Response Variable (y)", value="Reaction Time Score")
    y_unit = st.text_input("Units for y", value="points")

# Initialize Data Table in Session State
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame({
        "Student Name": ["Student 1", "Student 2", "Student 3", "Student 4", "Student 5", "Student 6"],
        "x": [None, None, None, None, None, None],
        "y": [None, None, None, None, None, None],
    })

btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 4])
with btn_col1:
    if st.button("Clear Table"):
        st.session_state.df_data = pd.DataFrame({
            "Student Name": [f"Student {i}" for i in range(1, 7)],
            "x": [None] * 6,
            "y": [None] * 6,
        })
        st.rerun()
with btn_col2:
    if st.button("Load Demo Data"):
        st.session_state.df_data = pd.DataFrame({
            "Student Name": ["Maya", "Liam", "Chloe", "Noah", "Sam", "Ava", "Lucas", "Mia"],
            "x": [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
            "y": [64, 70, 68, 76, 85, 82, 88, 93],
        })
        st.rerun()

st.caption("Type your collected measurements directly into the **x** and **y** columns below. Click the **+** row at the bottom to add more students.")

edited_df = st.data_editor(
    st.session_state.df_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Student Name": st.column_config.TextColumn("Classmate Name (Optional)"),
        "x": st.column_config.NumberColumn(f"x: {x_name} ({x_unit})", format="%.2f"),
        "y": st.column_config.NumberColumn(f"y: {y_name} ({y_unit})", format="%.2f"),
    },
    key="data_editor_widget"
)

# Clean numeric pairs
clean_df = edited_df.copy()
clean_df["x"] = pd.to_numeric(clean_df["x"], errors="coerce")
clean_df["y"] = pd.to_numeric(clean_df["y"], errors="coerce")
clean_df = clean_df.dropna(subset=["x", "y"]).reset_index(drop=True)

if len(clean_df) < 3 or clean_df["x"].nunique() < 2 or clean_df["y"].nunique() < 2:
    st.warning("⏳ Enter at least **3 valid numeric (x, y) data pairs** with different values above to unlock your statistical analysis and the 5-question game!")
    st.stop()

# --- STEP 2: STATISTICAL ANALYSIS ---
x_vals = clean_df["x"].to_numpy(dtype=float)
y_vals = clean_df["y"].to_numpy(dtype=float)

r = float(np.corrcoef(x_vals, y_vals)[0, 1])
r_sq = r ** 2
slope, intercept = np.polyfit(x_vals, y_vals, 1)
slope = float(slope)
intercept = float(intercept)

y_pred = intercept + slope * x_vals
residuals = y_vals - y_pred
clean_df["Predicted (ŷ)"] = np.round(y_pred, 3)
clean_df["Residual (y - ŷ)"] = np.round(residuals, 3)

sign_str = "+" if slope >= 0 else "-"
eq_str = f"ŷ = {intercept:.3f} {sign_str} {abs(slope):.3f}x"

st.divider()
st.header("Step 2: Statistical Analysis Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Correlation (r)", f"{r:.4f}")
m2.metric("R-Squared (r²)", f"{r_sq:.4f}")
m3.metric("Slope (b)", f"{slope:.3f}")
m4.metric("y-Intercept (a)", f"{intercept:.3f}")

st.success(f"**Least-Squares Regression Line (LSRL):**  `{eq_str}`")

chart_col, table_col = st.columns([1.4, 1])

with chart_col:
    line_df = pd.DataFrame({
        "x": [float(x_vals.min()), float(x_vals.max())],
        "y": [intercept + slope * float(x_vals.min()), intercept + slope * float(x_vals.max())]
    })

    scatter = (
        alt.Chart(clean_df)
        .mark_circle(size=90, color="#0f766e")
        .encode(
            x=alt.X("x:Q", title=f"{x_name} ({x_unit})", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title=f"{y_name} ({y_unit})", scale=alt.Scale(zero=False)),
            tooltip=["Student Name", "x", "y", "Predicted (ŷ)", "Residual (y - ŷ)"]
        )
    )

    reg_line = (
        alt.Chart(line_df)
        .mark_line(color="#1d4ed8", strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )

    chart = (scatter + reg_line).properties(
        title=f"{y_name} vs. {x_name} (n = {len(clean_df)})",
        height=350
    )
    st.altair_chart(chart, use_container_width=True)

with table_col:
    st.subheader("Data & Residuals Table")
    st.dataframe(clean_df, use_container_width=True, height=350)

# --- STEP 3: 5-QUESTION WORKSHEET CHALLENGE ---
st.divider()
st.header("Step 3: 5-Question Regression Challenge (100 Points)")
st.markdown("Use your scatterplot, correlation coefficient, and regression equation above to answer all 5 questions.")

score = 0

if abs(r) < 0.05:
    true_dir = "Neither (No linear direction)"
elif r > 0:
    true_dir = "Positive"
else:
    true_dir = "Negative"

if abs(r) >= 0.70:
    true_str = "Strong"
elif abs(r) >= 0.40:
    true_str = "Moderate"
else:
    true_str = "Weak"

# QUESTION 1: Direction, Strength & Context
with st.container(border=True):
    st.subheader("Question 1 (20 pts): Direction, Strength & Interpreting r")
    q1_c1, q1_c2 = st.columns(2)
    with q1_c1:
        ans_dir = st.radio(
            "1a. What is the direction of the linear relationship?",
            ["Positive", "Negative", "Neither (No linear direction)"],
            index=None,
            key="q1_dir"
        )
    with q1_c2:
        ans_str = st.radio(
            "1b. What is the strength of the linear relationship? (Weak: |r| < 0.4, Moderate: 0.4 ≤ |r| < 0.7, Strong: |r| ≥ 0.7)",
            ["Weak", "Moderate", "Strong"],
            index=None,
            key="q1_str"
        )
    ans_r_ctx = st.text_input(
        "1c. Write one complete sentence interpreting r in the context of your variables:",
        placeholder=f"There is a {true_str.lower()}, {true_dir.lower()} linear relationship between {x_name.lower()} and {y_name.lower()}...",
        key="q1_ctx"
    )

    if ans_dir is not None and ans_str is not None:
        if ans_dir == true_dir and ans_str == true_str and len(ans_r_ctx.strip()) >= 15:
            score += 20
            st.success(f"✅ Correct (+20 pts)! Since r = {r:.4f}, the relationship is **{true_str}** and **{true_dir}**.")
        elif ans_dir == true_dir and ans_str == true_str:
            st.warning("⚠️ Your direction and strength are right! Type your full contextual sentence in 1c to earn the 20 points.")
        else:
            st.error(f"❌ Not quite. Look at r = {r:.4f}: check the sign for direction and |r| = {abs(r):.4f} for strength.")

# QUESTION 2: Properties of r & Unit Conversion
with st.container(border=True):
    st.subheader("Question 2 (20 pts): Unit Conversions & Properties of Correlation")
    q2_opts = [
        f"It stays the exact same (r = {r:.4f}), because correlation is unitless and unaffected by linear unit conversions.",
        f"It changes proportionally to the conversion factor, and its units become {y_unit} per {x_unit}.",
        "It becomes closer to +1 or -1 because larger unit numbers increase linear strength.",
        "It becomes negative if you subtract a constant during the conversion."
    ]
    ans_q2 = st.radio(
        f"Suppose you convert all measurements of **{y_name}** into completely different units (for example, multiplying by 2.54 or converting °F to °C). What happens to your correlation coefficient r = {r:.4f}?",
        q2_opts,
        index=None,
        key="q2_radio"
    )
    if ans_q2 is not None:
        if ans_q2 == q2_opts[0]:
            score += 20
            st.success("✅ Correct (+20 pts)! Standardizing removes units, so changing units of x or y never changes r.")
        else:
            st.error("❌ Incorrect. Remember: r has no units and does not change when variables are rescaled or shifted.")

# QUESTION 3: Slope & Y-Intercept Interpretation
with st.container(border=True):
    st.subheader("Question 3 (20 pts): Interpreting Slope & y-Intercept")
    dir_word = "increases" if slope >= 0 else "decreases"
    q3_opts = [
        f"For each additional 1 {x_unit} of {x_name}, the predicted {y_name} {dir_word} by {abs(slope):.3f} {y_unit}.",
        f"For each additional 1 {y_unit} of {y_name}, the predicted {x_name} {dir_word} by {abs(slope):.3f} {x_unit}.",
        f"When {x_name} is 0 {x_unit}, the predicted {y_name} is {abs(slope):.3f} {y_unit}.",
        f"Every student's {y_name} is guaranteed to change by {abs(slope):.3f} {y_unit} per {x_unit}."
    ]
    ans_q3 = st.radio(
        f"3a. Which statement correctly interprets the slope (**{slope:.3f}**) of your regression line `{eq_str}`?",
        q3_opts,
        index=None,
        key="q3_slope"
    )
    ans_q3_yint = st.text_input(
        f"3b. Does your y-intercept ({intercept:.3f} {y_unit} when {x_name} = 0 {x_unit}) have a practical, meaningful interpretation here? Explain briefly:",
        key="q3_yint"
    )
    if ans_q3 is not None:
        if ans_q3 == q3_opts[0] and len(ans_q3_yint.strip()) >= 10:
            score += 20
            st.success("✅ Correct (+20 pts)! You identified the slope as the predicted change in y per 1-unit increase in x and evaluated the y-intercept.")
        elif ans_q3 == q3_opts[0]:
            st.warning("⚠️ Great job on the slope! Complete 3b explaining whether x = 0 makes practical sense to earn the 20 points.")
        else:
            st.error("❌ Incorrect. Check which variable is x (explanatory) vs. y (response), and remember regression gives *predicted* values.")

# QUESTION 4: Prediction & Largest Residual
max_res_idx = int(np.argmax(np.abs(residuals)))
target_x = float(x_vals[max_res_idx])
target_y = float(y_vals[max_res_idx])
target_yhat = intercept + slope * target_x
target_res = target_y - target_yhat
target_label = clean_df.loc[max_res_idx, "Student Name"] or f"Row {max_res_idx + 1}"

with st.container(border=True):
    st.subheader("Question 4 (20 pts): Prediction & Calculating a Residual")
    st.markdown(
        f"Look at the observation for **{target_label}**, which has the largest residual in your dataset: "
        f"**(x = {target_x:.2f} {x_unit}, y = {target_y:.2f} {y_unit})**."
    )
    q4_c1, q4_c2 = st.columns(2)
    with q4_c1:
        user_yhat = st.number_input(
            f"4a. Using `{eq_str}`, calculate the predicted value (ŷ) when x = {target_x:.2f}:",
            value=0.0,
            format="%.3f",
            key="q4_yhat"
        )
    with q4_c2:
        user_res = st.number_input(
            f"4b. Calculate the residual (Residual = Actual y − Predicted ŷ):",
            value=0.0,
            format="%.3f",
            key="q4_res"
        )
    check_q4 = st.checkbox("Submit Question 4 Answers", key="q4_submit")
    if check_q4:
        yhat_ok = abs(user_yhat - target_yhat) <= 0.15
        res_ok = abs(user_res - target_res) <= 0.15
        if yhat_ok and res_ok:
            score += 20
            pos_neg = "higher than" if target_res > 0 else "lower than"
            st.success(
                f"✅ Correct (+20 pts)! Predicted ŷ = {target_yhat:.3f} and Residual = {target_res:.3f}. "
                f"This student's actual {y_name} was {abs(target_res):.2f} {y_unit} {pos_neg} predicted."
            )
        else:
            st.error(
                f"❌ Check your arithmetic. Plug x = {target_x:.2f} into `{eq_str}` for ŷ, "
                f"then compute Residual = {target_y:.2f} − ŷ."
            )

# QUESTION 5: Extrapolation & Causation
extrap_x = round(float(x_vals.max() + 2 * (x_vals.max() - x_vals.min())), 1)

with st.container(border=True):
    st.subheader("Question 5 (20 pts): Extrapolation & Association vs. Causation")
    q5_opts = [
        "No — predicting far outside the observed x-range is unreliable (extrapolation), and correlation does not prove causation.",
        "Yes — as long as |r| is strong, the linear pattern is guaranteed to hold forever and proves x causes y.",
        "No — you can only make predictions if r is exactly +1.0 or -1.0.",
        "Yes — linear regression eliminates all confounding variables."
    ]
    ans_q5 = st.radio(
        f"Suppose someone uses your equation to predict **{y_name}** when **{x_name} = {extrap_x} {x_unit}**, "
        f"and claims that changing **{x_name}** directly causes **{y_name}** to change. Is this valid?",
        q5_opts,
        index=None,
        key="q5_radio"
    )
    if ans_q5 is not None:
        if ans_q5 == q5_opts[0]:
            score += 20
            st.success("✅ Correct (+20 pts)! Predicting far beyond your data is **extrapolation**, and observational class data shows association, not causation.")
        else:
            st.error("❌ Incorrect. Think about what happens when you predict far outside your observed x-values and whether observational data proves cause-and-effect.")

# --- SCOREBOARD SUMMARY ---
st.divider()
st.subheader(f"🏆 Final Challenge Score: {score} / 100 Points")
st.progress(score / 100)

if score == 100:
    st.balloons()
    who = student_names if student_names.strip() else "Your Group"
    st.success(
        f"🎉 **Challenge Complete!** {who} scored **100/100** analyzing **{y_name}** vs. **{x_name}** "
        f"(n = {len(clean_df)}, r = {r:.4f}, `{eq_str}`). Take a screenshot of this screen to submit!"
    )
