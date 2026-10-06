"""CommuniCorp – Case 6-2 Capacity Concerns (Streamlit app)."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds

st.set_page_config(page_title="Server Capacity Planner", page_icon="🖥️", layout="wide")
st.title("CommuniCorp – Server Capacity Planner")
st.caption("Case 6-2 Capacity Concerns: choose which servers to buy, and when, to support the intranet roll-out at minimum cost.")

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Policy settings")
    budget = st.number_input("Budget for months 1 + 2 ($)", 0, 1_000_000, 9500, step=500)
    disc_months = st.number_input("Discounts apply through month", 0, 5, 2)
    need_powerful = st.number_input("Powerful servers needed by Manufacturing's month", 0, 5, 1)
    mfg_month = st.number_input("Manufacturing joins in month", 1, 5, 3)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Roll-out timeline")
    users_df = st.data_editor(pd.DataFrame({
        "Month": [1, 2, 3, 4, 5],
        "Department": ["(information & buy-in)", "Sales", "Manufacturing", "Warehouse", "Marketing"],
        "New Users": [0, 60, 200, 30, 75],
    }), hide_index=True, disabled=["Month"])
with c2:
    st.subheader("Servers")
    srv_df = st.data_editor(pd.DataFrame({
        "Server": ["Mini Desktop", "Desktop", "Workstation", "Full Rack"],
        "Supports (employees)": [30, 80, 200, 2000],
        "Cost ($)": [2500, 5000, 10000, 25000],
        "Early Discount (%)": [0, 0, 10, 25],
        "Counts as Powerful": [False, True, True, True],
    }), hide_index=True)

new = users_df["New Users"].astype(float).to_numpy()
cum = np.cumsum(new)
T = len(new)
names = srv_df["Server"].tolist()
cap = srv_df["Supports (employees)"].astype(float).to_numpy()
price = srv_df["Cost ($)"].astype(float).to_numpy()
disc = srv_df["Early Discount (%)"].astype(float).to_numpy() / 100
powerful = srv_df["Counts as Powerful"].astype(bool).to_numpy()
K = len(names)

def unit_cost(m, k):
    return price[k] * (1 - disc[k]) if m <= disc_months else price[k]

TIE = 1.0  # tiny tie-breaker: prefer fewer servers when costs are equal

def solve_full():
    """Part B – optimise the whole planning period at once."""
    idx = [(m, k) for m in range(1, T + 1) for k in range(K)]
    c = np.array([unit_cost(m, k) + TIE for m, k in idx])
    A, lb, ub = [], [], []
    for M in range(1, T + 1):
        A.append([cap[k] if m <= M else 0 for m, k in idx]); lb.append(cum[M - 1]); ub.append(np.inf)
    A.append([unit_cost(m, k) if m <= 2 else 0 for m, k in idx]); lb.append(0); ub.append(budget)
    A.append([1 if (m <= mfg_month and powerful[k]) else 0 for m, k in idx]); lb.append(need_powerful); ub.append(np.inf)
    r = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(len(c)), bounds=Bounds(0, np.inf))
    if not r.success:
        return None
    x = np.round(r.x).astype(int).reshape(T, K)
    return x

def solve_myopic():
    """Part A – optimise one month at a time, given earlier purchases."""
    x = np.zeros((T, K), dtype=int)
    spent12 = 0.0
    for m in range(1, T + 1):
        owned = x[:m - 1].sum(axis=0)
        c = np.array([unit_cost(m, k) + TIE for k in range(K)])
        A, lb, ub = [cap], [cum[m - 1] - owned @ cap], [np.inf]
        if m <= 2:
            A.append([unit_cost(m, k) for k in range(K)]); lb.append(0); ub.append(budget - spent12)
        if m == mfg_month:
            A.append(powerful.astype(float)); lb.append(need_powerful - owned[powerful].sum()); ub.append(np.inf)
        r = milp(c, constraints=LinearConstraint(np.array(A), lb, ub), integrality=np.ones(K), bounds=Bounds(0, np.inf))
        if not r.success:
            return None, m
        x[m - 1] = np.round(r.x).astype(int)
        if m <= 2:
            spent12 += sum(unit_cost(m, k) * x[m - 1, k] for k in range(K))
    return x, None

def show_plan(x, label):
    rows = []
    for m in range(1, T + 1):
        bought = [f"{x[m-1,k]} × {names[k]}" for k in range(K) if x[m - 1, k] > 0]
        cost = sum(unit_cost(m, k) * x[m - 1, k] for k in range(K))
        rows.append({"Month": m, "Department": users_df["Department"][m - 1],
                     "Buy": ", ".join(bought) or "—", "Cost ($)": cost,
                     "Cumulative Users": cum[m - 1],
                     "Cumulative Capacity": (x[:m].sum(axis=0) @ cap)})
    df = pd.DataFrame(rows)
    total = df["Cost ($)"].sum()
    st.metric(f"{label} – total cost", f"${total:,.0f}")
    st.dataframe(df.style.format({"Cost ($)": "${:,.0f}", "Cumulative Users": "{:,.0f}", "Cumulative Capacity": "{:,.0f}"}),
                 hide_index=True, width="stretch")
    chart = df.set_index(df["Month"].map(lambda m: f"Month {m}"))[["Cumulative Users", "Cumulative Capacity"]]
    st.bar_chart(chart, stack=False)
    return total

st.divider()
tabA, tabB, tabF = st.tabs(["Part A – month by month", "Part B – whole period", "Model formulation"])
with tabA:
    st.markdown("Each month Emily buys the cheapest servers that cover that month's new users, given what she already owns.")
    xa, bad = solve_myopic()
    totA = show_plan(xa, "Part A") if xa is not None else st.error(f"No feasible purchase in month {bad}.")
with tabB:
    st.markdown("Emily plans all five months together, so she can buy bigger (discounted) servers early.")
    xb = solve_full()
    totB = show_plan(xb, "Part B") if xb is not None else st.error("No feasible plan with these settings.")
    if xa is not None and xb is not None:
        st.success(f"Planning the whole period saves **${totA - totB:,.0f}** compared with Part A.")
with tabF:
    st.markdown(r"""
**Decision variables:** $x_{m,k}$ = number of servers of type $k$ bought in month $m$ (whole numbers).

**Objective:** minimise $\sum_{m,k} c_{m,k}\,x_{m,k}$, where $c_{m,k}$ is the list price less any early-purchase discount.

**Constraints**
- Capacity: for every month $M$, $\sum_{m \le M}\sum_k \text{cap}_k\,x_{m,k} \ge$ cumulative users in month $M$.
- Budget: spending in months 1 and 2 $\le$ \$9,500.
- Manufacturing: at least one Desktop, Workstation or Full Rack server owned by month 3.

**Part A** solves this one month at a time (only that month's purchases change); **Part B** solves all months together.
Solved as an integer programme with SciPy's HiGHS solver.
""")
