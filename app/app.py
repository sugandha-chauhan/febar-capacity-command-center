"""LSH Capacity Command Center (synthetic data).

Pages: Command Center, Discharge Worklist, LOS Performance, Ask (Genie).
* Lakebase (app service principal): bed board + unit status reads, discharge plan / action write-back.
* SQL warehouse and Genie (signed-in user's token when available): metric views and questions,
  so Unity Catalog permissions, masks and row filters apply to the viewer.
If Lakebase isn't connected yet, the app reads the same tables from the lakehouse and disables write-back.
"""
import os
import time
from datetime import datetime, time as dtime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem, StatementState

WAREHOUSE_ID = os.environ["WAREHOUSE_ID"]
GENIE_SPACE_ID = os.environ["GENIE_SPACE_ID"]
GOLD = os.environ["GOLD_SCHEMA"]
SERVING = os.environ["SERVING_SCHEMA"]
LB_ENDPOINT = os.environ["LAKEBASE_ENDPOINT"]
LB_HOST = os.environ["LAKEBASE_HOST"]
LB_DB = os.environ.get("LAKEBASE_DB", "databricks_postgres")

NM = {"primary": "#4a4b96", "deep": "#241852", "violet": "#9348b4", "ocean": "#1580a8", "good": "#2e9e6b",
      "warn": "#e0a100", "bad": "#d13438", "muted": "#5d6073", "bg": "#f5f5fa", "nm100": "#e7e7f4"}
TIER_COLOR = {"High": NM["bad"], "Medium": NM["warn"], "Low": NM["good"], "Observation": "#8a8ca0"}
ACTION_TYPES = ["Escalate barrier", "SNF referral sent", "Family meeting", "Transport booked", "Pharmacy meds-to-beds"]

st.set_page_config(page_title="LSH Capacity Command Center", page_icon="\U0001F3E5", layout="wide")

# ---------------------------------------------------------------- theme
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{ font-family: 'Inter', system-ui, sans-serif; color: {NM['deep']}; }}
  h1, h2, h3, h4 {{ font-family: 'DM Sans', 'Inter', sans-serif !important; color: {NM['deep']} !important; letter-spacing: -0.01em; }}
  .stApp {{ background: {NM['bg']}; }}
  section[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid #ecedf3; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
  .nm-hero {{ background: linear-gradient(135deg, #322071 0%, #4a4b96 60%, #6462a8 100%); color: #fff; border-radius: 16px;
             padding: 18px 24px; box-shadow: 0 8px 30px -6px rgba(74,75,150,0.30); margin-bottom: 18px; }}
  .nm-hero h2 {{ color: #fff !important; margin: 0; font-size: 1.55rem; }}
  .nm-hero p {{ color: #e7e7f4; margin: 4px 0 0 0; font-size: 0.92rem; }}
  .nm-card {{ background: #fff; border-radius: 14px; padding: 14px 16px; height: 100%;
             box-shadow: 0 1px 0 0 rgba(36,24,82,0.04), 0 4px 16px -4px rgba(36,24,82,0.10); }}
  .nm-kpi-label {{ font-size: 0.78rem; color: {NM['muted']}; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
  .nm-kpi-value {{ font-family: 'DM Sans', sans-serif; font-size: 1.85rem; font-weight: 700; color: {NM['deep']};
                  font-variant-numeric: tabular-nums; line-height: 1.15; }}
  .nm-kpi-sub {{ font-size: 0.8rem; color: {NM['muted']}; }}
  .nm-tile {{ background: #fff; border-radius: 14px; padding: 12px 14px; border-left: 6px solid {NM['good']};
             box-shadow: 0 1px 0 0 rgba(36,24,82,0.04), 0 4px 16px -4px rgba(36,24,82,0.10); margin-bottom: 10px; }}
  .nm-tile .t-name {{ font-family: 'DM Sans'; font-weight: 700; font-size: 1.0rem; }}
  .nm-tile .t-num {{ font-family: 'DM Sans'; font-weight: 700; font-size: 1.5rem; font-variant-numeric: tabular-nums; }}
  .nm-insight {{ background: linear-gradient(135deg, rgba(74,75,150,0.08) 0%, rgba(147,72,180,0.04) 100%);
                border: 1px solid #dcdde8; border-radius: 14px; padding: 14px 18px; margin: 6px 0 14px 0; }}
  .nm-insight b.h {{ color: {NM['violet']}; font-family: 'DM Sans'; }}
  .nm-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff; font-size: 0.78rem; font-weight: 600; }}
  .nm-banner {{ background: #fff8e1; border: 1px solid #f1d27a; border-radius: 12px; padding: 10px 14px; font-size: 0.88rem; margin-bottom: 12px; }}
  .nm-note {{ background: #fff; border-radius: 12px; padding: 12px 14px; border: 1px solid #ecedf3; }}
  .nm-mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {NM['muted']}; }}
  /* Buttons: secondary = white with indigo text, primary = LSH indigo. No blank lines in this block: Markdown would end the HTML there */
  .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    background: #ffffff !important; color: {NM['deep']} !important; border: 1px solid #cfcfe9 !important;
    border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 1px 0 0 rgba(36,24,82,0.04);
  }}
  .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {{
    background: {NM['nm100']} !important; border-color: {NM['primary']} !important; color: #322071 !important;
  }}
  .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
  .stButton > button[data-testid$="Primary"], .stFormSubmitButton > button[data-testid$="Primary"] {{
    background: linear-gradient(135deg, #322071 0%, #4a4b96 60%, #6462a8 100%) !important; color: #ffffff !important; border: none !important;
  }}
  .stButton > button:disabled, .stFormSubmitButton > button:disabled {{
    background: #ecedf3 !important; color: #8a8ca0 !important; border-color: #dcdde8 !important;
  }}
  .stButton > button p, .stFormSubmitButton > button p {{ color: inherit !important; }}
  .stButton > button:focus-visible {{ box-shadow: 0 0 0 4px rgba(74,75,150,0.14) !important; }}
  /* Inputs, chat box and forms in the same light palette */
  div[data-baseweb="select"] > div, .stTextInput input, .stDateInput input, .stTimeInput input, div[data-baseweb="input"] {{
    background: #ffffff !important; color: {NM['deep']} !important; border-color: #dcdde8 !important;
  }}
  .stChatInput, .stChatInput textarea, [data-testid="stChatInput"] > div {{ background: #ffffff !important; color: {NM['deep']} !important; }}
  [data-testid="stForm"] {{ background: #ffffff; border: 1px solid #ecedf3 !important; border-radius: 14px; }}
  [data-testid="stExpander"] details {{ background: #ffffff; border-radius: 12px; border-color: #ecedf3; }}
  span[data-baseweb="tag"] {{ background: {NM['primary']} !important; }}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="nm-hero"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f'<div class="nm-card"><div class="nm-kpi-label">{label}</div><div class="nm-kpi-value">{value}</div>'
                     f'<div class="nm-kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    st.write("")


def insight(text):
    st.markdown(f'<div class="nm-insight"><b class="h">Insight</b><br>{text}</div>', unsafe_allow_html=True)


def badge(tier):
    return f'<span class="nm-badge" style="background:{TIER_COLOR.get(tier, "#8a8ca0")}">{tier}</span>'


def pct(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "-" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def style_fig(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#ffffff", font=dict(family="Inter", color=NM["deep"]), legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ecedf3")
    return fig


# ---------------------------------------------------------------- identity and clients
def user_email():
    return st.context.headers.get("X-Forwarded-Email") or st.context.headers.get("X-Forwarded-Preferred-Username") or "unknown user"


@st.cache_resource
def sp_client():
    return WorkspaceClient()


def user_client():
    """Signed-in user's token (on-behalf-of) when the app has user authorization; otherwise the app service principal."""
    token = st.context.headers.get("X-Forwarded-Access-Token")
    if token:
        return WorkspaceClient(host=sp_client().config.host, token=token, auth_type="pat")
    return sp_client()


# ---------------------------------------------------------------- SQL warehouse
NUMERIC = {"INT", "LONG", "BIGINT", "SHORT", "BYTE", "DOUBLE", "FLOAT", "DECIMAL"}


def _run_sql(sql, params):
    w = user_client()
    r = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=sql, wait_timeout="50s",
        parameters=[StatementParameterListItem(name=k, value=str(v)) for k, v in (params or {}).items()])
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(r.status.error.message if r.status.error else str(r.status.state))
    cols = r.manifest.schema.columns
    rows = list(r.result.data_array or []) if r.result else []
    for i in range(1, r.manifest.total_chunk_count or 1):
        rows += w.statement_execution.get_statement_result_chunk_n(r.statement_id, i).data_array or []
    df = pd.DataFrame(rows, columns=[c.name for c in cols])
    for c in cols:
        t = (c.type_name.value if c.type_name else "").upper()
        if t in NUMERIC:
            df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
        elif t in ("TIMESTAMP", "TIMESTAMP_NTZ", "DATE"):
            df[c.name] = pd.to_datetime(df[c.name], errors="coerce")
        elif t == "BOOLEAN":
            df[c.name] = df[c.name].map({"true": True, "false": False})
    return df


@st.cache_data(ttl=300, show_spinner=False)
def sql_cached(sql, params_items, viewer):   # viewer is part of the cache key: results follow each user's permissions
    return _run_sql(sql, dict(params_items))


def sql(q, **params):
    return sql_cached(q, tuple(sorted(params.items())), user_email())


# ---------------------------------------------------------------- Lakebase
@st.cache_resource(ttl=2400)   # OAuth database credential lasts 1 hour; refresh every 40 minutes
def _pg_token():
    return sp_client().postgres.generate_database_credential(endpoint=LB_ENDPOINT).token


def pg_conn():
    return psycopg.connect(host=LB_HOST, dbname=LB_DB, user=sp_client().config.client_id, password=_pg_token(),
                           sslmode="require", connect_timeout=10)


def pg_df(q, params=None):
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(q, params or ())
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


@st.cache_data(ttl=60, show_spinner=False)
def lakebase_status():
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {SERVING}.unit_status")
            cur.fetchone()
            cur.execute("SELECT count(*) FROM capacity_ops.discharge_plan")
            cur.fetchone()
        return True, ""
    except Exception as e:   # role or grants not set up yet
        return False, str(e).splitlines()[0][:200]


def lakebase_banner():
    ok, why = lakebase_status()
    if not ok:
        st.markdown('<div class="nm-banner"><b>Lakebase not connected yet</b>: reading the same tables from the lakehouse, '
                    'and saving plans and actions is turned off. Run notebook 06 Step 6 with this app\'s service principal '
                    f'ID to connect it. <span class="nm-mono">{why}</span></div>', unsafe_allow_html=True)
    return ok


def load_unit_status():
    if lakebase_status()[0]:
        df = pg_df(f"SELECT * FROM {SERVING}.unit_status")
    else:
        df = sql(f"SELECT * FROM {GOLD}.unit_status")
    for c in df.columns:
        if c not in ("unit_code", "facility_code", "facility_name", "unit_name", "unit_type", "snapshot_ts"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_board(facility):
    if lakebase_status()[0]:
        df = pg_df(f"""SELECT b.*, p.plan_status, p.target_discharge_ts, p.delay_reason, coalesce(a.actions, 0) AS actions
                       FROM {SERVING}.bed_board b
                       LEFT JOIN capacity_ops.discharge_plan p USING (encounter_id)
                       LEFT JOIN (SELECT encounter_id, count(*) AS actions FROM capacity_ops.discharge_actions GROUP BY 1) a USING (encounter_id)
                       WHERE b.facility_code = %s""", (facility,))
    else:
        df = sql(f"""SELECT *, CAST(NULL AS STRING) AS plan_status, CAST(NULL AS TIMESTAMP) AS target_discharge_ts,
                            CAST(NULL AS STRING) AS delay_reason, 0 AS actions
                     FROM {GOLD}.bed_board WHERE facility_code = :fac""", fac=facility)
    for c in ["long_stay_risk", "los_so_far_days", "expected_los", "days_past_expected", "p_discharge_today",
              "predicted_los_days", "age", "actions"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------- pages
def page_command_center():
    hero("Command Center", "Where will we run out of beds today? Unit capacity at the 7:00 AM huddle snapshot.")
    lakebase_banner()
    us = load_unit_status()
    beds, pts = us.staffed_beds.sum(), us.patients_now.sum()
    snap = pd.to_datetime(us.snapshot_ts.iloc[0])
    tight = us[us.forecast_open_beds_tonight <= 2].sort_values("forecast_open_beds_tonight")
    kpi_row([
        ("Occupancy now", pct(pts / beds), f"{num(pts)} patients / {num(beds)} staffed beds"),
        ("Open beds now", num(us.open_beds_now.sum()), "across 8 hospitals"),
        ("Open beds tonight", num(us.forecast_open_beds_tonight.sum()), "forecast after admissions and discharges"),
        ("Expected discharges", num(us.expected_discharges_today.sum()), "today, from the LOS model"),
        ("High-risk patients", num(us.high_risk_patients.sum()), "long-stay risk 50%+"),
        ("Past expected discharge", num(us.past_expected_discharge.sum()), "still in the hospital"),
    ])

    fac = (us.groupby(["facility_code", "facility_name"], as_index=False)
             .agg(beds=("staffed_beds", "sum"), patients=("patients_now", "sum"), tonight=("forecast_open_beds_tonight", "sum"),
                  discharges=("expected_discharges_today", "sum"), high=("high_risk_patients", "sum"))
             .sort_values("beds", ascending=False))
    fac["occ"] = fac.patients / fac.beds
    worst = fac.sort_values("occ", ascending=False).iloc[0]
    insight(f"<b>{worst.facility_name}</b> is the fullest hospital at {pct(worst.occ)} with {num(worst.tonight)} beds forecast "
            f"open tonight. <b>{len(tight)} of {len(us)}</b> units are forecast to have 2 or fewer open beds tonight. "
            f"About <b>{num(us.expected_discharges_today.sum())}</b> discharges are expected system-wide today: confirming "
            f"them early is the fastest way to open beds before the afternoon admission peak.")

    st.markdown("#### Hospitals")
    cols = st.columns(4)
    for i, r in enumerate(fac.itertuples()):
        red = r.occ > 0.95 or r.tonight < 0.05 * r.beds
        amber = r.occ > 0.90
        color = NM["bad"] if red else NM["warn"] if amber else NM["good"]
        cols[i % 4].markdown(
            f'<div class="nm-tile" style="border-left-color:{color}"><div class="t-name">{r.facility_name}</div>'
            f'<div class="t-num" style="color:{color}">{pct(r.occ)}</div>'
            f'<div class="nm-kpi-sub">{num(r.patients)} / {num(r.beds)} beds &middot; {num(r.tonight)} open tonight<br>'
            f'{num(r.discharges)} expected discharges &middot; {num(r.high)} high-risk</div></div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Unit occupancy now")
        u = us.assign(occupancy=us.occupancy_now * 100).sort_values("occupancy", ascending=True)
        fig = px.bar(u.tail(20), x="occupancy", y="unit_name", orientation="h", color="facility_code",
                     color_discrete_sequence=[NM["primary"], NM["ocean"], NM["violet"], "#8482bf", "#369ec4", "#6462a8", "#adacd6", "#00668f"],
                     labels={"occupancy": "Occupancy %", "unit_name": ""}, title="20 fullest units")
        fig.add_vline(x=95, line_dash="dash", line_color=NM["bad"], annotation_text="95%")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    with right:
        st.markdown("#### Units at risk tonight")
        st.caption("Forecast open beds tonight of 2 or fewer")
        st.dataframe(tight[["facility_code", "unit_name", "staffed_beds", "patients_now", "expected_discharges_today",
                            "expected_admissions_today", "forecast_open_beds_tonight"]]
                     .rename(columns={"facility_code": "Hospital", "unit_name": "Unit", "staffed_beds": "Beds", "patients_now": "Patients",
                                      "expected_discharges_today": "Exp. discharges", "expected_admissions_today": "Exp. admissions",
                                      "forecast_open_beds_tonight": "Open tonight"}),
                     hide_index=True, use_container_width=True, height=520)

    with st.expander("All units"):
        st.dataframe(us[["facility_code", "unit_name", "unit_type", "staffed_beds", "patients_now", "open_beds_now", "high_risk_patients",
                         "past_expected_discharge", "open_barriers", "expected_discharges_today", "expected_admissions_today",
                         "forecast_open_beds_tonight"]].sort_values(["facility_code", "unit_name"]), hide_index=True, use_container_width=True)
    st.caption(f"Snapshot {snap:%a %b %d, %Y %H:%M}. Synthetic data.")


def page_worklist():
    hero("Discharge Worklist", "Which patients should case managers work first, and what's in the way?")
    lb_ok = lakebase_banner()
    us = load_unit_status()
    facilities = us[["facility_code", "facility_name"]].drop_duplicates().sort_values("facility_code")
    c1, c2, c3 = st.columns([2, 3, 2])
    fac = c1.selectbox("Hospital", facilities.facility_code, format_func=lambda c: facilities.set_index("facility_code").facility_name[c],
                       index=int((facilities.facility_code == "LMH").values.argmax()))
    board = load_board(fac)
    units = ["All units"] + sorted(board.unit_name.dropna().unique().tolist())
    unit = c2.selectbox("Unit", units)
    tiers = c3.multiselect("Risk tier", ["High", "Medium", "Low", "Observation"], default=["High", "Medium"])
    b = board if unit == "All units" else board[board.unit_name == unit]
    b = b[b.risk_tier.isin(tiers)] if tiers else b
    b = b.sort_values("long_stay_risk", ascending=False, na_position="last")

    kpi_row([
        ("Patients shown", num(len(b)), f"of {num(len(board))} in house"),
        ("High risk", num((b.risk_tier == "High").sum()), "long-stay risk 50%+"),
        ("Past expected discharge", num((b.days_past_expected > 0).sum()), "need a plan today"),
        ("Open barriers", num(b.open_barrier_category.notna().sum()), "CarePort"),
        ("Plans confirmed", num(b.plan_status.eq("Confirmed").sum()), "in Lakebase" if lb_ok else "Lakebase not connected"),
    ])
    if len(b):
        top = b.iloc[0]
        insight(f"Start with <b>bed {top.bed}</b> ({top.unit_name}): {pct(top.long_stay_risk, 0)} long-stay risk, day "
                f"{num(top.los_so_far_days, 1)} of an expected {num(top.expected_los, 1)}. Drivers: {top.risk_drivers or 'none recorded'}.")

    view = b.assign(
        Risk=(b.long_stay_risk * 100).round(0),
        Day=b.los_so_far_days.round(1),
        Expected=b.expected_los.round(1),
        Predicted=pd.to_datetime(b.predicted_discharge_ts).dt.strftime("%b %d %H:%M"),
        Plan=b.plan_status.fillna("-"))[["bed", "patient_name", "age", "unit_name", "drg_description", "Day", "Expected", "Risk",
                                         "risk_tier", "Predicted", "open_barrier_category", "Plan", "actions", "risk_drivers"]]
    st.dataframe(view.rename(columns={"bed": "Bed", "patient_name": "Patient", "age": "Age", "unit_name": "Unit", "drg_description": "DRG",
                                      "risk_tier": "Tier", "open_barrier_category": "Barrier", "actions": "Actions", "risk_drivers": "Drivers"}),
                 hide_index=True, use_container_width=True, height=380,
                 column_config={"Risk": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%d%%")})

    if not len(b):
        return
    st.markdown("#### Patient detail")
    labels = {r.encounter_id: f"{r.bed} - {r.patient_name} ({pct(r.long_stay_risk, 0) if pd.notna(r.long_stay_risk) else r.risk_tier})"
              for r in b.itertuples()}
    enc = st.selectbox("Patient", list(labels), format_func=labels.get)
    p = b[b.encounter_id == enc].iloc[0]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"{badge(p.risk_tier)} &nbsp; <b>{p.patient_name}</b>, {num(p.age)} {p.sex} &middot; {p.unit_name}, bed {p.bed}",
                    unsafe_allow_html=True)
        st.markdown(f"**DRG:** {p.drg_description}  \n**Day of stay:** {num(p.los_so_far_days, 1)} of an expected {num(p.expected_los, 1)} "
                    f"&middot; **Predicted discharge:** {pd.to_datetime(p.predicted_discharge_ts):%a %b %d %H:%M} "
                    f"&middot; **Chance of discharge today:** {pct(p.p_discharge_today, 0)}  \n"
                    f"**Open barrier:** {p.open_barrier_category or 'none'}  \n**Risk drivers:** {p.risk_drivers or 'none recorded'}")
        note = p.huddle_note if isinstance(p.huddle_note, str) and p.huddle_note else "No AI note: notes are written for high-risk patients."
        st.markdown(f'<div class="nm-note"><b class="nm-kpi-label">AI huddle note</b><br>{note}</div>', unsafe_allow_html=True)
        if p.plan_status:
            st.caption(f"Current plan: {p.plan_status}" + (f", target {pd.to_datetime(p.target_discharge_ts):%b %d %H:%M}" if pd.notna(p.target_discharge_ts) else ""))
    with right:
        with st.form("plan", clear_on_submit=False):
            st.markdown("**Discharge plan**")
            d = st.date_input("Target date", value=pd.to_datetime(p.predicted_discharge_ts).date(), disabled=not lb_ok)
            t = st.time_input("Target time", value=dtime(14, 0), disabled=not lb_ok)
            status = st.selectbox("Status", ["Planned", "Confirmed", "Delayed", "Discharged"], index=1, disabled=not lb_ok)
            reason = st.text_input("Delay reason (if delayed)", disabled=not lb_ok)
            if st.form_submit_button("Save plan", disabled=not lb_ok, type="primary"):
                with pg_conn() as conn, conn.cursor() as cur:
                    cur.execute("""INSERT INTO capacity_ops.discharge_plan (encounter_id, target_discharge_ts, plan_status, delay_reason, updated_by)
                                   VALUES (%s, %s, %s, %s, %s)
                                   ON CONFLICT (encounter_id) DO UPDATE SET target_discharge_ts = EXCLUDED.target_discharge_ts,
                                     plan_status = EXCLUDED.plan_status, delay_reason = EXCLUDED.delay_reason,
                                     updated_by = EXCLUDED.updated_by, updated_at = now()""",
                                (int(enc), datetime.combine(d, t, tzinfo=timezone.utc), status, reason or None, user_email()))
                st.success("Plan saved")
                st.rerun()
        with st.form("action", clear_on_submit=True):
            st.markdown("**Log an action**")
            atype = st.selectbox("Action", ACTION_TYPES, disabled=not lb_ok)
            owner = st.selectbox("Owner", ["Case manager", "Social work", "Physician advisor", "Charge nurse"], disabled=not lb_ok)
            anote = st.text_input("Note", disabled=not lb_ok)
            if st.form_submit_button("Log action", disabled=not lb_ok):
                with pg_conn() as conn, conn.cursor() as cur:
                    cur.execute("""INSERT INTO capacity_ops.discharge_actions (encounter_id, action_type, note, owner_role, created_by)
                                   VALUES (%s, %s, %s, %s, %s)""", (int(enc), atype, anote or None, owner, user_email()))
                st.success("Action logged")
                st.rerun()
        if lb_ok:
            acts = pg_df("""SELECT action_type, owner_role, status, note, created_by, created_at FROM capacity_ops.discharge_actions
                            WHERE encounter_id = %s ORDER BY created_at DESC LIMIT 10""", (int(enc),))
            if len(acts):
                st.caption("Recent actions")
                st.dataframe(acts, hide_index=True, use_container_width=True)


FISCAL_YEARS = ["FY2026", "FY2025", "FY2027"]


def page_los():
    hero("LOS Performance", "Where are LOS and excess days above expected, and which barriers cause them?")
    fy = st.selectbox("Fiscal year (starts Sep 1)", FISCAL_YEARS, help="FY2027 has Sep-Oct 2026 only")
    k = sql(f"""SELECT MEASURE(`LOS O/E Index`) AS oe, MEASURE(`Excess Days`) AS excess, MEASURE(`Avoidable Days`) AS avoidable,
                       MEASURE(`Discharges Before Noon %`) AS dbn, MEASURE(`Long Stay Rate`) AS lsr, MEASURE(`ALOS`) AS alos,
                       MEASURE(`Inpatient Discharges`) AS ip
                FROM {GOLD}.los_metrics WHERE `Fiscal Year` = :fy""", fy=fy).iloc[0]
    occ = sql(f"SELECT MEASURE(`Occupancy %`) AS occ FROM {GOLD}.capacity_metrics WHERE `Fiscal Year` = :fy", fy=fy).iloc[0].occ
    kpi_row([
        ("LOS O/E index", num(k.oe, 2), "observed / expected, inpatient"),
        ("Excess days", num(k.excess), f"~{num(k.excess / 365)} beds all year" if fy != "FY2027" else "year to date"),
        ("Avoidable days", num(k.avoidable), "CarePort barriers"),
        ("Discharges before noon", pct(k.dbn), "target 30%+"),
        ("Long-stay rate", pct(k.lsr), "2+ days past expected"),
        ("Occupancy", pct(occ), "midnight census / staffed beds"),
    ])
    fac = sql(f"""SELECT `Facility` AS facility, `Facility Code` AS code, MEASURE(`LOS O/E Index`) AS oe, MEASURE(`Excess Days`) AS excess
                  FROM {GOLD}.los_metrics WHERE `Fiscal Year` = :fy GROUP BY ALL ORDER BY oe DESC""", fy=fy)
    bar = sql(f"""SELECT `Barrier Category` AS category, MEASURE(`Avoidable Days`) AS days, MEASURE(`Barriers`) AS barriers
                  FROM {GOLD}.barrier_metrics WHERE `Fiscal Year` = :fy GROUP BY ALL ORDER BY days DESC""", fy=fy)
    if len(fac) and len(bar):
        hi = fac.iloc[0]
        insight(f"<b>{hi.facility}</b> has the highest LOS O/E ({num(hi.oe, 2)}) in {fy}; system excess days total "
                f"<b>{num(k.excess)}</b>. <b>{bar.iloc[0].category}</b> is the top barrier with {num(bar.iloc[0].days)} avoidable days "
                f"({pct(bar.iloc[0].days / bar.days.sum(), 0)} of the total). Cutting excess days 15% would free about "
                f"{num(k.excess * 0.15 / 365)} staffed beds.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### LOS O/E index by hospital")
        fig = px.bar(fac, x="code", y="oe", hover_name="facility", labels={"code": "", "oe": "LOS O/E"},
                     color_discrete_sequence=[NM["primary"]], text=fac.oe.round(2))
        fig.add_hline(y=1.0, line_dash="dash", line_color=NM["bad"], annotation_text="Expected = 1.0")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        st.markdown("#### Avoidable days by barrier")
        bar["cum"] = bar.days.cumsum() / bar.days.sum() * 100
        fig = go.Figure()
        fig.add_bar(x=bar.category, y=bar.days, marker_color=NM["violet"], name="Avoidable days")
        fig.add_scatter(x=bar.category, y=bar.cum, yaxis="y2", mode="lines+markers", line_color=NM["deep"], name="Cumulative %")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False, ticksuffix="%"))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    trend = sql(f"""SELECT `Discharge Month` AS month, `Facility Code` AS code, MEASURE(`Excess Days`) AS excess,
                           MEASURE(`Discharges Before Noon %`) AS dbn
                    FROM {GOLD}.los_metrics WHERE `Discharge Month` >= DATE'2024-09-01' AND `Discharge Month` < DATE'2026-11-01'
                    GROUP BY ALL ORDER BY month""")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Monthly excess days")
        t = trend.groupby("month", as_index=False).excess.sum()
        fig = px.area(t, x="month", y="excess", labels={"month": "", "excess": "Excess days"}, color_discrete_sequence=[NM["primary"]])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        st.markdown("#### Discharges before noon")
        sel = trend[trend.code.isin(["LMH", "PCH", "BSH"])].assign(dbn=lambda x: x.dbn * 100)
        fig = px.line(sel, x="month", y="dbn", color="code", labels={"month": "", "dbn": "% before noon", "code": ""},
                      color_discrete_sequence=[NM["primary"], NM["ocean"], NM["violet"]])
        fig.add_hline(y=30, line_dash="dash", line_color=NM["good"], annotation_text="Target 30%")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with st.expander("Hospital detail"):
        st.dataframe(fac.rename(columns={"facility": "Hospital", "code": "Code", "oe": "LOS O/E", "excess": "Excess days"}).round(2),
                     hide_index=True, use_container_width=True)


SAMPLES = ["Which hospital had the highest LOS O/E index in FY2026?",
           "How many high-risk patients are in each hospital right now?",
           "Which units are forecast to have the fewest open beds tonight?",
           "What barrier category causes the most avoidable days?"]


def ask_genie(question):
    w = user_client()
    cid = st.session_state.get("genie_conv")
    if cid:
        msg = w.genie.create_message_and_wait(GENIE_SPACE_ID, cid, question)
    else:
        msg = w.genie.start_conversation_and_wait(GENIE_SPACE_ID, question)
    st.session_state.genie_conv = msg.conversation_id
    mid = getattr(msg, "message_id", None) or msg.id
    texts, tables = [], []
    for a in msg.attachments or []:
        if a.text and a.text.content:
            texts.append(a.text.content)
        if a.query:
            if a.query.description:
                texts.append(a.query.description)
            res = w.genie.get_message_attachment_query_result(GENIE_SPACE_ID, msg.conversation_id, mid, a.attachment_id)
            sr = res.statement_response
            cols = [c.name for c in sr.manifest.schema.columns]
            tables.append((pd.DataFrame(sr.result.data_array or [], columns=cols), a.query.query))
    return texts, tables


def page_ask():
    hero("Ask", "Plain-English questions about capacity, LOS and barriers. Answers come from the governed metric views.")
    st.session_state.setdefault("chat", [])
    cols = st.columns(len(SAMPLES))
    picked = None
    for c, q in zip(cols, SAMPLES):
        if c.button(q, use_container_width=True):
            picked = q
    if st.button("New conversation"):
        st.session_state.chat, st.session_state.genie_conv = [], None
        st.rerun()
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            for t in m.get("texts", []):
                st.markdown(t)
            for df, q in m.get("tables", []):
                st.dataframe(df, hide_index=True, use_container_width=True)
                with st.expander("SQL"):
                    st.code(q, language="sql")
    question = st.chat_input("Ask about beds, LOS, barriers...") or picked
    if question:
        st.session_state.chat.append({"role": "user", "texts": [question]})
        with st.spinner("Genie is working..."):
            try:
                texts, tables = ask_genie(question)
            except Exception as e:
                texts, tables = [f"Genie couldn't answer: {e}"], []
        st.session_state.chat.append({"role": "assistant", "texts": texts or ["(no text answer)"], "tables": tables})
        st.rerun()


# ---------------------------------------------------------------- shell
PAGES = {"Command Center": page_command_center, "Discharge Worklist": page_worklist,
         "LOS Performance": page_los, "Ask": page_ask}
with st.sidebar:
    st.markdown(f"<h3 style='margin-bottom:0'>LSH Capacity</h3><div class='nm-kpi-sub'>Command Center &middot; synthetic data</div>",
                unsafe_allow_html=True)
    st.write("")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.write("")
    st.caption(f"Signed in as {user_email()}")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
try:
    PAGES[choice]()
except Exception as e:
    if type(e).__name__ in ("RerunException", "StopException"):   # st.rerun() / st.stop() control flow
        raise
    st.error(f"Couldn't load this page: {e}")
    st.caption("If this is a permission error, check the app's user authorization scopes (sql, dashboards.genie) "
               "and that you have access to the SQL warehouse, the Genie space and the gold schema.")
