from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pid_core import PIDTuning, SimulationConfig, simulate_closed_loop, FOPDTModel
from robust_processing import (
    analyze_loop,
    OUTLIER_METHOD_LABELS,
    MODEL_LABELS,
    TUNING_METHOD_LABELS,
    controller_convention_text,
)
from reporting import build_report_docx


st.set_page_config(page_title="PID Tuner Pro v8", page_icon="🎛️", layout="wide")
st.title("🎛️ PID Tuner Pro v8 — versão de campo")
st.caption(
    "Diagnóstico robusto de malhas, múltiplos métodos de outlier, múltiplos modelos de processo, "
    "comparação gráfica da PV prevista e teste manual de sintonia."
)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "last_entry" not in st.session_state:
    st.session_state.last_entry = None
if "last_report_bytes" not in st.session_state:
    st.session_state.last_report_bytes = None
if "manual_tuning" not in st.session_state:
    st.session_state.manual_tuning = None
if "current_pid_tuning" not in st.session_state:
    st.session_state.current_pid_tuning = None


# ---------- helper functions ----------
def _abs_sim_frame(entry, payload):
    sim = payload["simulation"]
    baseline_pv = float(payload.get("baseline_pv", entry.clean_data["pv"].median()))
    baseline_sp = float(entry.clean_data["sp"].median()) if "sp" in entry.clean_data.columns else baseline_pv
    df = sim.data.copy()
    out = pd.DataFrame({
        "time_min": df["time"] / 60.0,
        "pv_abs": df["pv"] + baseline_pv,
        "sp_abs": df["sp"] + baseline_sp,
        "mv": df["mv"],
        "error": (df["sp"] + baseline_sp) - (df["pv"] + baseline_pv),
        "p_term": df.get("p_term", 0.0),
        "i_term": df.get("i_term", 0.0),
        "d_term": df.get("d_term", 0.0),
    })
    return out


def _make_raw_clean_figure(entry):
    raw = entry.raw_data.copy()
    clean = entry.clean_data.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw["time"], y=raw["pv"], name="PV bruta", opacity=0.45, line=dict(width=1)))
    fig.add_trace(go.Scatter(x=clean["time"], y=clean["pv"], name="PV limpa", line=dict(width=2)))
    if "sp" in raw.columns:
        fig.add_trace(go.Scatter(x=raw["time"], y=raw["sp"], name="SP", line=dict(width=1.5, dash="dash")))
    if "outlier_flag" in raw.columns:
        mask = raw["outlier_flag"] == 1
        if mask.any():
            fig.add_trace(go.Scatter(
                x=raw.loc[mask, "time"],
                y=raw.loc[mask, "pv"],
                mode="markers",
                name="Pontos sinalizados",
                marker=dict(size=6, symbol="x"),
            ))
    fig.update_layout(height=430, title="PV bruta x PV limpa, com pontos sinalizados", xaxis_title="Tempo", yaxis_title="PV")
    return fig


def _simulate_single(entry, tuning: PIDTuning, step_size: float, horizon_s: float):
    eq = entry.equivalent_model
    direct_action = bool(eq.gain >= 0)
    result = simulate_closed_loop(
        eq.to_fopdt(),
        tuning,
        SimulationConfig(
            horizon=horizon_s,
            dt=1.0,
            setpoint=step_size,
            direct_action=direct_action,
        ),
    )
    return {
        "tuning": tuning,
        "simulation": result,
        "direct_action": direct_action,
        "error_formula": "e = SP - PV" if direct_action else "e = PV - SP",
        "baseline_pv": float(entry.clean_data["pv"].median()) if len(entry.clean_data) else 0.0,
    }


def _build_comparison_payloads(entry, step_size, horizon_s, current_pid_enabled, current_mode, cur_vals):
    payloads = dict(entry.simulation_results)

    if current_pid_enabled:
        if current_mode == "time_constants":
            tuning = PIDTuning.from_time_constants(
                name="PID atual da malha",
                controller_type=cur_vals["controller_type"],
                kc=cur_vals["kc"],
                ti=cur_vals["ti"],
                td=cur_vals["td"],
                method="Atual",
            )
        else:
            tuning = PIDTuning.from_parallel_gains(
                name="PID atual da malha",
                controller_type=cur_vals["controller_type"],
                kc=cur_vals["kc"],
                ki=cur_vals["ki"],
                kd=cur_vals["kd"],
                method="Atual",
            )
        payloads[tuning.name] = _simulate_single(entry, tuning, step_size, horizon_s)
        st.session_state.current_pid_tuning = tuning

    if st.session_state.manual_tuning is not None:
        payloads[st.session_state.manual_tuning.name] = _simulate_single(entry, st.session_state.manual_tuning, step_size, horizon_s)

    return payloads


def _comparison_metrics_df(payloads):
    rows = []
    for name, payload in payloads.items():
        tuning = payload["tuning"]
        metrics = payload["simulation"].metrics
        rows.append({
            "Método": name,
            "Família": getattr(tuning, "method", "-"),
            "Tipo": tuning.controller_type,
            "Kc (ganho do controlador)": tuning.kc,
            "Ti (s)": None if math.isinf(tuning.ti) else tuning.ti,
            "Ki (1/s)": tuning.ki,
            "Td (s)": tuning.td,
            "Kd": tuning.kd,
            "Overshoot (%)": metrics["overshoot_pct"],
            "Settling (s)": metrics["settling_time"],
            "IAE": metrics["IAE"],
            "ISE": metrics["ISE"],
            "ITAE": metrics["ITAE"],
            "TV(MV)": metrics["total_variation_mv"],
        })
    return pd.DataFrame(rows)


def _plot_pv_comparison(entry, payloads, selected_names):
    fig = go.Figure()
    baseline_pv = float(entry.clean_data["pv"].median()) if len(entry.clean_data) else 0.0
    baseline_sp = float(entry.clean_data["sp"].median()) if "sp" in entry.clean_data.columns else baseline_pv
    for name in selected_names:
        frame = _abs_sim_frame(entry, payloads[name])
        fig.add_trace(go.Scatter(x=frame["time_min"], y=frame["pv_abs"], name=name))
    if selected_names:
        frame0 = _abs_sim_frame(entry, payloads[selected_names[0]])
        fig.add_trace(go.Scatter(x=frame0["time_min"], y=frame0["sp_abs"], name="SP previsto", line=dict(dash="dash", width=2)))
    fig.add_hline(y=baseline_pv, line_dash="dot", annotation_text="PV de referência")
    fig.update_layout(
        height=470,
        title="Como a PV deve se comportar para cada sintonia",
        xaxis_title="Tempo (min)",
        yaxis_title="PV em unidades reais",
        legend_title="Sintonia",
    )
    return fig


def _plot_mv_comparison(entry, payloads, selected_names):
    fig = go.Figure()
    for name in selected_names:
        frame = _abs_sim_frame(entry, payloads[name])
        fig.add_trace(go.Scatter(x=frame["time_min"], y=frame["mv"], name=name))
    fig.update_layout(
        height=380,
        title="Esforço da MV previsto para cada sintonia",
        xaxis_title="Tempo (min)",
        yaxis_title="MV",
    )
    return fig


def _plot_error_comparison(entry, payloads, selected_names):
    fig = go.Figure()
    for name in selected_names:
        frame = _abs_sim_frame(entry, payloads[name])
        fig.add_trace(go.Scatter(x=frame["time_min"], y=frame["error"], name=name))
    fig.update_layout(
        height=320,
        title="Erro SP - PV previsto",
        xaxis_title="Tempo (min)",
        yaxis_title="Erro",
    )
    return fig


with st.sidebar:
    st.header("Fluxo recomendado")
    st.write(
        "1. Carregue um histórico.\n"
        "2. Mapeie Tempo, PV, SP e MV.\n"
        "3. Escolha a limpeza de outliers.\n"
        "4. Compare os modelos.\n"
        "5. Compare as sintonias.\n"
        "6. Ajuste Kc, Ti e Td manualmente se quiser.\n"
        "7. Gere o relatório Word."
    )
    st.markdown("---")
    st.write(
        "**O que a v8 acrescenta**\n"
        "- gráfico principal em unidades reais\n"
        "- MV prevista por método\n"
        "- teste manual de sintonia\n"
        "- convenção do erro mostrada explicitamente\n"
        "- relatório de campo mais detalhado"
    )

uploaded = st.file_uploader("Carregue um CSV ou Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Carregue um arquivo para começar.")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
raw_df = pd.read_excel(uploaded) if suffix in [".xlsx", ".xls"] else pd.read_csv(uploaded)
st.dataframe(raw_df.head(10), use_container_width=True)

columns = list(raw_df.columns)
st.subheader("1) Mapeamento das colunas")
c1, c2, c3, c4 = st.columns(4)
time_col = c1.selectbox("Tempo", columns, index=0)
pv_col = c2.selectbox("PV", columns, index=min(1, len(columns) - 1))
sp_col = c3.selectbox("SP", ["<não usar>"] + columns, index=min(3, len(columns)))
mv_col = c4.selectbox("MV", ["<não usar>"] + columns, index=min(4, len(columns)))
sp_col = None if sp_col == "<não usar>" else sp_col
mv_col = None if mv_col == "<não usar>" else mv_col

st.subheader("2) Metadados da visita")
m1, m2, m3, m4 = st.columns(4)
client = m1.text_input("Cliente", "Klabin")
site = m2.text_input("Planta", "Puma")
loop_name = m3.text_input("Nome da malha", "Malha 01")
loop_tag = m4.text_input("Tag", "TAG-001")
m5, m6, m7 = st.columns(3)
loop_type = m5.selectbox("Tipo de malha", ["flow", "pressure", "consistency", "specific_energy", "level", "basis_weight", "moisture", "other"], index=2)
controller_type = m6.selectbox("Tipo de controlador para as propostas", ["PI", "PID"], index=0)
step_size = m7.number_input("Degrau de SP para a simulação", value=0.02, format="%.4f")

st.subheader("3) Limpeza de outliers e modelagem")
a1, a2, a3, a4 = st.columns(4)
outlier_method = a1.selectbox("Método de outliers", list(OUTLIER_METHOD_LABELS.keys()), format_func=lambda x: OUTLIER_METHOD_LABELS[x], index=0)
resample_rule = a2.selectbox("Reamostragem para modelagem", ["30s", "60s", "120s"], index=1)
apply_to = a3.selectbox("Aplicar filtro em", ["pv_mv", "pv", "mv", "none"], index=0, format_func=lambda x: {"pv_mv": "PV e MV", "pv": "PV", "mv": "MV", "none": "Nenhum"}[x])
model_family = a4.selectbox("Modelo usado para sintonia", list(MODEL_LABELS.keys()), format_func=lambda x: MODEL_LABELS[x], index=0)

b1, b2, b3, b4 = st.columns(4)
outlier_strategy = b1.selectbox("Tratamento dos pontos sinalizados", ["interpolate", "drop"], format_func=lambda x: "Interpolar" if x == "interpolate" else "Excluir", index=0)
aggressiveness = b2.slider("Intensidade da limpeza", min_value=0.5, max_value=3.0, value=1.6, step=0.1)
passes = b3.selectbox("Número de passadas", [1, 2, 3], index=2)
tuning_selection = b4.selectbox("Métodos de sintonia", list(TUNING_METHOD_LABELS.keys()), format_func=lambda x: TUNING_METHOD_LABELS[x], index=0)

c1, c2, c3 = st.columns(3)
mv_min = c1.number_input("Limite mínimo da MV", value=0.0, format="%.3f")
mv_max = c2.number_input("Limite máximo da MV", value=100.0, format="%.3f")
sim_horizon_min = c3.number_input("Horizonte de simulação (min)", value=60.0, min_value=5.0, step=5.0)

st.subheader("4) PID atual da malha — opcional")
pid_enabled = st.checkbox("Incluir PID atual na comparação", value=False)
pid_mode = st.radio("Forma dos parâmetros", ["time_constants", "parallel_gains"], horizontal=True, format_func=lambda x: "Kc, Ti, Td" if x == "time_constants" else "Kc, Ki, Kd")
p1, p2, p3, p4 = st.columns(4)
cur_controller_type = p1.selectbox("Tipo do PID atual", ["PI", "PID"], index=0)
if pid_mode == "time_constants":
    cur_kc = p2.number_input("Kc atual", value=1.0, format="%.6f")
    cur_ti = p3.number_input("Ti atual (s)", value=100.0, format="%.3f")
    cur_td = p4.number_input("Td atual (s)", value=0.0, format="%.3f")
    current_vals = {"controller_type": cur_controller_type, "kc": cur_kc, "ti": cur_ti, "td": cur_td}
else:
    cur_kc = p2.number_input("Kc atual", value=1.0, format="%.6f")
    cur_ki = p3.number_input("Ki atual (1/s)", value=0.01, format="%.6f")
    cur_kd = p4.number_input("Kd atual", value=0.0, format="%.6f")
    current_vals = {"controller_type": cur_controller_type, "kc": cur_kc, "ki": cur_ki, "kd": cur_kd}

run = st.button("Rodar análise", use_container_width=True)

if run:
    meta = {
        "client": client,
        "site": site,
        "loop_name": loop_name,
        "loop_tag": loop_tag,
        "loop_type": loop_type,
        "controller_type": controller_type,
        "resample_rule": resample_rule,
    }
    entry = analyze_loop(
        raw_df=raw_df,
        meta=meta,
        time_col=time_col,
        pv_col=pv_col,
        sp_col=sp_col,
        mv_col=mv_col,
        outlier_method=outlier_method,
        resample_rule=resample_rule,
        apply_to=apply_to,
        model_family=model_family,
        tuning_selection=tuning_selection,
        controller_type=controller_type,
        step_size=step_size,
        mv_limits=(mv_min, mv_max),
        aggressiveness=aggressiveness,
        passes=passes,
        outlier_strategy=outlier_strategy,
    )
    st.session_state.last_entry = entry
    st.session_state.last_report_bytes = None

entry = st.session_state.last_entry
if entry is not None:
    st.subheader("Resultado")
    conv = controller_convention_text(entry.equivalent_model.gain)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nota", f"{entry.loop_score:.1f}/10")
    c2.metric("Classificação", entry.grade)
    c3.metric("Outlier selecionado", OUTLIER_METHOD_LABELS.get(entry.selected_outlier_method, entry.selected_outlier_method))
    c4.metric("Modelo-base", entry.selected_model_name)
    c5.metric("Convenção do erro", conv["error_formula"])

    st.info(conv["note"])

    tabs = st.tabs(["Resumo executivo", "Qualidade do dado", "Modelagem", "Sintonia e simulação", "Relatório"])

    with tabs[0]:
        st.markdown("**Principais razões da nota**")
        for reason in entry.reasons:
            st.write("- " + reason)
        if entry.strengths:
            st.markdown("**Pontos positivos**")
            for strength in entry.strengths:
                st.write("- " + strength)
        met = entry.performance_metrics
        cols = st.columns(6)
        cols[0].metric("RMSE SP-PV", f"{met['rmse']:.4f}")
        cols[1].metric("Desvio padrão PV", f"{met['pv_std']:.4f}")
        cols[2].metric("Mov. médio MV", f"{met['mv_move_mean']:.4f}")
        cols[3].metric("PV em ±0,05", f"{met['within_005_pct']:.1f}%")
        cols[4].metric("PV em ±0,10", f"{met['within_010_pct']:.1f}%")
        cols[5].metric("Saturação MV", f"{0 if pd.isna(met['sat_frac_pct']) else met['sat_frac_pct']:.2f}%")

    with tabs[1]:
        st.markdown("### Dados brutos vs limpos")
        st.plotly_chart(_make_raw_clean_figure(entry), use_container_width=True)
        outlier_df = pd.DataFrame([m.__dict__ for m in entry.outlier_methods])
        outlier_df["method"] = outlier_df["method"].map(lambda x: OUTLIER_METHOD_LABELS.get(x, x))
        outlier_df = outlier_df.rename(columns={
            "method": "Método",
            "flagged_pct": "% sinalizado",
            "flagged_points": "Pontos sinalizados",
            "validation_rmse": "RMSE validação",
            "validation_fit_pct": "FIT validação (%)",
            "validation_r2": "R² validação",
        })
        st.dataframe(outlier_df, use_container_width=True)
        st.caption(
            f"Configuração usada: intensidade={entry.outlier_config['aggressiveness']:.1f}; "
            f"passadas={entry.outlier_config['passes']}; tratamento={'Interpolar' if entry.outlier_config['strategy']=='interpolate' else 'Excluir'}; "
            f"aplicar em={entry.outlier_config['apply_to']}."
        )

    with tabs[2]:
        model_df = pd.DataFrame([
            {
                "Modelo": m.model_name,
                "RMSE validação": m.validation_rmse,
                "FIT validação (%)": m.validation_fit_pct,
                "R² validação": m.validation_r2,
            }
            for m in entry.model_results
        ])
        st.dataframe(model_df, use_container_width=True)

        validation_tabs = st.tabs([m.model_name for m in entry.model_results])
        for tab, model_result in zip(validation_tabs, entry.model_results):
            with tab:
                dfv = entry.model_validation_series[model_result.model_name]
                fig_val = go.Figure()
                fig_val.add_trace(go.Scatter(x=dfv["time"], y=dfv["pv_real"], name="PV real", line=dict(width=2)))
                fig_val.add_trace(go.Scatter(x=dfv["time"], y=dfv["pv_pred"], name="PV prevista", line=dict(width=2)))
                fig_val.update_layout(
                    height=420,
                    xaxis_title="Tempo",
                    yaxis_title="PV",
                    title=f"PV real x PV prevista — {model_result.model_name}",
                )
                st.plotly_chart(fig_val, use_container_width=True)
                st.caption(
                    f"RMSE={model_result.validation_rmse:.5f} | FIT={model_result.validation_fit_pct:.2f}% | R²={model_result.validation_r2:.3f}"
                )

        st.markdown("### Modelo equivalente usado para sintonia")
        st.json({
            "source_model": entry.equivalent_model.source_model,
            "process_gain_signed": entry.equivalent_model.gain,
            "process_gain_magnitude": abs(entry.equivalent_model.gain),
            "tau_seconds": entry.equivalent_model.tau,
            "dead_time_seconds": entry.equivalent_model.dead_time,
            "recommended_error_formula": conv["error_formula"],
        })

    with tabs[3]:
        base_tune_df = entry.tuning_table.copy()
        if not base_tune_df.empty:
            show_df = base_tune_df[["name", "method", "controller_type", "kc", "pb_percent_if_normalized", "ti_seconds", "ki_per_s", "td"]].rename(
                columns={
                    "name": "Método",
                    "method": "Família",
                    "controller_type": "Tipo",
                    "kc": "Kc (ganho do controlador)",
                    "pb_percent_if_normalized": "PB equivalente (%) se normalizado",
                    "ti_seconds": "Ti (s)",
                    "ki_per_s": "Ki (1/s)",
                    "td": "Td (s)",
                }
            )
            st.dataframe(show_df, use_container_width=True)
        st.caption(
            "Kc é ganho do controlador, não banda proporcional. Ti e Td estão em segundos. Ki = Kc/Ti em 1/s. "
            "PB equivalente = 100/Kc só deve ser usada quando PV e MV estiverem normalizadas em %."
        )

        st.markdown("### Teste manual de sintonia")
        if not entry.tuning_table.empty:
            default_row = entry.tuning_table.iloc[0]
            default_kc = float(default_row["kc"])
            default_ti = float(default_row["ti_seconds"])
            default_td = float(default_row["td"])
        else:
            default_kc, default_ti, default_td = 1.0, 100.0, 0.0
        q1, q2, q3, q4, q5 = st.columns(5)
        manual_name = q1.text_input("Nome do teste manual", value="Teste manual")
        manual_type = q2.selectbox("Tipo do controlador manual", ["PI", "PID"], index=0)
        manual_kc = q3.number_input("Kc manual", value=default_kc, format="%.6f")
        manual_ti = q4.number_input("Ti manual (s)", value=default_ti, format="%.3f")
        manual_td = q5.number_input("Td manual (s)", value=0.0 if manual_type == "PI" else default_td, format="%.3f")
        add_manual = st.button("Adicionar/atualizar teste manual")
        if add_manual:
            st.session_state.manual_tuning = PIDTuning.from_time_constants(
                name=manual_name,
                controller_type=manual_type,
                kc=manual_kc,
                ti=manual_ti,
                td=manual_td,
                method="Manual",
            )
            st.success("Teste manual adicionado à comparação.")

        payloads = _build_comparison_payloads(
            entry,
            step_size=step_size,
            horizon_s=float(sim_horizon_min) * 60.0,
            current_pid_enabled=pid_enabled,
            current_mode=pid_mode,
            cur_vals=current_vals,
        )
        available_names = list(payloads.keys())
        selected_names = st.multiselect("Quais curvas você quer comparar?", available_names, default=available_names)
        if selected_names:
            st.plotly_chart(_plot_pv_comparison(entry, payloads, selected_names), use_container_width=True)
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(_plot_mv_comparison(entry, payloads, selected_names), use_container_width=True)
            with g2:
                st.plotly_chart(_plot_error_comparison(entry, payloads, selected_names), use_container_width=True)
            metrics_df = _comparison_metrics_df({k: payloads[k] for k in selected_names})
            st.dataframe(metrics_df, use_container_width=True)

    with tabs[4]:
        c1, c2, c3 = st.columns(3)
        if c1.button("Adicionar ao portfólio", use_container_width=True):
            st.session_state.portfolio.append(entry)
            st.success("Malha adicionada ao portfólio.")
        if c2.button("Gerar relatório desta malha", use_container_width=True):
            tmp = Path("report_single_v8.docx")
            build_report_docx([entry], tmp)
            st.session_state.last_report_bytes = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
        if c3.button("Gerar relatório do portfólio", use_container_width=True):
            package = st.session_state.portfolio if st.session_state.portfolio else [entry]
            tmp = Path("report_portfolio_v8.docx")
            build_report_docx(package, tmp)
            st.session_state.last_report_bytes = tmp.read_bytes()
            tmp.unlink(missing_ok=True)

if st.session_state.last_report_bytes is not None:
    st.download_button(
        "Baixar relatório Word",
        st.session_state.last_report_bytes,
        file_name="pid_tuner_report_v8.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
