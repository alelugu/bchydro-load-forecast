"""BC Hydro demand forecasting dashboard (Gradio).

    python app.py                 # http://127.0.0.1:7860

On Hugging Face Spaces, set DB_URL to the published database (see README) and the app
downloads a fresh copy when its copy is older than DB_MAX_AGE_HOURS.
"""
import os

import gradio as gr
import pandas as pd

from bchydro import config
from bchydro import dashboard as D

D.refresh()

VERSIONS = [D.VERSION_LABEL[v] for v in ["retrained", "bias_corrected", "frozen", "baseline"]]
CITIES = ["Provincial (population-weighted)", "Vancouver", "Victoria", "Kelowna", "Prince George"]
CSS = """
.gradio-container {max-width: 1280px !important}
#status {font-size: 0.92em; color: #4b5563}
.kpi table {width: 100%} .kpi td {font-size: 1.15em}
"""


def _last_actual_day() -> str:
    return f"{D.data()['load'].index.max().tz_convert(config.LOCAL_TZ):%Y-%m-%d}"


with gr.Blocks(title="BC Hydro demand forecast") as demo:
    gr.Markdown("# BC Hydro demand forecast\n"
                "Day-ahead hourly load for the BC Hydro grid, for grid operations and load-forecasting analysts. "
                "Forecasts are issued every day for the next day and correct themselves as BC Hydro publishes actuals.")
    status = gr.Markdown(elem_id="status")

    with gr.Row():
        watch_pct = gr.Slider(50, 95, value=75, step=1, label="Watch threshold (percentile of past winter daily peaks)")
        high_pct = gr.Slider(80, 99.5, value=95, step=0.5, label="High threshold (percentile of past winter daily peaks)")

    with gr.Tab("Tomorrow"):
        with gr.Row():
            day = gr.Dropdown(label="Forecast day", choices=[], scale=2)
            version = gr.Dropdown(VERSIONS, value=VERSIONS[0], label="Model version", scale=2)
            shift = gr.Slider(-10, 5, value=0, step=1, label="What if it were this much colder/warmer (°C)", scale=3)
        kpi = gr.Markdown(elem_classes="kpi")
        plot_t = gr.Plot(show_label=False)
        actions = gr.Markdown()
        gr.Markdown("Shaded green hours are the six lowest-demand hours: candidates for exports. "
                    "The what-if line reruns the model with every temperature shifted, keeping the corrections.")
        dl_forecast = gr.DownloadButton("Download this forecast and the 7-day outlook (Excel)", size="sm")

    with gr.Tab("Peak risk (7 days)"):
        temp_src = gr.Dropdown(CITIES, value=CITIES[0], label="Temperature shown")
        risk_msg = gr.Markdown()
        plot_r = gr.Plot(show_label=False)
        risk_table = gr.Dataframe(interactive=False, wrap=True)

    with gr.Tab("Explore history"):
        with gr.Row():
            h_start = gr.Textbox("2025-11-01", label="From (YYYY-MM-DD)")
            h_end = gr.Textbox(label="To (YYYY-MM-DD)")
            grain = gr.Radio(["Hourly", "Daily", "Weekly"], value="Daily", label="Granularity")
        with gr.Row():
            day_types = gr.CheckboxGroup(["Weekday", "Weekend", "Holiday"], value=["Weekday", "Weekend", "Holiday"],
                                         label="Day types")
            city = gr.Dropdown(CITIES, value=CITIES[0], label="Temperature")
        h_note = gr.Markdown()
        plot_h1 = gr.Plot(show_label=False)
        with gr.Row():
            plot_h2 = gr.Plot(show_label=False)
            plot_h3 = gr.Plot(show_label=False)

    with gr.Tab("Prediction correction"):
        gr.Markdown("The frozen model was trained once, up to 15 Nov 2025. The other versions correct "
                    "themselves: an hourly bias correction every day, and a retrain on Sundays or when a "
                    "trigger fires (accepted only if the refit beats the current model on the last 14 days). "
                    "**Replay** re-runs the year after the cutoff one day at a time, revealing each day's "
                    "actuals only when BC Hydro would have published them; **Live** is the daily job since then.")
        with gr.Row():
            phase = gr.Radio(["Both", "Replay", "Live"], value="Both", label="Phase")
            c_start = gr.Textbox("2025-11-16", label="From (YYYY-MM-DD)")
            c_end = gr.Textbox(label="To (YYYY-MM-DD)")
        with gr.Row():
            c_versions = gr.CheckboxGroup(VERSIONS[:3], value=VERSIONS[:3], label="Versions (first = heatmap and alerts)")
            metric = gr.Dropdown(["MAPE (%)", "MAE (MW)", "Peak error (%)"], value="MAPE (%)", label="Metric")
        c_note = gr.Markdown()
        gr.Markdown("### Live forecasts so far\nEach row is a forecast the daily job issued before the day happened, "
                    "scored once BC Hydro published the actual load.")
        live_tbl = gr.Dataframe(interactive=False, wrap=True)
        summary = gr.Dataframe(interactive=False, label="Accuracy over the selected period")
        dl_correction = gr.DownloadButton("Download these results (Excel)", size="sm")
        plot_c2 = gr.Plot(show_label=False)
        plot_c1 = gr.Plot(show_label=False)
        plot_c3 = gr.Plot(show_label=False)
        gr.Markdown("### Alert accuracy\nDays classed by peak: did the forecast flag the days that turned out "
                    "Watch or High? **Precision** = flagged days that were real; **recall** = real days that were "
                    "flagged. Few days reach High, so read the counts beside the ratios.")
        cls_table = gr.Dataframe(interactive=False)
        plot_cm = gr.Plot(show_label=False)
        gr.Markdown("### Model log")
        log = gr.Dataframe(interactive=False)

    with gr.Tab("About"):
        val = gr.Dataframe(interactive=False, label="Model selection on 2024 (trained 2015-2023)")
        gr.Markdown("""
**Audience.** BC Hydro grid operations and load-forecasting analysts, who decide a day ahead how dams,
imports and exports will cover demand. More than 90% of BC Hydro's generation is hydro; reservoir water is
limited in dry years, so forecasting demand well saves water and avoids short-notice imports.

**Data.** BC Hydro hourly balancing-authority load (2015 on, daily updates); Open-Meteo weather for Vancouver,
Victoria, Kelowna and Prince George (forecasts as issued from 2022, ERA5 before); BC statutory holidays.
All in one SQLite database, refreshed by a GitHub Actions job at 09:30 and 13:30 Pacific.

**Model.** LightGBM on calendar, temperature, heating and cooling degree days, and load at least 7 days old
(what is published by 09:30 even on weekends), with 10th/90th percentile models for the band. Trained on
2015-2023 and checked on 2024 using forecast weather, as in live use.

**Correction.** Hourly bias from recent errors (7-day half-life, capped at ±3%); retrain on Sundays, or when a
daily peak misses by more than 5%, errors lean one way for 5 days by more than 2%, or 7-day MAPE exceeds
1.25 × the validation MAPE twice; band widened or narrowed when 14-day coverage leaves 70-90%.

**Limits.** Experimental forecasts, not BC Hydro's. The archived weather forecasts are short-range, so replay
scores are slightly better than a true day-ahead run would be. A day is also High when its peak is within 3%
of the highest hour on record in this data.
""")

    # ------------------------------------------------------------ wiring
    def on_load():
        D.refresh()
        days = D.forecast_days()
        last = _last_actual_day()
        return (D.status_line(), gr.update(choices=days, value=days[0] if days else None), last, last,
                D.model_log(), D.validation_table(), D.live_days_table())

    t_inputs = [day, version, shift, watch_pct, high_pct]
    r_inputs = [watch_pct, high_pct, temp_src]
    h_inputs = [h_start, h_end, grain, day_types, city]
    c_inputs = [phase, c_start, c_end, c_versions, metric, watch_pct, high_pct]
    t_out, r_out = [kpi, plot_t, actions, dl_forecast], [plot_r, risk_table, risk_msg]
    h_out = [plot_h1, plot_h2, plot_h3, h_note]
    c_out = [plot_c1, plot_c2, plot_c3, c_note, summary, cls_table, plot_cm, dl_correction]

    def tomorrow_view(day_label, version_label, shift, watch, high):
        """The Tomorrow tab plus the Excel file behind its download button."""
        return (*D.tomorrow(day_label, version_label, shift, watch, high),
                D.export_forecast(day_label, version_label, watch, high))

    def correction_view(*args):
        return (*D.correction(*args), D.export_correction(*args))

    def all_views(*args):
        t = tomorrow_view(*args[:5])
        r = D.peak_risk(*args[5:8])
        h = D.history(*args[8:13])
        c = correction_view(*args[13:20])
        return (*t, *r, *h, *c)

    demo.load(on_load, outputs=[status, day, h_end, c_end, log, val, live_tbl]).then(
        all_views, inputs=t_inputs + r_inputs + h_inputs + c_inputs, outputs=t_out + r_out + h_out + c_out)
    for comp in t_inputs:
        comp.change(tomorrow_view, t_inputs, t_out)
    for comp in r_inputs:
        comp.change(D.peak_risk, r_inputs, r_out)
    for comp in [h_start, h_end]:
        comp.submit(D.history, h_inputs, h_out)
    for comp in [grain, day_types, city]:
        comp.change(D.history, h_inputs, h_out)
    for comp in [c_start, c_end]:
        comp.submit(correction_view, c_inputs, c_out)
    for comp in [phase, c_versions, metric, watch_pct, high_pct]:
        comp.change(correction_view, c_inputs, c_out)


if __name__ == "__main__":
    # No fixed address: Gradio listens on 127.0.0.1 locally and on the address Hugging Face
    # Spaces sets (GRADIO_SERVER_NAME / GRADIO_SERVER_PORT). HOST and PORT still override.
    demo.launch(server_name=os.getenv("HOST") or None,
                server_port=int(os.environ["PORT"]) if os.getenv("PORT") else None,
                theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
