from __future__ import annotations
import json
import re
import html as htmlmod
from typing import List
from .charts import compute_payment_grid, chart_sale_likelihood
from .helpers import fmt_pct, epc_letter_from_score

from .helpers import (
    esc, fmt_currency, fmt_int, fmt_km2, fmt_km_mi, fmt_km_mi_html, fmt_m2_ft2, fmt_m2_ft2_html,
    fmt_ppm2_with_ft2_html, pretty_date, round_ppm2, round_price, to_float
)
from .charts import chart_price_interval, chart_shap, chart_trend_toggle, chart_borrow_rates

# ---------- blocks ----------

def property_icons_block(info: dict) -> str:
    epc = info.get('energy_eff','-')
    rows = [
        ("📍 Address", info.get("full_address", "-")),
        ("🏗️ Built", info.get("built_date", "-")),
        ("🔑 Tenure", info.get("tenure", "-")),
        ("🏢 Type", f"{info.get('property_type','-')} · {info.get('built_form','-')}"),
        ("📐 Floor area", fmt_m2_ft2(info.get("floor_area"))),
        ("📐 Floor", fmt_int(info.get("floor_level"))),
        ("🛏️ Bedrooms", fmt_int(max(1, info.get("num_rooms") - 1))),
        ("🍃 EPC", f"{fmt_int(epc)} ({epc_letter_from_score(epc)})"),
        ("🌳 Park", fmt_km_mi(info.get("dist_to_park"))),
        ("🚇 Tube", fmt_km_mi(info.get("dist_to_tube"))),
        ("🏫 School", fmt_km_mi(info.get("dist_to_school"))),
    ]
    items = "".join(
        "<div class='prop-item'>"
        f"<div class='icon'>{htmlmod.escape(label)}</div>"
        f"<div class='val'>{htmlmod.escape(str(val))}</div>"
        "</div>"
        for label, val in rows
    )
    return (
        "<section>"
        "<h2>Property info</h2>"
        "<div class='prop-grid'>"
        f"{items}"
        "</div>"
        "</section>"
    )

def median_note():
    return """
    <section>
      <p class="muted">
        In this report, prices are often shown using the <b>median</b>.
        The median is the middle value: half of the comparable properties sold for less,
        and half sold for more. This makes it more reliable than an average, which can
        be skewed by unusually high or low sales.
      </p>
    </section>
    """


def comps_table_block(comps: list, base_value: float, confidence: str) -> str:
    base_value = round_price(base_value)

    header = (
        "<section>"
        f"<p><b>Base price:</b> {fmt_currency(base_value)} &nbsp;·&nbsp; "
        f"<b>Comps confidence:</b> {htmlmod.escape(str(confidence)).capitalize()}</p>"
        "<table class='comps'>"
        "<thead><tr>"
        "<th>Address</th><th>Date</th><th>Price</th>"
        "<th>£/m² ( £/ft² )</th>"
        "<th>Type</th><th>EPC</th><th>Rooms</th>"
        "<th>Size</th>"
        "<th>Floor</th>"
        "<th>Dist (km / mi)</th>"
        # "<th>Similarity</th>"
        "</tr></thead><tbody>"
    )

    rows_html: List[str] = []
    for i, c in enumerate(comps):
        price = round_price(c.get("price") or 0)
        ppm2 = round_ppm2(to_float(c.get("price_per_m2"), 0) or 0)
        epc = c.get("energy_eff") or c.get("epc") or c.get("energy_rating") or "-"
        date_raw = c.get("date")
        date_txt = pretty_date(date_raw) if date_raw else "-"
        similarity = fmt_pct(c.get("similarity"))

        # hide rows after the 5th by default
        row_class = " class='extra-comp' style='display:none;'" if i >= 5 else ""
        rows_html.append(
            f"<tr{row_class}>"
            f"<td>{htmlmod.escape(str(c.get('full_address', '-')))}</td>"
            f"<td>{htmlmod.escape(date_txt)}</td>"
            f"<td>{fmt_currency(price)}</td>"
            f"<td>{fmt_ppm2_with_ft2_html(ppm2)}</td>"
            f"<td>{htmlmod.escape(str(c.get('property_type', '-')))}</td>"
            f"<td>{htmlmod.escape(fmt_int(epc))} ({epc_letter_from_score(epc)})</td>"
            f"<td>{fmt_int(c.get('num_rooms'))}</td>"
            f"<td>{fmt_m2_ft2_html(c.get('floor_area'))}</td>"
            f"<td>{fmt_int(c.get('floor_level'))}</td>"
            f"<td>{fmt_km_mi_html(c.get('geo_dist_km'))}</td>"
            # f"<td>{htmlmod.escape(similarity)}</td>"
            "</tr>"
        )

    footer = (
        "</tbody></table>"
        # arrow toggle
        "<div id='toggle-comps' style='margin-top:8px; cursor:pointer; "
        "display:inline-flex; align-items:center; gap:6px; color:#23b4c8; font-weight:600;'>"
        "<span>Show more comparables</span>"
        "<span class='arrow' style='display:inline-block; transition:transform 0.3s; transform:rotate(0deg);'>&#9662;</span>"
        "</div>"
        "<script>"
        "(function(){"
        "var toggle = document.getElementById('toggle-comps');"
        "if(!toggle) return;"
        "var label = toggle.querySelector('span:first-child');"
        "var arrow = toggle.querySelector('.arrow');"
        "toggle.addEventListener('click', function(){"
        "  var rows = document.querySelectorAll('.extra-comp');"
        "  var hidden = rows[0] && rows[0].style.display === 'none';"
        "  rows.forEach(function(r){ r.style.display = hidden ? '' : 'none'; });"
        "  label.textContent = hidden ? 'Show fewer comparables' : 'Show more comparables';"
        "  arrow.style.transform = hidden ? 'rotate(180deg)' : 'rotate(0deg)';"
        "});"
        "})();"
        "</script>"
        "</section>"
    )

    return header + "\n".join(rows_html) + footer

def parse_nlp_sections(nlp: str):
    sections = {
        "Feature Group Impacts": "",
        "Trend Analysis": "",
        "Mortgage Analysis": "",
        "Rent Analysis": "",
        "Closing Insight": "",
    }
    if not nlp:
        return sections

    # Normalize: remove optional leading "-" and spaces before markers
    # So "- **Title**" → "**Title**"
    normalized = re.sub(r"^\s*-\s*(\*\*.+?\*\*)", r"\1", nlp, flags=re.MULTILINE)

    for title in sections.keys():
        marker = f"**{title}**"
        if marker in normalized:
            after = normalized.split(marker, 1)[1]
            # Find next marker
            next_markers = [f"**{t}**" for t in sections.keys() if t != title]
            ends = [after.find(m) for m in next_markers if after.find(m) != -1]
            end_idx = min(ends) if ends else len(after)
            txt = after[:end_idx].strip()
            sections[title] = txt.lstrip(":").strip()

    return sections



def comps_map_block(property_info: dict, comps: list) -> str:
    """Leaflet map with the subject property + top 5 comps. Escaped braces for f-string safety."""
    def _get_coord(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                v = to_float(d[k], None)
                if v is not None:
                    return v
        return None

    plat = _get_coord(property_info or {}, "lat", "latitude")
    plon = _get_coord(property_info or {}, "lon", "lng", "longitude", "long")

    if plat is None or plon is None:
        return (
            "<section>"
            "<h2>Map of subject & best comps</h2>"
            "<div class='muted' style='padding:12px;border:1px solid #e5e7eb;border-radius:10px;'>"
            "Map unavailable — missing subject property coordinates (expected keys: <code>lat/latitude</code> and <code>lon/lng/longitude/long</code>)."
            "</div>"
            "</section>"
        )

    comps10 = comps[:10]

    def comp_feature(c, idx):
        lat = _get_coord(c, "lat", "latitude"); lon = _get_coord(c, "lon", "lng", "longitude", "long")
        if lat is None or lon is None:
            return None
        addr = htmlmod.escape(str(c.get("full_address", "")))
        date = htmlmod.escape(pretty_date(c.get("date")))
        price = fmt_currency(round_price(c.get("price", 0)))
        ppm2 = f"{round_ppm2(to_float(c.get('price_per_m2', 0)) or 0):,.0f}"
        floor_area = f"{fmt_m2_ft2_html(c.get('floor_area', 0))}"
        dist = fmt_km2(c.get("geo_dist_km"))
        popup = f"<b>Comp {idx}</b><br>{addr}<br>{date}<br>{price}<br>{floor_area} · £/m² {ppm2}<br>{dist} km away"
        return {"lat": lat, "lon": lon, "popup": popup}

    comps_features = [cf for i, c in enumerate(comps10, 1) if (cf := comp_feature(c, i)) is not None]

    # Escape all literal JS/CSS braces by doubling them.
    return f"""
<section>
  <h2>Map of subject & best comps</h2>
  <div id="comp-map" style="height: 380px; border: 1px solid #e5e7eb; border-radius: 10px;"></div>
  <p class="muted">Subject property shown as a star; comparables are numbered markers.</p>
</section>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script>
(function(){{
  var MAX_TRIES = 50;
  function ready(fn){{ if(document.readyState==='loading'){{document.addEventListener('DOMContentLoaded', fn);}} else {{fn();}} }}
  function ensureMap(tries){{ 
    tries = (typeof tries==='number') ? tries : 0;
    var el = document.getElementById('comp-map');
    if(!el) return;
    if(typeof L==='undefined'){{ if(tries<MAX_TRIES) return setTimeout(function(){{ensureMap(tries+1);}},120); el.innerHTML='<div style="padding:12px;color:#6b7280;">Leaflet failed to load.</div>'; return; }}
    try {{
      var map = L.map(el).setView([{plat}, {plon}], 14);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
      window._leafletCompMap = map;
      var comps = {json.dumps(comps_features)};
      var bounds = L.latLngBounds([[{plat}, {plon}]]);
      bounds.extend([{plat}, {plon}]);
      comps.forEach(function(c, i){{ 
        var num = i + 1;
        var icon = L.divIcon({{ className:'comp-marker', html:'<div style="background:#2563eb;color:#fff;border-radius:12px;padding:2px 6px;font-size:12px;">'+num+'</div>', iconSize:[24,24], iconAnchor:[12,12] }});
        var m = L.marker([c.lat, c.lon], {{icon: icon}}).addTo(map);
        m.bindPopup(c.popup);
        bounds.extend([c.lat, c.lon]);
      }});
      if (comps.length>0) {{ map.fitBounds(bounds.pad(0.2)); }}
    }} catch(e) {{ console.error('Map init failed:', e); el.innerHTML='<div style="padding:12px;color:#6b7280;">Map failed to load.</div>'; }}
    var subjectIcon = L.divIcon({{ className: 'subject-marker', html: '<div style="font-size:20px; line-height:20px;">⭐</div>', iconSize: [20,20], iconAnchor: [10,10] }});
    var subject = L.marker([{plat}, {plon}], {{icon: subjectIcon}}).addTo(map);
    subject.bindPopup('<b>Subject property</b>');
  }}
  ready(function(){{ ensureMap(0);}});
}})();
</script>
"""


# ---------- top-level: build & write ----------

def build_html(data: dict) -> str:
    pred_price = round_price(float(data["pred_price"]))
    pred_price_form = fmt_currency(round_price(float(data["pred_price"])))
    price_low = round_price(float(data["price_low"]))
    price_high = round_price(float(data["price_high"]))
    pred_ppm2 = round_ppm2(float(data["pred_ppm2"]))
    base_value = round_price(float(data["base_value"]))
    comps_confidence = data['comps_confidence']
    epc = data['property_info']['energy_eff']
    epc_bump = data['epc_scenario']['epc_bump']
    price_bump = fmt_currency(round_price(data['epc_scenario']['price_bump']))

    price_rgn_div = chart_price_interval(pred_price, price_low, price_high)

    sale_likelihood = chart_sale_likelihood(data['sale_likelihood'], data['pred_price'])

    shap_div, _ = chart_shap(data.get("shap_values", {})['numeric_groups'], top_n=6)

    sections = parse_nlp_sections(data.get("nlp_analysis", ""))

    info = data.get("property_info", {}) or {}
    prop_block = property_icons_block(info)
    median_block = median_note()
    comps_block = comps_table_block(data.get("display_comps", []), base_value, data.get("comps_confidence", "-"))
    map_block = comps_map_block(info, data.get("display_comps", []))

    # --- Mortgage analysis ---
    borrow_rate = data.get("borrow_rates")
    mortgage_div = chart_borrow_rates(borrow_rate) if borrow_rate is not None else None
    mortgage_text = esc(sections.get("Mortgage Analysis", ""))

    plan_cfg = data.get("mortgage_planner") or {}
    planner_tables = {}

    def _to_rate(val):
        # Accept numbers or strings like "4.89" or "4.89%"
        if val is None:
            return None
        s = str(val).strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        try:
            return float(s)
        except Exception:
            return None

    if plan_cfg:
        property_price_cfg = plan_cfg.get("property_price")
        loan_amount_cfg = plan_cfg.get("loan_amount")
        mats = plan_cfg.get("maturities_years") or [10, 15, 20, 25, 30]
        ups = plan_cfg.get("upfront_percents") or [0, 5, 10, 20, 40]

        rates = {
            "2y": _to_rate(plan_cfg['rate_pct'].get("2y")),
            "5y": _to_rate(plan_cfg['rate_pct'].get("5y")),
        }

        for label, rate in rates.items():
            if rate is not None:
                grid_df = compute_payment_grid(
                    rate_pct=rate,
                    property_price=property_price_cfg,
                    loan_amount=loan_amount_cfg,
                    maturities_years=mats,
                    upfront_percents=ups,
                )
                header_cells = "".join(f"<th>{htmlmod.escape(str(c))}</th>" for c in grid_df.columns)
                body_rows = []
                for idx, row in grid_df.iterrows():
                    tds = "".join(f"<td>{fmt_currency(v)}</td>" for v in row.values)
                    body_rows.append(f"<tr><th>{htmlmod.escape(str(idx))}</th>{tds}</tr>")
                table_html = (
                    f"<div class='muted' style='margin-top:8px;'>Monthly repayment grid (rate {rate:.2f}%)</div>"
                    "<table class='comps' style='margin-top:6px;'>"
                    f"<thead><tr><th>Upfront %</th>{header_cells}</tr></thead>"
                    f"<tbody>{''.join(body_rows)}</tbody>"
                    "</table>"
                )
                planner_tables[label] = table_html
            else:
                planner_tables[label] = (
                    "<div class='muted' style='padding:12px;border:1px dashed var(--border);border-radius:10px;'>"
                    f"No {label} fixed rate provided — unable to compute the repayment grid."
                    "</div>"
                )

    pc = info.get("postcode") or "-"
    pc_prefix = str(pc).split()[0] if pc != "-" else "-"
    city_name = info.get("city") or info.get("town") or info.get("locality") or "Citywide"

    prefix_chart_title = f"Postcode {pc_prefix}" if pc_prefix != "-" else "Postcode prefix"
    city_chart_title = city_name

    # Keep CSS/JS as plain strings (no f-strings) to avoid brace escaping.
    head_css = """<style>
:root { --ink: #111; --muted: #6b7280; --bg: #ffffff; --border: #e5e7eb; }
* { box-sizing: border-box; }
body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color: var(--ink); background: var(--bg); margin: 0; line-height: 1.5; }
.container { max-width: 960px; margin: auto; padding: 28px 20px 64px; }

/* adjust these two numbers only */
:root{
  --logo-height: 100px;      /* visual size */
  --logo-left-trim: 12px;   /* pulls image left to cancel PNG padding */
}

/* header stays as-is */
header{
  display:flex;
  align-items:center;       /* nice vertical balance with KPIs */
  gap:24px;
  margin-bottom:12px;
}

/* only the logo moves */
.logo-block{ flex:0 0 auto; }

/* use a background so we can offset it left; this ignores global img rules */
.logo{
  width: calc(300px + var(--logo-left-trim));   /* keep your usual width */
  height: var(--logo-height);
  background-image: url('https://propmind-reports.s3.eu-west-2.amazonaws.com/logo.png');
  background-repeat: no-repeat;
  background-size: auto 100%;                   /* scale by height */
  background-position: calc(-1 * var(--logo-left-trim)) center; /* trim left padding */
  display:block;
}


.kpis {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
  padding-top: 6px;
}


.title-wrap { display:flex; flex-direction:column; gap:4px; } /* you can delete if unused */
.title { font-size: 28px; font-weight: 800; }
.address { color: var(--muted); font-size: 14px; }

.kpis { display:flex; gap:16px; flex-wrap:wrap; }
.kpi { padding:0; min-width: 180px; }
.kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing:.04em; }
.kpi .value { font-size: 18px; font-weight:700; }

section { margin: 28px 0; }
h2 { font-size: 18px; margin: 0 0 12px; }
.prop-grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px 16px; }
.prop-item { display:flex; gap:10px; align-items:center; padding:4px 0; }
.prop-item .icon { width: 140px; font-weight:600; }
.prop-item .val { color: var(--ink); word-break: break-word; }

.comps { width:100%; border-collapse: collapse; }
.comps th, .comps td { border-bottom: 1px solid var(--border); padding: 10px 8px; font-size: 13px; text-align:left; }
.comps thead th { font-weight:600; }

p { margin: 10px 0 0; }
.muted { color: var(--muted); }

.print-btn{ position: fixed; top: 16px; right: 16px; background: #111; color: #fff; border: 0; border-radius: 10px; padding: 10px 14px; font-weight: 600; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.12); z-index: 1000; }
.print-btn:active{ transform: translateY(1px); }

.comps td .dual .base { display:block; }
.comps td .dual .alt  { display:block; color: var(--muted); font-size: 12px; margin-top: 2px; }
/* --- Mortgage toggle buttons --- */
#mortgage-toggle {
  display: inline-flex;
  gap: 8px;
  margin: 8px 0 12px;
}
#mortgage-toggle button {
  border: 1px solid var(--border);
  background: #fff;
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s ease, border-color .15s ease;
}
#mortgage-toggle button:hover {
  background: #f8fafc;
}
#mortgage-toggle button.active {
  background: #111;
  color: #fff;
  border-color: #111;
}


.footer-note {
  margin: 24px 0 0;
  padding-top: 10px;
  border-top: 1px dashed var(--border);
  text-align: center;
  font-size: 12px;
  color: var(--muted);
  page-break-inside: avoid;
}
</style>
"""

    script_pdf = """<script>
(function(){
  var btn = document.getElementById('to-pdf');
  if(!btn) return;
  function slugify(s){ return (s||'').toString().toLowerCase().replace(/[^a-z0-9]+/g,'-')
                        .replace(/^-+|-+$/g,'').substring(0,80) || 'property-report'; }
  btn.addEventListener('click', function(){
    var el = document.querySelector('.container');
    var fname = slugify(document.title) + '.pdf';
    var mapEl = document.getElementById('comp-map');
    var prev = mapEl ? mapEl.style.display : null;
    if(mapEl) mapEl.style.display = 'none';
    var opt = { margin: 10, filename: fname,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, logging: false },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' } };
    html2pdf().set(opt).from(el).save()
      .then(function(){ if(mapEl) mapEl.style.display = prev; })
      .catch(function(){ if(mapEl) mapEl.style.display = prev; });
  });
})();
</script>"""

    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Property Valuation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
{head_css}
</head>
<body>
<button id="to-pdf" class="print-btn" title="Save this report as PDF">Download PDF</button>
<div class='container'>
<header>
  <div class="logo-block">
    <div class="logo" role="img" aria-label="PropMind logo"></div>
  </div>
  <div class='kpis'>
    <div class='kpi'><div class='label'>Predicted price</div><div class='value'>{pred_price_form}</div></div>
    <div class='kpi'><div class='label'>£/m²</div><div class='value'>{pred_ppm2:,.0f}</div></div>
    <div class='kpi'><div class='label'>Confidence interval</div><div class='value'>{fmt_currency(price_low)} – {fmt_currency(price_high)}</div></div>
  </div>
</header>
{prop_block}
{median_block}
<section>
  <h2>Price range</h2>
  {price_rgn_div}
  <p>The predicted price for this property is {pred_price_form}, with a confidence interval ranging from {fmt_currency(price_low)} to {fmt_currency(price_high)}. This interval reflects market uncertainty, such as variations in property condition, local demand, or how quickly the property might sell. It means the actual price could fall anywhere within this range depending on market conditions.</p>
</section>
<section>
    <h2>Base price & comparables</h2>
    <p>The base price of {fmt_currency(base_value)} is derived from nearby sales within a 1 km radius (0.6 miles) over the past 12 months. The most similar properties are displayed in the comparables list, and the confidence level for these comparisons is {comps_confidence}.</p>
    {comps_block}
    <br>
    <p>The map below shows the location of the most comparable properties sold in the past year.</p>
    {map_block}
</section>
<section>
  <h2>Feature impacts (SHAP)</h2>
  <p>The SHAP chart below shows which features explain the difference between the base price (from comparable sales) and the model’s predicted price. Positive bars indicate factors that increase the estimate, while negative bars show those that reduce it.</p>
  {shap_div}
  <div>{esc(sections.get('Feature Group Impacts', ''))}</div>
</section>
<section>
    <h2>EPC rating Analysis</h2>
    <p>Improving this property’s EPC rating from {epc_letter_from_score(epc)} to {epc_letter_from_score(epc_bump)} could raise its estimated value from {pred_price_form} to {price_bump}.
    <p>You can boost a UK property’s energy efficiency by adding insulation (loft, cavity wall, or floor), upgrading to double or triple glazing, and installing a more efficient heating system such as a condensing boiler or heat pump. Simple steps like switching to LED lighting or adding renewable energy (e.g. solar panels) also help improve the rating.</p>
</section>
<section>
    <h2>Trend Analysis</h2>
    <p>The following charts show how prices in the local postcode area and across the wider city have evolved over time. Comparing these trends helps put the property’s valuation in context, showing whether it is aligned with or diverging from broader market movements.</p>
      <h4>{htmlmod.escape(prefix_chart_title)}</h4>
      {chart_trend_toggle(data.get('prefix_trend', []), pred_ppm2, pred_price, prefix_chart_title)}
</section>
<section>
    <h4>{htmlmod.escape(city_chart_title)}</h4>
    {chart_trend_toggle(data.get('city_trend', []), pred_ppm2, pred_price, city_chart_title)}
    <div>{esc(sections.get('Trend Analysis', ''))}</div>
</section>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<section>
  <h2>Sale likelihood</h2>
  <p>This chart shows the estimated likelihood of sale at different asking prices around the model’s fair value in the next three months. The dashed line marks the predicted price; pricing below it (discounts) raises the estimated chance of selling, while pricing above it (premiums) lowers it.</p>
  {sale_likelihood}
</section>
<section>
  <h2>Mortgage analysis</h2>
  <br>
  <p>The chart below shows how the Bank of England’s two-year and five-year fixed borrowing rates have changed over the past year. A fixed rate means the interest on a mortgage stays the same for the chosen period, giving borrowers certainty over monthly repayments. The two-year rate reflects shorter-term costs, while the five-year rate indicates longer-term affordability and market expectations.</p>
  <br>
  {mortgage_div or ""}
  <br>
  <p class='muted'>Two- and five-year fixed borrow rates over time.</p>
  <br>
  <div>{mortgage_text or ""}</div>
  <br>
  <p>You can view estimated monthly repayments under either the 2-year or 5-year fixed rate. Figures assume a standard repayment mortgage with monthly instalments.</p>
</section>
<div id="mortgage-toggle">
  <button class="active" onclick="showTable('2y', this)">2-year fixed</button>
  <button onclick="showTable('5y', this)">5-year fixed</button>
</div>

<script>
function showTable(which, btn) {{
  document.getElementById('table-2y').style.display = (which === '2y') ? 'block' : 'none';
  document.getElementById('table-5y').style.display = (which === '5y') ? 'block' : 'none';

  var buttons = document.querySelectorAll('#mortgage-toggle button');
  buttons.forEach(function(b) {{ b.classList.remove('active'); }});

  if (btn) {{ btn.classList.add('active'); }}
}}
</script>

<div id="table-2y" class="mortgage-table">
  {planner_tables.get("2y", "")}
</div>
<div id="table-5y" class="mortgage-table" style="display:none;">
  {planner_tables.get("5y", "")}
</div>
</section>
<section>
  <h2>Rent Analysis</h2>
  <p>{esc(sections.get('Rent Analysis', ''))}</p>
</section>
<section>
  <h2>Conclusion</h2>
  <p>{esc(sections.get('Closing Insight', ''))}</p>
</section>
<div class="footer-note">
  This report is based on recent sales data and statistical models. It isn’t a formal survey or lender’s valuation. Actual sale price may vary with condition and demand.
</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
{script_pdf}
</body>
</html>"""

    return html


# def write_report(data: dict, out_path: str) -> str:
#     html = build_html(data)
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     return out_path
