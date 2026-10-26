# Data sources and licensing decisions

Status as of 2026-10-26. Everything here is public data; nothing is scraped from behind access controls.

| Source | What it gives | Access | Collector |
|---|---|---|---|
| OpenSky `/states/all` (anonymous) | live positions in the WAW box every ~4.5 min → take-off / landing at WAW, icao24, all operators (congestion) | no account, 400 credits/day | `ripple collect states` (launchd, every 270 s) **running** |
| OpenSky `/flights/*` (account) | firstSeen / lastSeen + estimated airports per flight (actual times at both ends) | free account → API client (OAuth2) | `ripple collect opensky-flights` (daily, skipped until `.env` has credentials) |
| OpenSky historical DB (Trino) | full state-vector history | academic application, see `opensky_application.md` | not yet |
| OpenSky aircraft DB | icao24 → registration, type | public CSV | `ripple collect aircraft-db` |
| IEM ASOS archive | global METAR history (EPWA + all network airports) | free, no key | `ripple collect metar` (daily) **running** |
| aviationweather.gov Data API | current TAFs; IEM only archives US TAFs, so we build our own archive from today | free, no key | `ripple collect taf` (launchd, hourly) **running** |
| AeroDataBox FIDS | **scheduled** + revised + runway times at both ends, registration, Mode-S | API key (RapidAPI free tier works) | `ripple collect schedules` (daily, skipped until `.env` has a key) |
| EUROCONTROL Aviation Data Repository for Research | historical flights with filed + actual off-block, arrival, registration (Mar/Jun/Sep/Dec, 2-year delay) | free for research, OneSky Online account | loader `ripple.data.sources.eurocontrol` |

## Why not the Chopin Airport flight board

Checked on 2026-10-26:

* `https://lotnisko-chopina.pl/robots.txt` disallows only `/admin/` and `/django-admin/`.
* The site has no general website terms of use; `regulamin-uzytkowania-lotniska` covers the airport
  premises, not the website.
* **But** the board itself (`/pl/przyloty-i-odloty/`, `/en/arrivals-and-departures/`) is served behind an
  F5 BIG-IP bot-defence challenge (`/TSPD/...` script + image CAPTCHA). Other pages are not.

A CAPTCHA on exactly the page with the data is an explicit technical "no" to automated access.
Bypassing it would be circumvention, regardless of what robots.txt says. Ripple therefore does
not scrape the board. If you want to use it, write to the airport (e.g. press/B2B contact) and ask for a
feed or permission for a research project.

## Paid / keyed alternatives for scheduled times

Prices checked on 2026-10-26 from the vendors' pages; verify before paying.

| Provider | Free tier | Cheapest paid | Fit for Ripple |
|---|---|---|---|
| **AeroDataBox** (RapidAPI / API.Market / direct) | RapidAPI Basic: 400 units/month, 1 req/s; no commercial use; attribution required | RapidAPI Pro $8/month = 5,000 units; API.Market Pro $7.50; direct Starter $19 = 40,000 | **Best.** FIDS is Tier 2 = 2 units/call, ≤12 h per call → WAW costs ~4 units/day ≈ 124/month, which fits the free tier. History up to 365 days → a one-month Pro plan (~$8) backfills a full year (~1,460 units). |
| FlightAware AeroAPI | Personal tier: $5 of free queries per month, no minimum, 10 result sets/min | pay-per-query above $5 | Good live data, but the Personal tier has **no historical data**, and per-result-set prices could not be confirmed. |
| Aviationstack | 100 requests/month (personal use) | Basic $49.99/month = 10,000 requests; history from Basic up | 100 requests is too few for daily collection. Paid tiers cost more than AeroDataBox for the same job. |

### AeroDataBox caveats (read before signing up)

* **Retention.** Terms art. 5.5 cap caching of raw contents at 7 days unless your plan or a written
  exemption allows longer. A training set needs months. Ask AeroDataBox (support@aerodatabox.com)
  for an academic/research exemption *before* building the archive. Until you have one, set
  `AERODATABOX_RETAIN_DAYS=7` in `.env`. The daily job then deletes raw files after 7 days.
  Derived labels (delay minutes per flight) are a "Derived Work" in their terms; check with
  them whether keeping those is acceptable.
* Coverage: whether EPWA has *live* updates (revised/runway times) or only schedules is not verified
  yet. Check with `GET /health/services/airports/EPWA/feeds` after signing up.
* The 12 h window per call is an assumption based on the docs ("determined by your pricing plan").

## Recommended path

1. **Today (free):** create an OpenSky account → *Account → API client* → put
   `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET` into `.env`. The daily job picks them up automatically.
2. **Today (free):** RapidAPI account → subscribe to AeroDataBox *Basic* → `AERODATABOX_KEY` in `.env`,
   and email them about the academic retention exemption.
3. **This week:** OneSky Online account → request the EUROCONTROL R&D dataset (historical flights with
   registrations; ideal for training before our own archive is long enough).
4. **Optional, ~$8:** one month of AeroDataBox Pro to backfill 365 days of WAW history once the
   exemption question is settled.
5. Send the OpenSky academic application (`docs/opensky_application.md`) from your UAB e-mail.

Until real schedules arrive, every model is developed and tested on the synthetic network simulator
(`ripple.data.synthetic`). The README reports synthetic and real results separately and never mixes them.
