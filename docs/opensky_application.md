# OpenSky historical data access: application draft

**How to apply.** OpenSky grants Trino access to its historical database to researchers at universities
and non-profit research institutions. Log in at <https://opensky-network.org>, then use the data access /
contact form for research access, or e-mail the OpenSky team. Send it from your **UAB institutional**
address, because institutional e-mail is what they use to verify affiliation. Ask your TFG supervisor
if you can name them as the responsible academic.

Fill in the `[...]` fields before sending.

---

**Subject:** Research access request: delay propagation in an airline hub network (UAB bachelor thesis)

Dear OpenSky team,

My name is Kacper Kotowski. I am a third-year student of the Bachelor's degree in Artificial Intelligence
at the Universitat Autònoma de Barcelona. I am writing to request access to the OpenSky historical
database (Trino) for my bachelor thesis (TFG), supervised by [supervisor name, department, e-mail].

**Project.** *Ripple* studies how delays propagate through a hub-and-spoke airline network, using LOT
Polish Airlines and its Warsaw Chopin (EPWA) hub as a case study. Reactionary delays account for
roughly 40–48% of delay minutes in Europe (EUROCONTROL CODA, 2025). I model each flight as a node in a
dynamic graph. Edges represent aircraft rotations, passenger connections at the hub, and shared
airport/time congestion. A graph neural network then predicts probabilistic arrival and departure
delays 1–6 hours ahead. It also predicts the probability that a connection at the hub is missed. The
model is compared against a gradient-boosting baseline.

**Why OpenSky.** Aircraft rotations are the main propagation channel. They can only be rebuilt from
per-aircraft flight sequences (icao24 → consecutive legs) with actual take-off and landing times. The
REST API covers recent days; the thesis needs at least 12–24 months of history to train and evaluate
across seasons.

**Data requested.**
- `flights_data4` (or equivalent flight table): flights with callsign prefix `LOT` and flights departing
  from or arriving at EPWA, [2023-01-01] to [present].
- Optionally, `state_vectors_data4` within 30 NM of EPWA for the same period, at reduced resolution, to
  estimate taxi times and runway congestion.
- Estimated volume is small: about 300 LOT legs per day.

**Use and publication.** Non-commercial academic use only. Results will appear in my TFG at UAB and
possibly a workshop paper, with OpenSky credited using the recommended citation
(Schäfer et al., "Bringing up OpenSky", IPSN 2014). No raw data will be redistributed. The code will
be published on GitHub without data.

I would be grateful for access, and I am happy to provide any further information.

Kind regards,
Kacper Kotowski
Bachelor in Artificial Intelligence, Universitat Autònoma de Barcelona
[UAB e-mail] · [GitHub / LinkedIn]
