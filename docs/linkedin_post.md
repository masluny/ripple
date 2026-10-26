# LinkedIn post - draft

Written in Kacper's own voice (write-as-kacper scorer: 93/100). The numbers come from `reports/synthetic/results.md`. Re-check them before posting and add the dashboard screenshot (`reports/figures/dashboard.png`).

---

Did you ever wonder why your flight is late, when the weather is perfect and the airport is almost empty? Most of the time the answer is pretty simple. The plane was late before it even came to you.

So, when it comes to European aviation, around 40-48% of all delay minutes in 2025 were reactionary delays (EUROCONTROL CODA). It means that the delay was just inherited from the previous flight of the same aircraft. And I think that in a hub like Warsaw Chopin, where LOT connects passengers from all of Europe to the long haul flights, one late plane can probably break a lot of connections in a very short time.

That is why for last weeks I was building Ripple. It's a graph neural network (heterogeneous GATv2 + GRU) where every flight is a node and the delay can travel between them by aircraft rotations, connections at WAW and shared airport and weather. It does not give just one number, it predicts the whole distribution of delay 1, 3 and 6 hours ahead, the probability that a connection will be missed and also it tries to explain why, for example "inherited from previous leg of SP-LxA from CDG".

To be fair, my GNN had to beat a LightGBM model with exactly the same inputs. On the test set it got CRPS {GNN_CRPS} min vs {LGBM_CRPS} min, with Brier skill {BSS} for missed connections, so I think it's a good start but for sure not the end.

However, i have to be honest about one thing. Right now the model is trained on a synthetic LOT-like network calibrated on real OpenSky rotations, because real scheduled times need API acces. Luckily, the data collectors for OpenSky, METAR and TAF are already running, so maybe in few months I will show real results.

In conclusion, if you work in airline operations or Revenue Management, I would really love to hear what you think, especially what would make such tool useful in Ops Control.

#aviation #machinelearning #graphneuralnetworks #LOT #datascience
