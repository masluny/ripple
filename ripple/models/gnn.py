"""Dynamic heterogeneous GNN for delay propagation.

Snapshot graph at prediction time t (every 30 min):
    nodes  = flights scheduled in [t - 10 h, t + 6.5 h] (context + targets)
    edges  = rotation   (previous leg of the same tail -> next leg)          attr: turn slack
             connection (WAW arrival -> WAW departure, 35-150 min slack)     attr: slack
             airport    (same departure or arrival airport within +-40 min)  attr: time offset
    node x = the same as-of features LightGBM gets (FeatureStore), standardised

Model: input MLP -> L x [sum over edge types of GATv2Conv, residual, LayerNorm] -> GRUCell that
carries each flight's state across consecutive snapshots -> two monotone quantile heads
(departure and arrival delay, 19 quantiles each).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from ripple import config
from ripple.data.features import CATEGORICAL, MAX_LEAD, MAX_OVERDUE, SLOT, FeatureStore
from ripple.eval import metrics

log = logging.getLogger(__name__)
CTX_PAST = 600
EDGE_TYPES = ("rotation", "connection", "airport")
TAUS = torch.tensor(config.QUANTILES)
SCALE = 10.0  # model works in units of 10 minutes


# ------------------------------------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------------------------------------
class EdgeIndex:
    """All edges of one type over the whole dataset, sorted so a snapshot is a cheap slice."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, attr: np.ndarray):
        lo = np.minimum(src, dst)
        o = np.argsort(lo, kind="stable")
        self.src, self.dst, self.attr = src[o], dst[o], attr[o].astype(np.float32)
        self.lo = lo[o]
        self.hi = np.maximum(src, dst)[o]

    def select(self, lo: int, hi: int):
        a, b = np.searchsorted(self.lo, [lo, hi])
        m = self.hi[a:b] < hi
        return self.src[a:b][m] - lo, self.dst[a:b][m] - lo, self.attr[a:b][m]


def build_edges(store: FeatureStore, window: float = 40, k: int = 6) -> dict[str, EdgeIndex]:
    n = store.n
    # rotation
    dst = np.where(store.prev >= 0)[0]
    src = store.prev[dst]
    rot = EdgeIndex(src, dst, np.nan_to_num((store.buffer[dst] - store.min_turn[dst]) / 60))
    # connection
    cnt = store.feed_ptr[1:] - store.feed_ptr[:-1]
    cdst = np.repeat(np.arange(n), cnt)
    conn = EdgeIndex(store.feed_in, cdst, store.feed_slack / 60)
    # airport/time: k nearest neighbours on each side in the same airport's departure (arrival) queue
    s_all, d_all, a_all = [], [], []
    for ap_arr, t_arr in ((store.dep_ap, store.sd), (store.arr_ap, store.sa)):
        for a in np.unique(ap_arr):
            m = np.where(ap_arr == a)[0]
            o = m[np.argsort(t_arr[m])]
            for off in range(1, k + 1):
                i, j = o[:-off], o[off:]
                dt = t_arr[j] - t_arr[i]
                ok = dt <= window
                s_all += [i[ok], j[ok]]
                d_all += [j[ok], i[ok]]
                a_all += [dt[ok] / 60, -dt[ok] / 60]
    apt = EdgeIndex(np.concatenate(s_all), np.concatenate(d_all), np.concatenate(a_all))
    return {"rotation": rot, "connection": conn, "airport": apt}


# Delay-like inputs also get an unsquashed linear channel in hours: the GNN must be able to pass a
# 3-hour inbound delay through max(0, delay - slack) without it being saturated.
LINEAR_COLS = ("own_delay", "prev_delay", "chain_delay", "chain_slack", "proj_reactionary",
               "turn_slack", "feed_mean", "feed_max", "feed_break", "dep_ap_recent_dep",
               "arr_ap_recent_arr", "hub_recent_dep", "hub_recent_arr", "lead", "prev_lead")  # fmt: skip


@dataclass
class Preproc:
    names: list[str]
    keep: list[int]  # numeric columns used
    cat: list[int]
    med: list[float]
    std: list[float]
    nan_cols: list[int]  # positions (within keep) that get a missing indicator
    lin_cols: list[int] = field(
        default_factory=list
    )  # positions (within keep) with an hours channel
    version: int = 1  # 1 = robust z clipped to +-6 (first models); 2 = asinh(z) + hours channels

    @classmethod
    def fit(cls, x: np.ndarray, names: list[str], exclude: tuple[str, ...]) -> Preproc:
        cat = [names.index(c) for c in CATEGORICAL]
        keep = [
            i
            for i, nm in enumerate(names)
            if i not in cat and not nm.startswith(exclude or ("\0",))
        ]
        sub = x[:, keep]
        med = np.nanmedian(sub, axis=0)
        q75, q25 = np.nanpercentile(sub, 75, axis=0), np.nanpercentile(sub, 25, axis=0)
        std = np.nanstd(sub, axis=0)
        scale = np.where((q75 - q25) > 1e-6, (q75 - q25) / 1.35, np.where(std > 1e-6, std, 1.0))
        nan_cols = [j for j in range(len(keep)) if np.isnan(sub[:, j]).any()]
        lin_cols = [j for j, i in enumerate(keep) if names[i] in LINEAR_COLS]
        med = np.nan_to_num(med)
        return cls(names, keep, cat, med.tolist(), scale.tolist(), nan_cols, lin_cols, version=2)

    def __call__(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        sub = x[:, self.keep]
        miss = np.isnan(sub[:, self.nan_cols]).astype(np.float32)
        z = (sub - np.array(self.med, np.float32)) / np.array(self.std, np.float32)
        z = np.nan_to_num(z, nan=0.0)
        if self.version == 1:
            parts = [np.clip(z, -6, 6), miss]
        else:
            hours = np.nan_to_num(sub[:, self.lin_cols], nan=0.0) / 60.0
            parts = [np.arcsinh(z), hours, miss]
        cat = np.nan_to_num(x[:, self.cat], nan=-1).astype(np.int64) + 1
        return torch.from_numpy(np.hstack(parts).astype(np.float32)), torch.from_numpy(cat)

    @property
    def dim(self) -> int:
        lin = len(self.lin_cols) if self.version >= 2 else 0
        return len(self.keep) + lin + len(self.nan_cols)


# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------
class QuantileHead(nn.Module):
    def __init__(self, d: int, n_q: int = len(config.QUANTILES)):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_q))
        self.mid = n_q // 2

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        o = self.mlp(h)
        center = o[:, self.mid : self.mid + 1]
        inc = nn.functional.softplus(o) + 1e-3
        up = torch.cumsum(inc[:, self.mid + 1 :], dim=1)
        down = torch.flip(torch.cumsum(torch.flip(inc[:, : self.mid], [1]), dim=1), [1])
        return torch.cat([center - down, center, center + up], dim=1)


class PeriodicEmbedding(nn.Module):
    """Per-feature periodic embeddings (PLR, Gorishniy et al. 2022, "On Embeddings for Numerical
    Features in Tabular Deep Learning"): x -> ReLU(Linear([sin(2 pi c x), cos(2 pi c x)])).
    Lets an MLP carve the threshold-like structure that gradient-boosted trees get for free."""

    def __init__(self, n_features: int, k: int = 8, d: int = 8, sigma: float = 0.5):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_features, k) * sigma)
        self.w = nn.Parameter(torch.randn(n_features, 2 * k, d) / (2 * k) ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_features, d))
        self.out_dim = n_features * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2 * torch.pi * x[:, :, None] * self.c[None]
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=2)
        return torch.relu(torch.einsum("nfk,fkd->nfd", v, self.w) + self.b).flatten(1)


class RippleGNN(nn.Module):
    def __init__(self, d_in: int, edge_types=EDGE_TYPES, hidden: int = 96, layers: int = 2,
                 heads: int = 4, temporal: bool = True, dropout: float = 0.1,
                 num_emb: str = "linear"):  # fmt: skip
        super().__init__()
        n_ap = len(config.AIRPORTS) + 2
        self.ap_emb = nn.Embedding(n_ap, 12)
        self.type_emb = nn.Embedding(16, 6)
        self.num = PeriodicEmbedding(d_in) if num_emb == "plr" else None
        d_num = self.num.out_dim if self.num is not None else d_in
        self.inp = nn.Sequential(
            nn.Linear(d_num + 2 * 12 + 6, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.edge_types = tuple(edge_types)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(nn.ModuleDict({
                et: GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1,
                              add_self_loops=False, dropout=0.0)
                for et in self.edge_types
            }))  # fmt: skip
            self.norms.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.temporal = temporal
        self.gru = nn.GRUCell(hidden, hidden) if temporal else None
        self.head_dep = QuantileHead(2 * hidden)
        self.head_arr = QuantileHead(2 * hidden)

    def forward(self, x, cat, edges, h_prev=None, return_attention=False):
        xn = self.num(x) if self.num is not None else x
        e = torch.cat(
            [xn, self.ap_emb(cat[:, 0]), self.ap_emb(cat[:, 1]), self.type_emb(cat[:, 2])], 1
        )
        h0 = self.inp(e)
        h = h0
        att = {}
        for li, (convs, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            msg = torch.zeros_like(h)
            for et in self.edge_types:
                ei, ea = edges[et]
                if ei.shape[1] == 0:
                    continue
                if return_attention and li == len(self.convs) - 1:
                    out, (ei_a, alpha) = convs[et](h, ei, ea, return_attention_weights=True)
                    att[et] = (ei_a, alpha.mean(1))
                else:
                    out = convs[et](h, ei, ea)
                msg = msg + out
            h = norm(h + self.drop(nn.functional.gelu(msg)))
        if self.temporal:
            h_t = self.gru(h, h_prev if h_prev is not None else torch.zeros_like(h))
        else:
            h_t = h
        z = torch.cat([h_t, h0], 1)
        return self.head_dep(z), self.head_arr(z), h_t, att


def pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = y[:, None] - q
    taus = TAUS.to(q.device)
    return torch.maximum(taus * d, (taus - 1) * d).mean()


# ------------------------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------------------------
@dataclass
class GNNConfig:
    hidden: int = 96
    layers: int = 2
    heads: int = 4
    temporal: bool = True
    seq_len: int = 4
    batch: int = 24
    epochs: int = 10
    lr: float = 1e-3  # defaults = best validation config of scripts/sweep_gnn.py ("reg_v2")
    weight_decay: float = 1e-3
    dropout: float = 0.2
    num_emb: str = "linear"  # or "plr" (periodic embeddings)
    start_stride: int = 2  # sequence starts every 2 slots (1 h)
    drop_edges: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    patience: int = 3
    seed: int = 0
    device: str = field(default_factory=lambda: os.environ.get("RIPPLE_DEVICE", "cpu"))


class GNNTrainer:
    def __init__(self, store: FeatureStore, split: dict, **kw):
        self.store = store
        self.split = split
        self.cfg = GNNConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})
        self.edges = build_edges(store)
        self.edge_types = tuple(e for e in EDGE_TYPES if e not in self.cfg.drop_edges)
        self.pre: Preproc | None = None
        self.model: RippleGNN | None = None
        self.history: list[dict] = []
        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(int(os.environ.get("RIPPLE_THREADS", os.cpu_count() or 4)))

    # --- snapshots ---------------------------------------------------------------------------
    def node_range(self, t: float, extra: np.ndarray | None = None) -> tuple[int, int]:
        sd = self.store.sd
        lo = int(np.searchsorted(sd, t - CTX_PAST))
        hi = int(np.searchsorted(sd, t + MAX_LEAD + SLOT, side="right"))
        if extra is not None and len(extra):
            lo, hi = min(lo, int(extra.min())), max(hi, int(extra.max()) + 1)
        return lo, hi

    def snapshots(self, ts, ranges, occlude: dict | None = None):
        """Several snapshot graphs; node features for all of them come from one vectorised call."""
        idx_all = np.concatenate([np.arange(lo, hi) for lo, hi in ranges])
        t_all = np.concatenate(
            [np.full(hi - lo, t) for t, (lo, hi) in zip(ts, ranges, strict=True)]
        )
        x_all = self.store.features(idx_all, t_all)
        out, o = [], 0
        for t, (lo, hi) in zip(ts, ranges, strict=True):
            out.append(self.snapshot(t, lo, hi, occlude, x=x_all[o : o + hi - lo]))
            o += hi - lo
        return out

    def snapshot(self, t: float, lo: int, hi: int, occlude: dict | None = None, x=None):
        st = self.store
        idx = np.arange(lo, hi)
        if x is None:
            x = st.features(idx, np.full(len(idx), t))
        if occlude:
            x = occlude["features"](x, idx)
        xn, xc = self.pre(x)
        edges = {}
        for et in self.edge_types:
            s, d, a = self.edges[et].select(lo, hi)
            if occlude and et in occlude.get("drop_into", ()):
                keep = ~np.isin(d + lo, occlude["nodes"])
                s, d, a = s[keep], d[keep], a[keep]
            edges[et] = (np.vstack([s, d]), a)
        tgt = (st.ad[idx] > t) & (st.sd[idx] >= t - MAX_OVERDUE) & (st.sd[idx] <= t + MAX_LEAD)
        tgt &= ~np.isnan(st.aa[idx])
        y = st.targets(idx)
        return xn, xc, edges, tgt, y

    def _batch(self, snaps):
        """Concatenate snapshot graphs into one disjoint graph."""
        xs, cs, ys, ts, offs = [], [], [], [], [0]
        edges = {et: ([], []) for et in self.edge_types}
        for xn, xc, e, tgt, y in snaps:
            off = offs[-1]
            xs.append(xn)
            cs.append(xc)
            ys.append(y)
            ts.append(tgt)
            for et in self.edge_types:
                edges[et][0].append(e[et][0] + off)
                edges[et][1].append(e[et][1])
            offs.append(off + len(xn))
        dev = self.cfg.device
        out_e = {
            et: (torch.from_numpy(np.hstack(v[0]).astype(np.int64)).to(dev),
                 torch.from_numpy(np.concatenate(v[1])[:, None]).to(dev))
            for et, v in edges.items()
        }  # fmt: skip
        return (torch.cat(xs).to(dev), torch.cat(cs).to(dev), out_e,
                np.concatenate(ts), np.vstack(ys), offs)  # fmt: skip

    def run_sequences(self, t_last: np.ndarray, extra: list[np.ndarray] | None = None,
                      train: bool = False, occlude: dict | None = None, return_attention=False):  # fmt: skip
        """Run B sequences ending at t_last[b]; yields per step (outputs, targets, ranges)."""
        L = self.cfg.seq_len if self.cfg.temporal else 1
        B = len(t_last)
        ranges = []
        for b in range(B):
            ex = extra[b] if extra else None
            r = [self.node_range(t_last[b] - SLOT * (L - 1 - k), ex) for k in range(L)]
            ranges.append(r)
        H = [None] * B
        base = [r[0][0] for r in ranges]
        for k in range(L):
            snaps = self.snapshots(
                [t_last[b] - SLOT * (L - 1 - k) for b in range(B)],
                [ranges[b][k] for b in range(B)],
                occlude,
            )
            xn, xc, e, tgt, y, offs = self._batch(snaps)
            h_prev = None
            if self.cfg.temporal and k > 0:
                parts = []
                for b in range(B):
                    lo, hi = ranges[b][k]
                    parts.append(H[b][lo - base[b] : hi - base[b]])
                h_prev = torch.cat(parts)
            last = k == L - 1
            qd, qa, h, att = self.model(
                xn, xc, e, h_prev, return_attention=return_attention and last
            )
            if self.cfg.temporal:
                for b in range(B):
                    lo, hi = ranges[b][k]
                    size = max(r[1] for r in ranges[b]) - base[b]
                    if H[b] is None:
                        H[b] = torch.zeros(size, h.shape[1], device=h.device)
                    rows = torch.arange(lo - base[b], hi - base[b], device=h.device)
                    H[b] = H[b].index_copy(0, rows, h[offs[b] : offs[b + 1]])
            yield k, qd, qa, tgt, y, offs, [r[k] for r in ranges], att

    # --- training ----------------------------------------------------------------------------
    def _slots(self, part: str) -> np.ndarray:
        m = self.split[part]
        sd = self.store.sd[m]
        lo = np.ceil(sd.min() / SLOT) * SLOT + CTX_PAST
        hi = np.floor(sd.max() / SLOT) * SLOT - MAX_LEAD
        return np.arange(lo, hi, SLOT)

    def fit(self) -> GNNTrainer:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        train_t = self._slots("train")
        # preprocessing stats from a sample of training snapshots
        sample_t = rng.choice(train_t, 150, replace=False)
        xs = []
        for t in sample_t:
            lo, hi = self.node_range(t)
            xs.append(self.store.features(np.arange(lo, hi), np.full(hi - lo, t)))
        self.pre = Preproc.fit(np.vstack(xs), self.store.names, cfg.exclude)
        self.model = RippleGNN(self.pre.dim, self.edge_types, cfg.hidden, cfg.layers, cfg.heads,
                               cfg.temporal, cfg.dropout, cfg.num_emb).to(cfg.device)  # fmt: skip
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        starts = train_t[cfg.seq_len - 1 :: cfg.start_stride]
        steps_per_epoch = int(np.ceil(len(starts) / cfg.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
        )
        best, best_state, bad = np.inf, None, 0
        val_rows = self._val_rows()
        for ep in range(cfg.epochs):
            self.model.train()
            perm = rng.permutation(starts)
            t0, tot, nb = time.time(), 0.0, 0
            for i in range(0, len(perm), cfg.batch):
                t_last = perm[i : i + cfg.batch] + rng.integers(0, cfg.start_stride) * SLOT
                loss = 0.0
                for _k, qd, qa, tgt, y, *_ in self.run_sequences(t_last, train=True):
                    m = torch.from_numpy(tgt).to(cfg.device)
                    if m.sum() == 0:
                        continue
                    yt = torch.from_numpy(np.clip(y, -60, 600) / SCALE).float().to(cfg.device)
                    loss = loss + pinball_loss(qd[m], yt[m, 0]) + pinball_loss(qa[m], yt[m, 1])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.detach())
                nb += 1
            val = self.validate(val_rows)
            self.history.append(
                {"epoch": ep, "train_loss": tot / nb, **val, "secs": time.time() - t0}
            )
            log.info("epoch %d loss %.4f val crps %.3f (%.0fs)", ep, tot / nb, val["val_crps"],
                     time.time() - t0)  # fmt: skip
            if val["val_crps"] < best - 1e-3:
                best, bad = val["val_crps"], 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def _val_rows(self, max_t: int = 600) -> pd.DataFrame:
        st = self.store
        idx = np.where(self.split["val"] & ~np.isnan(st.ad))[0]
        rows = pd.concat([pd.DataFrame({"idx": idx, "t": np.floor((st.sd[idx] - 60 * h) / SLOT) * SLOT})
                          for h in config.HORIZONS])  # fmt: skip
        ts = np.unique(rows.t)
        keep = np.random.default_rng(1).choice(ts, min(max_t, len(ts)), replace=False)
        return rows[rows.t.isin(keep)].reset_index(drop=True)

    def validate(self, rows: pd.DataFrame) -> dict:
        p = self.predict(self.store, rows)
        y = self.store.targets(rows.idx.to_numpy())
        c = (metrics.crps(y[:, 0], p["dep"]) + metrics.crps(y[:, 1], p["arr"])) / 2
        return {"val_crps": c}

    # --- inference ---------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, store: FeatureStore, rows: pd.DataFrame, batch: int = 48,
                occlude: dict | None = None) -> dict[str, np.ndarray]:  # fmt: skip
        assert store is self.store, "GNN predicts on the store it was built for"
        self.model.eval()
        out = {"dep": np.zeros((len(rows), len(config.QUANTILES)), np.float32),
               "arr": np.zeros((len(rows), len(config.QUANTILES)), np.float32)}  # fmt: skip
        groups = rows.groupby("t").indices
        ts = np.array(sorted(groups))
        for i in range(0, len(ts), batch):
            tb = ts[i : i + batch]
            extra = [rows.idx.to_numpy()[groups[t]] for t in tb]
            for k, qd, qa, _tgt, _y, offs, rng_k, _att in self.run_sequences(
                tb, extra, occlude=occlude
            ):
                if k != (self.cfg.seq_len if self.cfg.temporal else 1) - 1:
                    continue
                qd, qa = qd.cpu().numpy() * SCALE, qa.cpu().numpy() * SCALE
                for b, t in enumerate(tb):
                    pos = groups[t]
                    local = rows.idx.to_numpy()[pos] - rng_k[b][0] + offs[b]
                    out["dep"][pos] = qd[local]
                    out["arr"][pos] = qa[local]
        return out

    @torch.no_grad()
    def attention(self, t: float, idx: np.ndarray) -> dict:
        """Last-layer attention (head-averaged) on edges *into* the given nodes at time t."""
        self.model.eval()
        for k, *_rest, _offs, rng_k, att in self.run_sequences(
            np.array([t]), [idx], return_attention=True
        ):
            if k != (self.cfg.seq_len if self.cfg.temporal else 1) - 1:
                continue
            lo = rng_k[0][0]
            res = {}
            for et, (ei, alpha) in att.items():
                ei = ei.cpu().numpy() + lo
                alpha = alpha.cpu().numpy()
                m = np.isin(ei[1], idx)
                res[et] = pd.DataFrame({"src": ei[0][m], "dst": ei[1][m], "alpha": alpha[m]})
            return res
        return {}

    # --- persistence -------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        cfg = asdict(self.cfg)
        cfg["device"] = "cpu"
        (path / "config.json").write_text(json.dumps(
            {"cfg": cfg, "pre": asdict(self.pre), "history": self.history}, indent=1))  # fmt: skip

    @classmethod
    def load(cls, path: Path, store: FeatureStore, split: dict) -> GNNTrainer:
        meta = json.loads((path / "config.json").read_text())
        cfg = meta["cfg"]
        cfg["device"] = os.environ.get("RIPPLE_DEVICE", "cpu")
        tr = cls(store, split, **cfg)
        tr.pre = Preproc(**meta["pre"])
        tr.history = meta["history"]
        tr.model = RippleGNN(tr.pre.dim, tr.edge_types, tr.cfg.hidden, tr.cfg.layers, tr.cfg.heads,
                             tr.cfg.temporal, tr.cfg.dropout, tr.cfg.num_emb).to(tr.cfg.device)  # fmt: skip
        tr.model.load_state_dict(torch.load(path / "model.pt", map_location=tr.cfg.device))
        return tr
