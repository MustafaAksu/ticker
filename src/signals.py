# src/signals.py
def compute_rfs(F, mcap):
    net_in = F.sum(0) - F.sum(1)
    rfs = net_in / mcap.values
    return (rfs - rfs.mean()) / rfs.std()  # z-score