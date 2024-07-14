import numpy as np
from scipy.stats import lognorm

from qstrader.signals.signal import Signal


class CDFSignal(Signal):
    """
    Indicator class to calculate cumulative price
    probability of last N periods for a set of prices.

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime (UTC) of the signal.
    universe : `Universe`
        The universe of assets to calculate the signals for.
    lookbacks : `list[int]`
        The number of lookback periods to store prices for.
    """

    def __init__(self, start_dt, universe, lookbacks):
        super().__init__(start_dt, universe, lookbacks)

    def _cumulative_price_probability(self, asset, lookback):
        price_series = self.buffers.prices["%s_%s" % (asset, lookback)]
        log_price_series = np.log(price_series)
        mean_log_price = log_price_series.mean()
        std_log_price = log_price_series.std()

        shape = std_log_price
        loc = 0
        scale = np.exp(mean_log_price)

        return lognorm.cdf(price_series[-1], shape, loc=loc, scale=scale)

    def __call__(self, asset, lookback):
        return float(self._cumulative_price_probability(asset, lookback))
