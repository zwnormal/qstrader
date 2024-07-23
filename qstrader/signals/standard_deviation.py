import pandas as pd

from qstrader.signals.signal import Signal


class StandardDeviationSignal(Signal):
    def __init__(self, start_dt, universe, lookbacks, rolling):
        bumped_lookbacks = [lookback + 1 for lookback in lookbacks]
        super().__init__(start_dt, universe, bumped_lookbacks)
        self.rolling = rolling

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return "%s_%s" % (asset, lookback + 1)

    def __call__(self, asset, lookback):
        series = pd.Series(
            self.buffers.prices[self._asset_lookback_key(asset, lookback)]
        ).dropna()

        return series.rolling(window=self.rolling, min_periods=1).std()
