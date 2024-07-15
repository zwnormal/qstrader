import pandas as pd
from qstrader.signals.signal import Signal


class MACDSignal(Signal):
    """
    Indicator class to calculate macd oscillator
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

    @classmethod
    def calculate_simple_moving_average(cls, series, n: int = 20) -> pd.Series:
        """Calculates the simple moving average"""
        return pd.Series(series).rolling(n).mean()

    @classmethod
    def calculate_macd_oscillator(cls, series, n1: int = 5, n2: int = 34) -> pd.Series:
        """
        Calculate the moving average convergence divergence oscillator, given a
        short moving average of length n1 and a long moving average of length n2
        """
        assert n1 < n2, f"n1 must be less than n2"
        return cls.calculate_simple_moving_average(
            series, n1
        ) - cls.calculate_simple_moving_average(series, n2)

    def __call__(self, asset, lookback):
        price_series = self.buffers.prices["%s_%s" % (asset, lookback)]
        macd_oscillator = self.calculate_macd_oscillator(price_series)
        max_macd_oscillator = macd_oscillator.max()
        min_macd_oscillator = macd_oscillator.min()
        current_macd_oscillator = macd_oscillator.iloc[-1]
        if current_macd_oscillator < 0:
            return 1.0 - float(
                (abs(min_macd_oscillator) - abs(current_macd_oscillator))
                / (max_macd_oscillator - min_macd_oscillator)
            )
        else:
            return float(
                (max_macd_oscillator - current_macd_oscillator)
                / (max_macd_oscillator - min_macd_oscillator)
            )
