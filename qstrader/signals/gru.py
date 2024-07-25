import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from qstrader.signals.signal import Signal


def std_normalized(d: pd.Series) -> pd.Series:
    series_frame = d.to_frame()
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series_frame)
    scaled_series = pd.Series(scaled_series.flatten(), index=d.index, name="scaled")
    return scaled_series


def calculate_simple_moving_average(series: pd.Series, n: int = 20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).mean()


def calculate_macd_oscillator(
    series: pd.Series, n1: int = 5, n2: int = 34
) -> pd.Series:
    """
    Calculate the moving average convergence divergence oscillator, given a
    short moving average of length n1 and a long moving average of length n2
    """
    assert n1 < n2, f"n1 must be less than n2"
    return calculate_simple_moving_average(
        series, n1
    ) - calculate_simple_moving_average(series, n2)


def calculate_return_momentum(series: pd.Series, n: int = 20) -> pd.Series:
    return series.rolling(n).apply(
        lambda x: (np.cumprod(1.0 + x.pct_change()) - 1.0).iloc[-1]
    )


def calculate_simple_moving_sample_stdev(series: pd.Series, n: int = 20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).std()


def calculate_return_series(series: pd.Series) -> pd.Series:
    """
    Calculates the return series of a given time series.
    The first value will always be NaN.
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1


def calculate_rolling_sharpe_ratio(price_series: pd.Series, n: int = 20) -> pd.Series:
    """
    Compute an approximation of the Sharpe ratio on a rolling basis.
    Intended for use as a preference value.
    """
    rolling_return_series = calculate_return_series(price_series).rolling(n)
    return rolling_return_series.mean() / rolling_return_series.std()


def calculate_simple_moving_median(series: pd.Series, n: int = 20) -> pd.Series:
    return series.rolling(n).median()


class GRUSignal(Signal):
    def __init__(self, start_dt, universe, lookbacks, model, encoder):
        bumped_lookbacks = [lookback + 1 for lookback in lookbacks]
        super().__init__(start_dt, universe, bumped_lookbacks)
        self.model = model
        self.encoder = encoder
        self.last_prediction = ""
        self.last_decision = 0
        self.continuous_positives = 0

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        return "%s_%s" % (asset, lookback + 1)

    @staticmethod
    def _generate_metrics_dataframe(
        df: pd.DataFrame,
        fast_period: int = 14,
        slow_period: int = 84,
        start: int | None = None,
        end: int | None = None,
    ) -> pd.DataFrame:
        assert fast_period < slow_period

        close_prices = df[start:end]["Close"]
        df_result = pd.DataFrame(index=df.index)
        df_result["macd_oscillator"] = std_normalized(
            calculate_macd_oscillator(close_prices, n1=fast_period, n2=slow_period)
        )
        df_result["return_momentum_slow"] = std_normalized(
            calculate_return_momentum(close_prices, n=slow_period)
        )
        df_result["return_momentum_fast"] = std_normalized(
            calculate_return_momentum(close_prices, n=fast_period)
        )
        df_result["stdev_slow"] = std_normalized(
            calculate_simple_moving_sample_stdev(close_prices, n=slow_period)
        )
        df_result["stdev_fast"] = std_normalized(
            calculate_simple_moving_sample_stdev(close_prices, n=fast_period)
        )
        df_result["sharpe_ratio_slow"] = std_normalized(
            calculate_rolling_sharpe_ratio(close_prices, n=slow_period)
        )
        df_result["sharpe_ratio_fast"] = std_normalized(
            calculate_rolling_sharpe_ratio(close_prices, n=fast_period)
        )
        df_result["median_slow"] = std_normalized(
            calculate_simple_moving_median(close_prices, n=slow_period)
        )
        df_result["median_fast"] = std_normalized(
            calculate_simple_moving_median(close_prices, n=fast_period)
        )
        return df_result

    def __call__(self, asset, lookback):
        df = pd.Series(
            self.buffers.prices[self._asset_lookback_key(asset, lookback)]
        ).to_frame("Close")

        df_metrics = self._generate_metrics_dataframe(
            df,
            fast_period=14,
            slow_period=168,
            start=-168 - 14 - 1,
        )
        del df_metrics["sharpe_ratio_fast"]
        del df_metrics["sharpe_ratio_slow"]
        x = np.expand_dims(df_metrics.tail(14).values, axis=0)

        predictions = self.model.predict(x)
        inverse_predictions = self.encoder.inverse_transform(predictions)
        predicted_confidences = np.max(predictions, axis=1)

        predict_results = []
        for i in range(len(predictions)):
            predicted_class = inverse_predictions[i][0]
            predict_results.append((predicted_class, predicted_confidences[i]))

        print((predict_results[0][0], predict_results[0][1]))
        decision = 0
        if predict_results[0][0] in ["recovery", "markup"]:
            self.continuous_positives += 1
        else:
            self.continuous_positives = 0

        if (
            self.last_prediction == "recovery"
            and predict_results[0][0] == "markup"
            and predict_results[0][1] > 0.85
            and self.last_decision != 1
        ):
            decision = 1
            self.continuous_positives = 0
        elif (
            predict_results[0][0] in ["distribution"]
            and predict_results[0][1] > 0.70
        ):
            decision = -1
        self.last_prediction = predict_results[0][0]
        self.last_decision = decision

        return decision
