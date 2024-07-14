import operator
import os

import pandas as pd
import pytz

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.signals.cdf import CDFSignal


class TopNCPPAlphaModel(AlphaModel):
    def __init__(self, signals, lookback, top_n, universe, data_handler):
        self.signals = signals
        self.lookback = lookback
        self.top_n = top_n
        self.universe = universe
        self.data_handler = data_handler

    def _highest_probability_assets(self, dt):
        assets = self.signals["cdf"].assets
        all_probabilities = {
            asset: 1 - self.signals["cdf"](asset, self.lookback) for asset in assets if asset != "EQ:USDC-USD"
        }
        return [
            asset
            for asset in sorted(
                all_probabilities.items(), key=operator.itemgetter(1), reverse=True
            )
        ][: self.top_n]

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= self.lookback:
            top_assets = self._highest_probability_assets(dt)
            if len(top_assets) == 1:
                asset = top_assets[0]
                weights[asset[0]] = asset[1] if asset[1] >= 0.5 else 0.0
                weights["EQ:USDC-USD"] = 1 - weights[asset[0]]
            else:
                all_probabilities = [asset[1] for asset in top_assets]
                max_probability = max(all_probabilities)
                sum_of_probabilities = sum(all_probabilities)
                for asset in top_assets:
                    weights[asset[0]] = (asset[1] / sum_of_probabilities) * max_probability
                weights["EQ:USDC-USD"] = 1 - max_probability
        return weights


if __name__ == "__main__":
    start_dt = pd.Timestamp("2014-09-17 14:30:00", tz=pytz.UTC)
    burn_in_dt = pd.Timestamp("2017-09-17 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2024-07-10 23:59:00", tz=pytz.UTC)

    # Model parameters
    lookback = 365 * 3
    top_n = 1

    strategy_symbols = ["BTC-USD", "USDC-USD"]
    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", ".")
    strategy_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=strategy_symbols
    )
    strategy_data_handler = BacktestDataHandler(
        strategy_universe, data_sources=[strategy_data_source]
    )

    cdf = CDFSignal(start_dt, strategy_universe, lookbacks=[lookback])
    signals = SignalsCollection({"cdf": cdf}, strategy_data_handler)

    strategy_alpha_model = TopNCPPAlphaModel(
        signals, lookback, top_n, strategy_universe, strategy_data_handler
    )

    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance="daily",
        long_only=True,
        cash_buffer_percentage=0.01,
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler,
    )
    strategy_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        title="Cryptocurrency CDF TAA",
    )
    tearsheet.plot_results()
