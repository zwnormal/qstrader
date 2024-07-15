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
from qstrader.signals.momentum import MomentumSignal


class MomentumAlphaModel(AlphaModel):
    def __init__(self, signals, universe, data_handler):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= 30:
            assets = self.signals["momenta"].assets
            momenta_month, momentum_of_momentum = self.signals["momenta"](assets[0], 30)

            print((momenta_month, momentum_of_momentum))

            if momentum_of_momentum <= -0.06:
                weights[assets[0]] = 0.0
                weights["EQ:USDC-USD"] = 1 - weights[assets[0]]
            elif 0.1 <= momenta_month < 1.0 and 0.1 <= momentum_of_momentum < 1.0:
                weights[assets[0]] = 0.8
                weights["EQ:USDC-USD"] = 1 - weights[assets[0]]
            else:
                return {}
        return weights


if __name__ == "__main__":
    start_dt = pd.Timestamp("2017-11-09 14:30:00", tz=pytz.UTC)
    burn_in_dt = pd.Timestamp("2017-12-09 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2024-07-10 23:59:00", tz=pytz.UTC)

    strategy_symbols = ["ETH-USD", "USDC-USD"]
    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", ".")
    strategy_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=strategy_symbols
    )
    strategy_data_handler = BacktestDataHandler(
        strategy_universe, data_sources=[strategy_data_source]
    )

    momenta = MomentumSignal(start_dt, strategy_universe, lookbacks=[30])
    signals = SignalsCollection({"momenta": momenta}, strategy_data_handler)

    strategy_alpha_model = MomentumAlphaModel(
        signals, strategy_universe, strategy_data_handler
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
        title="Cryptocurrency Momenta TAA",
    )
    tearsheet.plot_results()
