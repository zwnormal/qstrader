import os

import joblib
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
from qstrader.signals.gru import GRUSignal
from tensorflow.keras.models import load_model


class GRUAlphaModel(AlphaModel):
    def __init__(self, signals, universe, data_handler, lookback):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.lookback = lookback

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= lookback:
            assets = self.signals["gru"].assets
            decision = self.signals["gru"]("EQ:SPY", lookback)

            if decision == -1:
                weights["EQ:SPY"] = 0.0
                weights["EQ:USDC-USD"] = 1 - weights[assets[0]]
            elif decision == 1:
                weights["EQ:SPY"] = 0.8
                weights["EQ:USDC-USD"] = 1 - weights[assets[0]]
            else:
                return {}
        return weights


if __name__ == "__main__":
    start_dt = pd.Timestamp("2018-10-08 14:30:00", tz=pytz.UTC)
    burn_in_dt = pd.Timestamp("2019-07-08 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2024-07-10 23:59:00", tz=pytz.UTC)

    lookback = 365

    strategy_symbols = ["SPY", "USDC-USD"]
    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", ".")
    strategy_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=strategy_symbols
    )
    strategy_data_handler = BacktestDataHandler(
        strategy_universe, data_sources=[strategy_data_source]
    )

    model_filename = os.path.join(csv_dir, "BTC_GRU_Model.keras")
    model = load_model(model_filename)
    encoder_filename = os.path.join(csv_dir, "BTC_GRU_Encoder.pkl")
    one_hot_encoder = joblib.load(encoder_filename)

    gru = GRUSignal(
        start_dt,
        strategy_universe,
        lookbacks=[lookback],
        model=model,
        encoder=one_hot_encoder,
    )
    signals = SignalsCollection({"gru": gru}, strategy_data_handler)

    strategy_alpha_model = GRUAlphaModel(
        signals, strategy_universe, strategy_data_handler, lookback
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
        title="Cryptocurrency TAA",
    )
    tearsheet.plot_results()
