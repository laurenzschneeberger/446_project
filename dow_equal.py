# region imports
from AlgorithmImports import *
# endregion

class EmotionalBrownDinosaur(QCAlgorithm):

    def Initialize(self):
        # Set the start and end dates for the backtest
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 12)
        self.SetCash(100000)  # Set initial capital

        # Add Dow Jones Industrial Average (DJIA) as a tradable index
        self.index = self.AddEquity("DIA", Resolution.Minute).Symbol  # Use the SPDR Dow Jones ETF as a proxy

    def OnData(self, data: Slice):
        # Check if the portfolio is not invested
        if not self.Portfolio.Invested:
            self.SetHoldings(self.index, 1)  # Invest 100% of the portfolio in DJIA proxy (DIA)