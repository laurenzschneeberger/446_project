# region imports
from AlgorithmImports import *
from scipy.optimize import minimize
# endregion


class CryingTanGorilla(QCAlgorithm):

    def Initialize(self):
        # Correct method name capitalization
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 12)
        self.SetCash(100000)

        # Initialize symbols as a list of Symbol objects
        dow_jones_constituents = ["MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "NVDA", "PG", "CRM", "TRV", "UNH", "VZ", "V", "WBA", "WMT"]
        self.symbols = [self.AddEquity(ticker).Symbol for ticker in dow_jones_constituents]

        # Schedule the rebalance method each Monday at 8 AM
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(8, 0), self.rebalance)

    def OnData(self, data: Slice):
        # Correct capitalization for Portfolio and SetHoldings
        if not self.Portfolio.Invested:
            for symbol in self.symbols:
                self.SetHoldings(symbol, 1/len(self.symbols))  # Allocate 14.2% to each symbol initially
    
    def rebalance(self): 

        # A matrix containing returs per ticker, columns are tickers
        ret = self.history(self.symbols, 253, Resolution.DAILY).close.unstack(0).pct_change().dropna()

        # Optimization
        x0 = [1/ret.shape[1]] * ret.shape[1]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * ret.shape[1]
        opt = minimize(lambda w: 0.5 * (w.T @ ret.cov() @ w) - x0 @ w, x0=x0, constraints=constraints, bounds=bounds, tol=1e-8, method="SLSQP")

        # Rebalance
        self.set_holdings([PortfolioTarget(symbol, weight) for symbol, weight in zip(ret.columns, opt.x)])
