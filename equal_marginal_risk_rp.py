# region imports
from AlgorithmImports import *
from scipy.optimize import minimize
import numpy as np
# endregion

class CryingTanGorilla(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)

        # Dow Jones 30 constituents
        dow_jones_constituents = [
            "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
            "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK",
            "MSFT", "NKE", "NVDA", "PG", "CRM", "TRV", "UNH", "VZ", "V",
            "WBA", "WMT"
        ]

        # Initialize symbols as a list of Symbol objects
        self.symbols = [self.AddEquity(ticker).Symbol for ticker in dow_jones_constituents]

        # Schedule the rebalance method each Monday at 8 AM
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(8, 0), self.rebalance)

        # Set the warm-up period to ensure sufficient historical data
        self.SetWarmUp(253, Resolution.Daily)

    def rebalance(self): 
        # Fetch historical returns
        history = self.History(self.symbols, 253, Resolution.Daily)
        if history.empty:
            self.Debug("History is empty; cannot rebalance.")
            return

        # Reshape the data
        close_prices = history.close.unstack(level=0)
        if close_prices.empty:
            self.Debug("Close prices are empty; cannot rebalance.")
            return

        # Ensure the columns are in the same order as self.symbols
        close_prices = close_prices[self.symbols]

        # Compute returns
        ret = close_prices.pct_change().dropna()
        if ret.empty:
            self.Debug("Returns are empty; cannot rebalance.")
            return

        # Compute covariance matrix
        cov = ret.cov().values  # Convert to NumPy array for optimization
        if np.isnan(cov).any():
            self.Debug("Covariance matrix contains NaNs; cannot rebalance.")
            return

        # Number of assets
        num_assets = len(self.symbols)

        # Initial weights
        x0 = np.ones(num_assets) / num_assets

        # Constraints: sum of weights equals 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds for weights: between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Define the function to calculate risk contributions
        def risk_contribution(w, cov_matrix):
            portfolio_var = np.dot(w.T, np.dot(cov_matrix, w))
            portfolio_std = np.sqrt(portfolio_var)
            # Avoid division by zero
            if portfolio_std == 0:
                return np.zeros_like(w)
            # Marginal Risk Contribution
            mrc = np.dot(cov_matrix, w)
            # Risk contribution of each asset
            rc = w * mrc / portfolio_std
            return rc

        # Define the objective function to minimize
        def objective_function(w):
            rc = risk_contribution(w, cov)
            avg_rc = np.mean(rc)
            return np.sum((rc - avg_rc) ** 2)

        # Perform the optimization
        opt = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-8
        )

        # Check if optimization was successful
        if opt.success:
            weights = opt.x
            # Set holdings
            targets = [PortfolioTarget(symbol, weight) for symbol, weight in zip(self.symbols, weights)]
            self.SetHoldings(targets)
            self.Debug("Rebalanced portfolio with EMRC weights.")
        else:
            self.Debug('Optimization failed: ' + opt.message)

    def OnData(self, data: Slice):
        # Ensure we only set holdings after warm-up is complete
        if self.IsWarmingUp:
            return

        # Check if we're invested; if not, trigger rebalance
        if not self.Portfolio.Invested:
            self.rebalance()
