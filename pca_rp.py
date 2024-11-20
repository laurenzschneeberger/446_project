from AlgorithmImports import *
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

class PcaRiskParity(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2022, 12, 12)
        self.set_cash(100000)

        # Dow 30 symbols
        self.dow_symbols = [
            'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DWDP', 'GS', 'HD',
            'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
            'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WBA', 'WMT', 'XOM'
        ]

        # Add symbols to universe
        for symbol in self.dow_symbols:
            self.add_equity(symbol, Resolution.Minute)
        
        # Initialize parameters
        self.num_components = 5
        self.lookback_days = 5  # One trading week
        self.last_rebalance_date = None  # Track last rebalance date
        self.pca = PCA(n_components=self.num_components)
        
    def should_rebalance(self):
        """Check if we should rebalance the portfolio"""
        # Rebalance at the start of Monday's market hours
        current_time = self.Time
        
        is_monday = current_time.weekday() == 0
        is_market_open = current_time.hour == 9 and current_time.minute == 31  # Just after market open
        
        # Check if we haven't rebalanced today
        not_rebalanced_today = (self.last_rebalance_date is None or 
                               self.last_rebalance_date.date() != current_time.date())
        
        return is_monday and is_market_open and not_rebalanced_today

    def rebalance(self):
        """Compute PCA and rebalance portfolio weights"""
        try:
            # Get historical data
            history = self.history(self.dow_symbols, self.lookback_days * 390, Resolution.Minute)
            
            if history.empty:
                self.log("Not enough historical data for rebalancing")
                return
            
            # Compute log returns
            closes = history.close.unstack(level=0)
            log_returns = np.log(closes / closes.shift(1)).dropna()
            
            if log_returns.empty:
                self.log("Not enough data for computing returns")
                return
            
            # Compute covariance matrix
            cov_matrix = log_returns.cov()
            
            # Perform PCA
            self.pca.fit(log_returns)
            loadings = self.pca.components_  # Shape: (num_components, num_stocks)
            
            # Compute risk parity weights
            pc_volatilities = np.sqrt(self.pca.explained_variance_)
            target_risk_contribution = 1.0 / self.num_components
            
            # Initialize weights
            weights = np.zeros(len(self.dow_symbols))
            
            # For each principal component
            for i in range(self.num_components):
                # Scale the loadings by the inverse of PC volatility to achieve equal risk contribution
                component_weights = loadings[i] / pc_volatilities[i]
                weights += component_weights * target_risk_contribution
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(np.abs(weights))
            
            # Rebalance portfolio
            for symbol, weight in zip(self.dow_symbols, weights):
                self.set_holdings(symbol, weight)
            
            # Update last rebalance date
            self.last_rebalance_date = self.Time
            
            self.log(f"Portfolio rebalanced. Weights: {dict(zip(self.dow_symbols, weights))}")
            
            # Log the explained variance ratios
            explained_variance = self.pca.explained_variance_ratio_
            self.log(f"Explained variance ratios: {explained_variance}")
            
        except Exception as e:
            self.error(f"Error in rebalance: {str(e)}")

    def on_data(self, data: Slice):
        """
        OnData event is the primary entry point for your algorithm.
        Check if we need to rebalance the portfolio.
        """
        if self.should_rebalance():
            self.log("Starting weekly rebalance...")
            self.rebalance()

    def log(self, message):
        """Helper method to log messages with timestamps"""
        self.debug(f"{self.Time}: {message}")