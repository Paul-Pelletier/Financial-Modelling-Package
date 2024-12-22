import numpy as np
import pandas as pd
from financial_modelling.data_pre_processing.Preprocessor import Preprocessor

class IVPreprocessor(Preprocessor):
    def __init__(self, data, spot_col='UNDERLYING_LAST', strike_col='STRIKE',
                 call_iv_col='C_IV', put_iv_col='P_IV', call_vol_col='C_VOLUME', put_vol_col='P_VOLUME'):
        """
        Initialize the IVPreprocessor with the input DataFrame and column names.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing all the data.
        - spot_col (str): Column name for the spot price.
        - strike_col (str): Column name for the strike prices.
        - call_iv_col (str): Column name for implied volatilities of calls.
        - put_iv_col (str): Column name for implied volatilities of puts.
        - call_vol_col (str): Column name for volumes of calls.
        - put_vol_col (str): Column name for volumes of puts.
        """
        super().__init__(data)
        self.spot_col = spot_col
        self.strike_col = strike_col
        self.call_iv_col = call_iv_col
        self.put_iv_col = put_iv_col
        self.call_vol_col = call_vol_col
        self.put_vol_col = put_vol_col

    def preprocess(self, call_limits, put_limits, volume_limits=1, mode="overlap"):
        """
        Preprocess the data to select calls and puts based on Strike/Spot limits, and concatenate or split the data.

        Parameters:
        - call_limits (tuple): A tuple (lower_limit, upper_limit) for Strike/Spot selection of calls.
        - put_limits (tuple): A tuple (lower_limit, upper_limit) for Strike/Spot selection of puts.
        - volume_limits (int): Minimum volume threshold to include in the final data.
        - mode (str): Either "overlap" to combine calls and puts or "split" to separate them.

        Returns:
        - pd.DataFrame: A DataFrame containing Log Moneyness, implied volatilities, and volumes.
        """
        self.validate_data([self.spot_col, self.strike_col, self.call_iv_col, self.put_iv_col,
                            self.call_vol_col, self.put_vol_col, 'QUOTE_UNIXTIME', 'EXPIRE_UNIX'])

        # Calculate Strike/Spot ratio
        self.data['Strike/Spot'] = self.data[self.strike_col] / self.data[self.spot_col]
        self.data['Residual_Maturity'] = (self.data['EXPIRE_UNIX'].astype(float) - self.data['QUOTE_UNIXTIME'].astype(float))/31_536_000

        if mode == "overlap":
            # Select calls and puts based on the overlapping limits
            combined_data = self.data[
                ((self.data['Strike/Spot'] >= call_limits[0]) & (self.data['Strike/Spot'] <= call_limits[1])) |
                ((self.data['Strike/Spot'] >= put_limits[0]) & (self.data['Strike/Spot'] <= put_limits[1]))
            ]

            # Prepare call data
            call_data = combined_data[[self.strike_col, self.spot_col, self.call_iv_col, self.call_vol_col, 'Residual_Maturity']]
            call_data = call_data.rename(columns={self.call_iv_col: "Implied_Volatility", self.call_vol_col: "Volume"})
            call_data['Option Type'] = 'Call'

            # Prepare put data
            put_data = combined_data[[self.strike_col, self.spot_col, self.put_iv_col, self.put_vol_col, 'Residual_Maturity']]
            put_data = put_data.rename(columns={self.put_iv_col: "Implied_Volatility", self.put_vol_col: "Volume"})
            put_data['Option Type'] = 'Put'

            # Concatenate call and put data
            combined_data = pd.concat([call_data, put_data], ignore_index=True)

        elif mode == "split":
            # Select calls based on the limits
            call_data = self.data[
                (self.data['Strike/Spot'] >= call_limits[0]) & (self.data['Strike/Spot'] <= call_limits[1])
            ][[self.strike_col, self.spot_col, self.call_iv_col, self.call_vol_col, 'Residual_Maturity']]

            call_data = call_data.rename(columns={self.call_iv_col: 'Implied_Volatility', self.call_vol_col: 'Volume'})
            call_data['Option Type'] = 'Call'

            # Select puts based on the limits
            put_data = self.data[
                (self.data['Strike/Spot'] >= put_limits[0]) & (self.data['Strike/Spot'] <= put_limits[1])
            ][[self.strike_col, self.spot_col, self.put_iv_col, self.put_vol_col, 'Residual_Maturity']]

            put_data = put_data.rename(columns={self.put_iv_col: 'Implied_Volatility', self.put_vol_col: 'Volume'})
            put_data['Option Type'] = 'Put'

            # Concatenate calls and puts
            combined_data = pd.concat([call_data, put_data], ignore_index=True)

        else:
            raise ValueError("Invalid mode. Choose either 'overlap' or 'split'.")

        # Calculate Log Moneyness
        combined_data['Log_Moneyness'] = np.log(combined_data[self.strike_col] / combined_data[self.spot_col])
        

        # Select final columns
        final_data = combined_data[['Log_Moneyness', 'Implied_Volatility', 'Volume', 'Option Type', 'Residual_Maturity']]

        # Drop rows where Implied Volatility is NaN or Volume < volume_limits
        final_data = final_data.dropna(subset=['Implied_Volatility'])
        final_data = final_data[final_data['Implied_Volatility'].str.replace(",", ".").astype(float) > 0.05]
        final_data['Volume'] = pd.to_numeric(final_data['Volume'], errors='coerce').fillna(0).astype(int)
        final_data = final_data[final_data['Volume'] >= volume_limits]

        return final_data
