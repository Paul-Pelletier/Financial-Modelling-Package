import pandas as pd
import numpy as np

class IVPreprocessor:
    def __init__(self, data, spot_col = 'UNDERLYING_LAST', strike_col = 'STRIKE', call_iv_col = 'C_IV', put_iv_col = 'P_IV', call_vol_col = 'C_VOLUME', put_vol_col = 'P_VOLUME'):
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
        self.data = data
        self.spot_col = spot_col
        self.strike_col = strike_col
        self.call_iv_col = call_iv_col
        self.put_iv_col = put_iv_col
        self.call_vol_col = call_vol_col
        self.put_vol_col = put_vol_col

    def preprocess(self, call_limits, put_limits, mode="overlap"):
        """
        Preprocess the data to select calls and puts based on Strike/Spot limits, and concatenate or split the data.

        Parameters:
        - call_limits (tuple): A tuple (lower_limit, upper_limit) for Strike/Spot selection of calls.
        - put_limits (tuple): A tuple (lower_limit, upper_limit) for Strike/Spot selection of puts.
        - mode (str): Either "overlap" to combine calls and puts or "split" to separate them.

        Returns:
        - pd.DataFrame: A DataFrame containing Log Moneyness, implied volatilities, and volumes.
        """
        # Calculate Strike/Spot ratio
        self.data = self.data.copy()
        self.data.loc[:, 'Strike/Spot'] = self.data[self.strike_col] / self.data[self.spot_col]

        if mode == "overlap":
            # Select calls and puts based on the overlapping limits
            combined_data = self.data[
                ((self.data['Strike/Spot'] >= call_limits[0]) & (self.data['Strike/Spot'] <= call_limits[1])) |
                ((self.data['Strike/Spot'] >= put_limits[0]) & (self.data['Strike/Spot'] <= put_limits[1]))
            ]

            # Prepare call data
            call_data = combined_data[[
                self.strike_col, self.spot_col, self.call_iv_col, self.call_vol_col
            ]].rename(columns={
                self.call_iv_col: "Implied_Volatility",
                self.call_vol_col: "Volume"
            })
            call_data['Option Type'] = 'Call'

            # Prepare put data
            put_data = combined_data[[
                self.strike_col, self.spot_col, self.put_iv_col, self.put_vol_col
            ]].rename(columns={
                self.put_iv_col: "Implied_Volatility",
                self.put_vol_col: "Volume"
            })
            put_data['Option Type'] = 'Put'

            # Concatenate call and put data
            combined_data = pd.concat([call_data, put_data], ignore_index=True)

        elif mode == "split":
            # Select calls based on the limits
            call_data = self.data[
                (self.data['Strike/Spot'] >= call_limits[0]) &
                (self.data['Strike/Spot'] <= call_limits[1])
            ][[self.strike_col, self.spot_col, self.call_iv_col, self.call_vol_col]]

            call_data = call_data.rename(columns={
                self.call_iv_col: 'Implied_Volatility',
                self.call_vol_col: 'Volume'
            })
            call_data['Option Type'] = 'Call'

            # Select puts based on the limits
            put_data = self.data[
                (self.data['Strike/Spot'] >= put_limits[0]) &
                (self.data['Strike/Spot'] <= put_limits[1])
            ][[self.strike_col, self.spot_col, self.put_iv_col, self.put_vol_col]]

            put_data = put_data.rename(columns={
                self.put_iv_col: 'Implied_Volatility',
                self.put_vol_col: 'Volume'
            })
            put_data['Option Type'] = 'Put'

            # Concatenate calls and puts
            combined_data = pd.concat([call_data, put_data], ignore_index=True)

        else:
            raise ValueError("Invalid mode. Choose either 'overlap' or 'split'.")

        # Calculate Log Moneyness
        combined_data['Log_Moneyness'] = np.log(combined_data[self.strike_col] / combined_data[self.spot_col])

        # Select final columns
        final_data = combined_data[['Log_Moneyness', 'Implied_Volatility', 'Volume', 'Option Type']]
        # Drop rows where Implied_Volatility is NaN
        final_data = final_data.dropna(subset=["Implied_Volatility"])
        # Convert Volume from string to integer
        final_data["Volume"] = pd.to_numeric(final_data["Volume"].str.strip(), errors="coerce").fillna(0).astype(int)
        #Drop rows where Volume == 0
        final_data = final_data[final_data["Volume"] != 0]

        return final_data