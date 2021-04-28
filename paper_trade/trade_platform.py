import alpaca_trade_api as tradeapi
import requests
from collections import defaultdict
import config

class TradePlatform():
    def __init__(self):
        self.alpaca = tradeapi.REST(
            base_url=config.BASE_URL,
            key_id=config.KEY_ID,
            secret_key=config.SECRET_KEY
        )
        self.account = self.alpaca.get_account()
        self.portfolio_value = float(self.account.portfolio_value)
        self.portfolio = self.alpaca.list_positions()
        print(self.portfolio)

    def cancel_existing_orders(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)
    
    def execute_action(self, action, tickers):
        # action should be action as returned by TradeEnv, already softmaxed
        # tickers should be list of tickers, corresponding to actions
        # ie action[0] should be the action on ticker[0], and action[-1] should be leftover cash bal
        new_holdings_in_cash = self.portfolio_value * action[:-1]
        new_cash = self.portfolio_value * action[-1]

        current_notional = defaultdict(0.)
        notional_delta = dict()

        # just filling current_notional to get current $ in a stock
        for position in self.portfolio:
            current_notional[position.symbol] = float(position.market_value)

        # now getting the notional delta to figure out how much to buy or sell
        for idx in range(len(tickers)):
            ticker = tickers[idx]
            a = action[i]
            new_notional = self.portfolio_value * a

            # how much we want - how much we have
            # + means buy, - means sell
            notional_delta[ticker] = new_notional - current_notional[ticker]

        # ok now we execute all the trades on the platform
        # not really sure how to deal with slippage here.. TODO
        success = []
        for ticker, delta in notional_delta.items():
            if delta == 0: continue # don't do anything if delta is 0
            notional = abs(delta)
            side = 'buy' if delta > 0 else 'sell'
            success.append(self.submit_order(ticker, notional, side))

        return success


    def submit_order(self, ticker, notional, side):
        try:
            self.alpaca.submit_order(symbol=ticker, notional=notional, side=side, type="market", time_in_force="day")
            print(f"Market order of | {side} ${notional} {ticker} | completed.")
            return True
        except:
            print(f"Order of | {side} ${notional} {ticker} | did not go through.")
            return False
        

if __name__ == '__main__':
    p = TradePlatform()