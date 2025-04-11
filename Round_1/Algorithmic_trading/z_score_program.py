from typing import Dict, List, Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Position
import json
f=1
def debug_print(a,f):
    if f:
        print(a)

import numpy as np

def log_returns(prices):
    return np.diff(np.log(prices))


def volatility(prices, window=20):
    returns = log_returns(prices)
    daily_vol = np.std(returns[-window:])      # Std dev of recent returns
    return daily_vol * np.sqrt(252)            # Annualized vol (252 trading days)

POS_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


traderdata_init = {
    "KELP" : {
        "LAST_N_MID_VALUES": [],
        "LAST_N_WEIGHTED_AVG_VALUES": []
        },

    "RAINFOREST_RESIN" : {
        "LAST_MID_VALUE" : 0                ## WEIGHTED AVG
    },

    "SQUID_INK" : {
        "LAST_N_MID_VALUES": []
        },
    "EARNED_MONEY" : 0,
    "DEBUG_PRINT" : ""
}

params = {
    "KELP" : {
        "HISTORY_LENGTH(N)" : 7
        },

    "RAINFOREST_RESIN" : {
        "EDGE_FROM_PRED": 1
    },

    "SQUID_INK" : {
        "HISTORY_LENGTH(N)" : 300
        }
}



class Trader:

    def run_raisin(self, order_depths: OrderDepth, position: Position) -> List[Order]:
        symbol = "RAINFOREST_RESIN"
        orders: List[Order] = []
        buy_at_less_than = 10000
        sell_at_more_than = 10000
        sell_orders_placed_quantity = 0
        buy_orders_placed_quantity = 0

        # Order(symbol, )

        buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depths.sell_orders.items())

        buy_sell_limit = 50 #Do not place orders of quantity more than this to manage product placement


        total_vol_for_instant_sell = 0
        best_bid, best_bid_amt = buy_orders[0]
        total_vol_for_instant_sell = sum(x[1] for x in order_depths.buy_orders.items() if (x[0]>sell_at_more_than))
        # print("trades we instant sell to: ", list(x for x in order_depths.buy_orders.items() if (x[0]>sell_at_more_than)), file=open("testing_out.txt", "a"))

        # if (best_bid>sell_at_more_than):
        #     total_vol_for_instant_sell = best_bid_amt
        #     print("(price, quant) ", (best_bid, best_bid_amt), file=open("testing_out.txt", "a"))
            
        selling_price_for_challenge = max(sell_at_more_than+1, min(list(x for x in order_depths.sell_orders.keys() if (x > sell_at_more_than+1)), default=sell_at_more_than+1)-1)
            
        if position-total_vol_for_instant_sell < -POS_LIMITS[symbol]:
            orders.append(Order(symbol, best_bid, -min(position+POS_LIMITS[symbol], buy_sell_limit)))
            # sell_orders_placed_quantity = min(position+POS_LIMITS[symbol], buy_sell_limit)
        else:
            orders.append(Order(symbol, best_bid, -min(total_vol_for_instant_sell, buy_sell_limit)))
            can_place_this_many_more = POS_LIMITS[symbol] + (position - min(total_vol_for_instant_sell, buy_sell_limit))
            orders.append(Order(symbol, selling_price_for_challenge, - can_place_this_many_more))
            # sell_orders_placed_quantity = min(total_vol_for_instant_sell, buy_sell_limit)
            
        total_vol_for_instant_buy = 0

        best_ask, best_ask_amt = sell_orders[0]
        assert(best_ask_amt<=0)
        if (best_ask<buy_at_less_than):
            total_vol_for_instant_buy -= best_ask_amt

        total_vol_for_instant_buy = -sum(x[1] for x in order_depths.sell_orders.items() if (x[0]<buy_at_less_than))


        buying_price_for_challenge = min(buy_at_less_than-1, max(list(x for x in order_depths.buy_orders.keys() if (x < buy_at_less_than-1)), default=buy_at_less_than-1)+1)
        # print("buying_price_for_challenge: ", buying_price_for_challenge, list(x for x in order_depths.buy_orders.keys() if (x < buy_at_less_than-1)),  order_depths.buy_orders.keys(), file=open("testing_out.txt", "a"))
        

        if total_vol_for_instant_buy+position > POS_LIMITS[symbol]:
            orders.append(Order(symbol, best_ask, min(POS_LIMITS[symbol]-position, buy_sell_limit)))
            # buy_orders_placed_quantity = min(POS_LIMITS[symbol]-position, buy_sell_limit)
        else:
            orders.append(Order(symbol, best_ask, min(total_vol_for_instant_buy, buy_sell_limit)))
            # buy_orders_placed_quantity = min(total_vol_for_instant_buy, buy_sell_limit)
            can_place_this_many_more = POS_LIMITS[symbol] - (position + min(total_vol_for_instant_buy, buy_sell_limit))
            orders.append(Order(symbol, buying_price_for_challenge, can_place_this_many_more))

        return orders
    
    def calc_PnL(self, position, current_prices,earned_money):
        ans = 0
        for symbol, pos in position.items():
            ans+= pos*current_prices[symbol]
        ans += earned_money
        return ans

    def run_kelp1(self, order_depths: OrderDepth, position: Position) -> List[Order]:
        symbol = "KELP"
        orders: List[Order] = [];
        buy_at_less_than = 2019
        sell_at_more_than = 2019

        # buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depths.sell_orders.items())

        # total_vol_for_instant_sell = 0
        # for (price, amt) in buy_orders: #We decide the best people to sell to based on the buy orders.
        #     if (price>sell_at_more_than):
        #         total_vol_for_instant_sell += amt
            
        # if position-total_vol_for_instant_sell >= -POS_LIMITS[symbol]:
        #     orders.append(Order(symbol, sell_at_more_than+1, position-POS_LIMITS[symbol]))
        # else:
        #     orders.append(Order(symbol, sell_at_more_than+1, total_vol_for_instant_sell))

        orders.append(Order(symbol, sell_at_more_than+3, -POS_LIMITS[symbol]-position)) #position+x = -pos_limit
        orders.append(Order(symbol, buy_at_less_than-4, POS_LIMITS[symbol]-position))
        return orders

    def run_kelp(self, order_depths: OrderDepth, position: Position, last_n_trades) -> List[Order]:
        symbol = "KELP"
        orders: List[Order] = [];
        if len(last_n_trades) == params["KELP"]["HISTORY_LENGTH(N)"]:
            intercept = 17.317515632852974
            coeffs = [0.02935884,0.03880124,0.06456323,0.1129974,0.11089013,0.25088261,0.38392037]
            # intercept = 17.59482603537458
            # coeffs = [0.09143235 ,0.12958247, 0.12058332, 0.25956105, 0.39011803]
            # intercept = 18.408089860604377
            # coeffs = [0.16526783 ,0.14608133, 0.27304808 ,0.4064776 ]
            my_pred = intercept
            for i in range(len(last_n_trades)):
                my_pred += last_n_trades[i]*coeffs[i] #last_n_trades: [[trade.price, trade.quantity, trade.timestamp], ...]
            # my_pred = sum(last_n_trades[-1:])/len(last_n_trades[-1:])
            buy_at_less_than = round(my_pred)
            sell_at_more_than = round(my_pred)

            # debug_print("prediction: ", my_pred, file=open("testing_out.txt", "a"))
            # debug_print("last_n_trades: ", last_n_trades, file=open("testing_out.txt", "a"))

            buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depths.sell_orders.items())

            buy_sell_limit = 50 #Do not place orders of quantity more than this to manage product placement

            total_vol_for_instant_sell = 0
            for (price, amt) in buy_orders: #We decide the best people to sell to based on the buy orders.
                # debug_print("While deciding sells, I see: ", price, amt, file=open("testing_out.txt", "a"))
                if (price>sell_at_more_than):
                    # debug_print("It is good", file=open("testing_out.txt", "a"))
                    total_vol_for_instant_sell += amt
                
            # if position-total_vol_for_instant_sell < -POS_LIMITS[symbol]:
            #     orders.append(Order(symbol, sell_at_more_than+1, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.8)))
            #     orders.append(Order(symbol, sell_at_more_than+0, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.2)))
            #     orders.append(Order(symbol, sell_at_more_than+2, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.2)))

            
            # else:
            #     orders.append(Order(symbol, sell_at_more_than+1, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.8)))
            #     orders.append(Order(symbol, sell_at_more_than+0, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.2)))
            #     orders.append(Order(symbol, sell_at_more_than+2, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.2)))

                
            total_vol_for_instant_buy = 0
            for (price, amt) in sell_orders: #We decide the best people to buy from based on the sell orders.
                assert(amt<=0)
                if (price<buy_at_less_than):
                    total_vol_for_instant_buy -= amt
            
            # if total_vol_for_instant_buy+position > POS_LIMITS[symbol]:
            #     orders.append(Order(symbol, buy_at_less_than-1, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.8)))
            #     orders.append(Order(symbol, buy_at_less_than+0, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.2)))
            #     orders.append(Order(symbol, buy_at_less_than-2, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.2)))
            # else:
            #     orders.append(Order(symbol, buy_at_less_than-1, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.8)))
            #     orders.append(Order(symbol, buy_at_less_than+0, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.2)))
            #     orders.append(Order(symbol, buy_at_less_than-2, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.2)))
            # debug_print("orders: ", orders, file=open("testing_out.txt", "a"))
            # debug_print("position: ", orders, file=open("testing_out.txt", "a"))
            # debug_print("total_vol_for_instant_buy: ", total_vol_for_instant_buy, file=open("testing_out.txt", "a"))
            # debug_print("total_vol_for_instant_sell: ", total_vol_for_instant_sell, file=open("testing_out.txt", "a"))
            orders.append(Order(symbol, sell_at_more_than+1, -POS_LIMITS[symbol]-position)) #position+x = -pos_limit
            orders.append(Order(symbol, buy_at_less_than-1, POS_LIMITS[symbol]-position))
        return orders
    
    def run_squid_ink(self, order_depths: OrderDepth, position: Position, last_n_mid_prices_inp,PnL):

        symbol = "SQUID_INK"
        orders: List[Order] = []

        if len(last_n_mid_prices_inp) < params["SQUID_INK"]["HISTORY_LENGTH(N)"]:
            return [orders, -1, -1, -1, -1]
        

        asset_vol = volatility(last_n_mid_prices_inp)
        target_vol = 0.1

        gamma = 0.1
        import math
        capital = 50000*(1+ gamma*PnL/math.sqrt(1+(gamma*PnL)**2))

        
        entry_threshold = 0.59e-2
        exit_threshold = 1.21e-2

        invest_size = round(capital*target_vol/asset_vol)
        # invest_size = min(1, abs(Z)/entry_threshold)*POS_LIMITS[symbol]                   # try this later

        curr_buy_limit = POS_LIMITS[symbol] - position
        curr_sell_limit = -POS_LIMITS[symbol] - position

        import statistics
        avg = statistics.mean(last_n_mid_prices_inp[:-1])
        std = statistics.stdev(last_n_mid_prices_inp[:-1])
        z_score = (last_n_mid_prices_inp[-1]-avg)/std
        long_price = round(avg - entry_threshold*std)
        short_price = round(avg + entry_threshold*std)

        fst_avg = statistics.mean(last_n_mid_prices_inp[:int(len(last_n_mid_prices_inp)/2)])
        snd_avg = statistics.mean(last_n_mid_prices_inp[int(len(last_n_mid_prices_inp)/2):])
        

        if position > 0 and z_score > -exit_threshold:                                    # ALREADY LONG
            orders.append(Order(symbol, long_price, -position))


        elif position < 0 and z_score < exit_threshold:                                 # ALREADY SHORT
            orders.append(Order(symbol, short_price, - position))


        elif position == 0:
            if z_score > entry_threshold:
                orders.append(Order(symbol, short_price, -min(invest_size, -curr_sell_limit)))
                # orders.append(Order(symbol, short_price, min(invest_size, -curr_sell_limit)))


            elif z_score < -entry_threshold:
                orders.append(Order(symbol, long_price, min(invest_size, curr_buy_limit)))
                # orders.append(Order(symbol, long_price, -min(invest_size, curr_buy_limit)))






        return [orders, invest_size, z_score, long_price, short_price]

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        # debug_print(state.traderData, file=open("testing_out.txt","a"))
        if state.traderData == "":
            trader_data = traderdata_init
        else:
            trader_data = json.loads(state.traderData)

        if state.timestamp == 0:
            current_prices = {"KELP": 2000, "RAINFOREST_RESIN": 10000, "SQUID_INK": 2000}
        else:
            current_prices = {"KELP": trader_data["KELP"]["LAST_N_WEIGHTED_AVG_VALUES"][-1], "RAINFOREST_RESIN": trader_data["RAINFOREST_RESIN"]["LAST_MID_VALUE"], "SQUID_INK": trader_data["SQUID_INK"]["LAST_N_MID_VALUES"][-1]}



        PnL = self.calc_PnL(state.position, current_prices, trader_data["EARNED_MONEY"])
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
                if product == "RAINFOREST_RESIN":
                    result[product] = self.run_raisin(state.order_depths[product] , state.position.get(product, 0))
                    continue
                if product == "KELP":
                    result[product] = self.run_kelp(state.order_depths[product] , state.position.get(product, 0), trader_data["KELP"]["LAST_N_WEIGHTED_AVG_VALUES"])
                    continue
                if product == "SQUID_INK":
                    temp = self.run_squid_ink(state.order_depths[product], state.position.get(product, 0), trader_data["SQUID_INK"]["LAST_N_MID_VALUES"],PnL)
                    # print(temp)
                    result[product] = temp[0]
                    trader_data["DEBUG_PRINT"] = str(temp[1:])
                    continue
        
        # highest_kelp_buy = max(state.order_depths["KELP"].buy_orders.keys())
        # lowest_kelp_sell = min(state.order_depths["KELP"].sell_orders.keys())
        # mid_val = (highest_kelp_buy+lowest_kelp_sell)/2
        
        weighted_avg_kelp = (sum(map(lambda x: x[0]*x[1], state.order_depths["KELP"].buy_orders.items()))+sum(map(lambda x: x[0]*(-x[1]), state.order_depths["KELP"].sell_orders.items()))) / (sum(map(lambda x: x[1], state.order_depths["KELP"].buy_orders.items()))+sum(map(lambda x: (-x[1]), state.order_depths["KELP"].sell_orders.items())))
        trader_data["KELP"]["LAST_N_WEIGHTED_AVG_VALUES"].append(weighted_avg_kelp)

        n = params["KELP"]["HISTORY_LENGTH(N)"]
        trader_data["KELP"]["LAST_N_WEIGHTED_AVG_VALUES"]=trader_data["KELP"]["LAST_N_WEIGHTED_AVG_VALUES"][-n:]


        n = params["SQUID_INK"]["HISTORY_LENGTH(N)"]
        rainforest_resin_mid_val = (sum(map(lambda x: x[0]*x[1], state.order_depths["RAINFOREST_RESIN"].buy_orders.items()))+sum(map(lambda x: x[0]*(-x[1]), state.order_depths["RAINFOREST_RESIN"].sell_orders.items()))) / (sum(map(lambda x: x[1], state.order_depths["RAINFOREST_RESIN"].buy_orders.items()))+sum(map(lambda x: (-x[1]), state.order_depths["RAINFOREST_RESIN"].sell_orders.items())))
        lowest_squid_ink_sell = min(state.order_depths["SQUID_INK"].sell_orders.keys())
        highest_squid_ink_buy = max(state.order_depths["SQUID_INK"].buy_orders.keys())
        squid_ink_mid_val = (highest_squid_ink_buy+lowest_squid_ink_sell)/2

        trader_data["RAINFOREST_RESIN"]["LAST_MID_VALUE"] = rainforest_resin_mid_val

        trader_data["SQUID_INK"]["LAST_N_MID_VALUES"].append(squid_ink_mid_val)
        trader_data["SQUID_INK"]["LAST_N_MID_VALUES"] =  trader_data["SQUID_INK"]["LAST_N_MID_VALUES"][-n:]

        for symbol,trades in state.market_trades.items():
            for trade in trades:
                if trade.buyer == "SUBMISSION":
                    trader_data["EARNED_MONEY"] -= trade.price*trade.quantity
                elif trade.seller == "SUBMISSION":  
                    trader_data["SPENT_MONEY"] += trade.price*trade.quantity


            


        trader_data = json.dumps(trader_data)
        conversions = 1 
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
