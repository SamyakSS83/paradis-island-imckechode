from typing import Dict, List, Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Position
import json


POS_LIMITS = {"RAINFOREST_RESIN": 50, "KELP": 50}

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

class Trader:

    def run_raisin(self, order_depths: OrderDepth, position: Position) -> List[Order]:
        symbol = "RAINFOREST_RESIN"
        orders: List[Order] = [];
        buy_at_less_than = 10000
        sell_at_more_than = 10000

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

        orders.append(Order(symbol, sell_at_more_than+2, -POS_LIMITS[symbol]-position)) #position+x = -pos_limit
        orders.append(Order(symbol, buy_at_less_than-2, POS_LIMITS[symbol]-position))
        return orders

    def run_kelp(self, order_depths: OrderDepth, position: Position, last_n_trades) -> List[Order]:
        symbol = "KELP"
        orders: List[Order] = [];
        if len(last_n_trades) == 10:
            intercept = 16.55687365458857
            coeffs = [-0.00442061, -0.01372848, -0.00291883,  0.03911388,  0.04284338,  0.06904266,
  0.11434546,  0.11314294,  0.25076363,  0.38360758]
        
            my_pred = intercept
            for i in range(len(last_n_trades)):
                my_pred += last_n_trades[i]*coeffs[i] #last_n_trades: [[trade.price, trade.quantity, trade.timestamp], ...]
            # my_pred = sum(last_n_trades[-1:])/len(last_n_trades[-1:])
            buy_at_less_than = round(my_pred)
            sell_at_more_than = round(my_pred)

            print("prediction: ", my_pred, file=open("testing_out.txt", "a"))
            print("last_n_trades: ", last_n_trades, file=open("testing_out.txt", "a"))

            buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depths.sell_orders.items())

            buy_sell_limit = 50 #Do not place orders of quantity more than this to manage product placement

            total_vol_for_instant_sell = 0
            for (price, amt) in buy_orders: #We decide the best people to sell to based on the buy orders.
                print("While deciding sells, I see: ", price, amt, file=open("testing_out.txt", "a"))
                if (price>sell_at_more_than):
                    print("It is good", file=open("testing_out.txt", "a"))
                    total_vol_for_instant_sell += amt
                
            if position-total_vol_for_instant_sell < -POS_LIMITS[symbol]:
                orders.append(Order(symbol, sell_at_more_than+1, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.6)))
                orders.append(Order(symbol, sell_at_more_than+0, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.2)))
                orders.append(Order(symbol, sell_at_more_than+2, round(-min(position+POS_LIMITS[symbol], buy_sell_limit)*0.2)))

            
            else:
                orders.append(Order(symbol, sell_at_more_than+1, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.6)))
                orders.append(Order(symbol, sell_at_more_than+0, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.2)))
                orders.append(Order(symbol, sell_at_more_than+2, round(-min(total_vol_for_instant_sell, buy_sell_limit)*0.2)))

                
            total_vol_for_instant_buy = 0
            for (price, amt) in sell_orders: #We decide the best people to buy from based on the sell orders.
                assert(amt<=0)
                if (price<buy_at_less_than):
                    total_vol_for_instant_buy -= amt
            
            if total_vol_for_instant_buy+position > POS_LIMITS[symbol]:
                orders.append(Order(symbol, buy_at_less_than-1, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.6)))
                orders.append(Order(symbol, buy_at_less_than+0, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.2)))
                orders.append(Order(symbol, buy_at_less_than-2, round(min(POS_LIMITS[symbol]-position, buy_sell_limit)*0.2)))
            else:
                orders.append(Order(symbol, buy_at_less_than-1, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.6)))
                orders.append(Order(symbol, buy_at_less_than+0, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.2)))
                orders.append(Order(symbol, buy_at_less_than-2, round(min(total_vol_for_instant_buy, buy_sell_limit)*0.2)))
            print("orders: ", orders, file=open("testing_out.txt", "a"))
            print("position: ", orders, file=open("testing_out.txt", "a"))
            print("total_vol_for_instant_buy: ", total_vol_for_instant_buy, file=open("testing_out.txt", "a"))
            print("total_vol_for_instant_sell: ", total_vol_for_instant_sell, file=open("testing_out.txt", "a"))
            # orders.append(Order(symbol, sell_at_more_than+1, -POS_LIMITS[symbol]-position)) #position+x = -pos_limit
            # orders.append(Order(symbol, buy_at_less_than-1, POS_LIMITS[symbol]-position))
        return orders
    



    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        # print(state.traderData, file=open("testing_out.txt","a"))
        if state.traderData != "":
            last_n_kelp_trades = json.loads(state.traderData)
        else:
            last_n_kelp_trades = []

        # conversions = 0

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
                if product == "RAINFOREST_RESIN":
                    result[product] = self.run_raisin(state.order_depths[product] , state.position.get(product, 0))
                    continue
                if product == "KELP":
                    result[product] = self.run_kelp(state.order_depths[product] , state.position.get(product, 0), last_n_kelp_trades)
                    continue
                # Retrieve the Order Depth containing all the market BUY and SELL orders
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Note that this value of 1 is just a dummy value, you should likely change it!
                acceptable_price = 10000

                # If statement checks if there are any SELL orders in the market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        # print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        # print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders
        
        # kelp_trades: List[Trade] = state.market_trades.get("KELP", [])
        # kelp_trades.sort(key=lambda x: x.timestamp, reverse=True)
        # if len(kelp_trades) >= 5:
        #     trader_data = json.dumps(map(lambda x: [x.price, x.quantity, x.timestamp], kelp_trades[:5]))
        # else:
        #     # trader_data = "SomethingHere"
        #     trader_data = json.dumps(list(map(lambda x: [x.price, x.quantity, x.timestamp], kelp_trades))+last_n_kelp_trades[:5-len(kelp_trades)])

        highest_kelp_buy = max(state.order_depths["KELP"].buy_orders.keys())
        lowest_kelp_sell = min(state.order_depths["KELP"].sell_orders.keys())
        mid_val = (highest_kelp_buy+lowest_kelp_sell)/2
        
        last_n_kelp_trades.append(mid_val)
        trader_data = json.dumps(last_n_kelp_trades[-10:])
        print("last_n_kelp_trades: ", last_n_kelp_trades, file=open("testing_out.txt", "a"))
        print("trader_Data: ", trader_data, file=open("testing_out.txt", "a"))
        
        conversions = 1 

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
