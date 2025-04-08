from typing import Dict, List, Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Position
import json
f=1
def debug_print(a,f):
    if f:
        print(a)
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
        if len(last_n_trades) == 7:
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
    
    def run_squid_ink(self, order_depths: OrderDepth, position: Position, last_n_plus_3_trades_inp) -> List[Order]:

        symbol = "SQUID_INK"
        orders: List[Order] = []


                                                                # <predicting market price>

        n = 7         # changeable parameters
        threshold = 9 # changeable parameters

                                                                    # <n dependent variables>
        
        powers = [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 2],
            [3, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0],
            [2, 0, 1, 0, 0, 0, 0],
            [2, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 0, 1, 0, 0],
            [2, 0, 0, 0, 0, 1, 0],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 2, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
            [1, 0, 2, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 2, 0],
            [1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [0, 3, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0],
            [0, 2, 0, 1, 0, 0, 0],
            [0, 2, 0, 0, 1, 0, 0],
            [0, 2, 0, 0, 0, 1, 0],
            [0, 2, 0, 0, 0, 0, 1],
            [0, 1, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 2, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 2, 0, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 2, 0],
            [0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 2],
            [0, 0, 3, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0],
            [0, 0, 2, 0, 0, 1, 0],
            [0, 0, 2, 0, 0, 0, 1],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 2, 0, 0],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 2, 0],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 2],
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0],
            [0, 0, 0, 2, 0, 1, 0],
            [0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 2, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 2],
            [0, 0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 2, 1, 0],
            [0, 0, 0, 0, 2, 0, 1],
            [0, 0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 2, 1],
            [0, 0, 0, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 3]
        ]

        coeffs = [
                    -9.01296786258271e-09,
                    -0.10896882846497509,
                    -0.10853982919194856,
                    -0.10797172136815252,
                    -0.10836245006176713,
                    -0.10717362854214801,
                    -0.10630809589458619,
                    -0.10611668475280976,
                    0.1018441088127854,
                    -0.29156263370672636,
                    0.15030977435755485,
                    -0.27970276573232306,
                    -0.10856180727323031,
                    0.7202027229620934,
                    -0.394296885687019,
                    0.08841203603686551,
                    -0.07876974851221297,
                    0.4639642198438167,
                    -0.1338209748056427,
                    -0.30737611618016203,
                    0.1704964007391532,
                    0.038419655754387494,
                    -0.35297065070996425,
                    0.28954322872379035,
                    -0.5936142752168633,
                    0.5088708892379177,
                    0.2017921011370662,
                    0.3664161731467015,
                    -0.09256800886548264,
                    -0.5082676826299879,
                    -0.33066729858376653,
                    0.435871169142668,
                    -0.18755188929994585,
                    -0.14802778415890552,
                    0.13385752898933034,
                    0.13861381729691324,
                    0.00033697203339216886,
                    0.000577036622566876,
                    -0.00026703343509569544,
                    -0.002085712410566487,
                    0.0006488737222526131,
                    -0.0022226528642789456,
                    0.0022908850274204196,
                    0.00010065343042160724,
                    -0.0036152458122224174,
                    -0.0013891283203401016,
                    0.011499487875956338,
                    -0.001993802075751727,
                    -0.005711921780282625,
                    -0.0006885566880878158,
                    0.014481913617715864,
                    -0.01105626390136949,
                    -0.0002795159763976651,
                    0.0022982013572447266,
                    -0.009217706690401395,
                    0.0011676219870600663,
                    0.007305841244967498,
                    0.0011866424602698378,
                    -0.000660773260111186,
                    0.004303907546897023,
                    -0.005842868665378403,
                    -0.0019415051623548618,
                    -0.0013635024107342963,
                    0.0025227981029237005,
                    -0.0012598097621594473,
                    0.0019788666202843475,
                    0.005873556118795635,
                    -0.0109564959493781,
                    0.0013847802133823972,
                    0.0053491382344005856,
                    0.005319428824723138,
                    -0.020257391457117968,
                    0.009506692940759204,
                    0.0034406767191801635,
                    -0.0036240608493637166,
                    0.00990513006240662,
                    0.0010213965077521259,
                    -0.009504834567627851,
                    -0.0016613418070822866,
                    0.0012582494694848984,
                    -0.008206069214710995,
                    0.0056449819327975895,
                    0.004604671566847407,
                    0.004442263670410222,
                    -0.004937645231239185,
                    -0.0021830550683160366,
                    0.003940928769105734,
                    -0.000741610076521515,
                    -0.002105082347255538,
                    0.000804357527374705,
                    -0.0024380886490421028,
                    -0.0028568526842344955,
                    0.0068516256354572425,
                    -0.0010533346826473167,
                    0.0032743028047156897,
                    0.002429723180374141,
                    -0.0032297903509589387,
                    -0.0031557432816384234,
                    -0.0016366478648261462,
                    0.0026981258235022343,
                    0.0006353888728583695,
                    0.0038040250507793763,
                    -0.0067766944434226595,
                    0.0027217206448240105,
                    -0.007259891580550737,
                    0.013725523034511439,
                    -0.006323941572345726,
                    -0.004886260449524282,
                    0.004990511990065449,
                    -0.0011700057850328738,
                    0.001495410972971567,
                    -0.007177690445013429,
                    0.006235245504385201,
                    0.0006783836456762393,
                    0.0005375669127365423,
                    -0.0015796595707499583,
                    0.0034708882754972414,
                    -0.005639548139745543,
                    0.0021213263441583197,
                    9.187340558305945e-05
                ]

        intercept = 1197.4828546444014

        import copy 
        last_n_plus_3_trades = copy.deepcopy(last_n_plus_3_trades_inp)

                                                                        # <Damping outliers>
        for i in range(10, len(last_n_plus_3_trades)):
            summ = 0
            for j in range(10):
                summ += last_n_plus_3_trades[i-j]

            if abs(last_n_plus_3_trades[i]-summ/10) > threshold:
                last_n_plus_3_trades[i] = summ/10 + threshold*(last_n_plus_3_trades[i]-summ/10)/abs(last_n_plus_3_trades[i]-summ/10)

                                                                        # </Damping outliers>

        last_n_trades = last_n_plus_3_trades[-n:]
                                                                    # </n dependent variables>

        prediction = intercept

        X = []
        for i in range(len(powers)):
            value = 1
            for j in range(len(powers[i])):
                value *= (last_n_trades[j] ** powers[i][j])
            X.append(value)

        for i in range(len(X)):
            prediction += X[i] * coeffs[i]
                                                                # </predicting market price>

        
        buy_at_less_than = prediction - 1
        sell_at_more_than = prediction + 1

        


        



        

        












    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        # debug_print(state.traderData, file=open("testing_out.txt","a"))
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
                    # result[product] = self.run_kelp1(state.order_depths[product] , state.position.get(product, 0))
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
                        # debug_print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        # debug_print("SELL", str(best_bid_volume) + "x", best_bid)
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

        # highest_kelp_buy = max(state.order_depths["KELP"].buy_orders.keys())
        # lowest_kelp_sell = min(state.order_depths["KELP"].sell_orders.keys())
        # mid_val = (highest_kelp_buy+lowest_kelp_sell)/2
        
        mid_val = (sum(map(lambda x: x[0]*x[1], state.order_depths["KELP"].buy_orders.items()))+sum(map(lambda x: x[0]*(-x[1]), state.order_depths["KELP"].sell_orders.items()))) / (sum(map(lambda x: x[1], state.order_depths["KELP"].buy_orders.items()))+sum(map(lambda x: (-x[1]), state.order_depths["KELP"].sell_orders.items())))
        # print("mid_val: ", mid_val, file=open("testing_out.txt", "a"))
        last_n_kelp_trades.append(mid_val)

        
        # last_n_kelp_trades.append(mid_val)
        trader_data = json.dumps(last_n_kelp_trades[-7:])
        # debug_print("last_n_kelp_trades: ", last_n_kelp_trades, file=open("testing_out.txt", "a"))
        # debug_print("trader_Data: ", trader_data, file=open("testing_out.txt", "a"))
        
        conversions = 1 

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
