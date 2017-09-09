import math
import numpy as np
from time import time
from talib import ATR

def initialize(context):
    """
    Initialize parameters.
    """
    context.is_test = True
    context.is_debug = True
    context.is_timed = False
    context.is_info = True

    if context.is_timed:
        start_time = time()

    # Data
    context.symbols = [
        'BP',
        'CD',
        'CL',
        'ED',
        'GC',
        'HG',
        'HO',
        'HU',
        'JY',
        'SB',
        'SF',
        'SP',
        'SV',
        'TB',
        'TY',
        'US',
        'CN',
        'SY',
        'WC',
        'ES',
        'NQ',
        'YM',
        'QM',
        'FV',
    ]
    context.markets = map(
        lambda symbol: continuous_future(symbol),
        context.symbols
    )

    context.price = None
    context.prices = None
    context.contract = None
    context.contracts = None
    context.open_orders = None
    context.average_true_range = {}
    context.dollar_volatility = {}
    context.trade_size = {}

    # Breakout signals
    context.strat_one_breakout = 20
    context.strat_one_breakout_high = {}
    context.strat_one_breakout_low = {}
    context.strat_one_exit = 10
    context.strat_one_exit_high = {}
    context.strat_one_exit_low = {}
    # Keyed by root symbol
    context.is_strat_one = {}

    context.strat_two_breakout = 55
    context.strat_two_breakout_high = {}
    context.strat_two_breakout_low = {}
    context.strat_two_exit = 20
    context.strat_two_exit_high = {}
    context.strat_two_exit_low = {}
    # Keyed by root symbol
    context.is_strat_two = {}

    # Risk
    context.capital = context.portfolio.starting_cash
    context.profit = 0
    context.capital_risk_per_trade = 0.01
    context.capital_multiplier = 2
    context.stop = {}
    context.has_stop = {}
    context.stop_multiplier = 2
    context.market_risk_limit = 4
    context.market_risk = {}
    context.direction_risk_limit = 12
    context.long_risk = 0
    context.short_risk = 0

    # Order
    context.orders = {}
    context.filled = 1
    context.canceled = 2
    context.rejected = 3
    context.long_direction = 'long'
    context.short_direction = 'short'

    for market in context.markets:
        context.orders[market] = []
        context.stop[market] = 0
        context.has_stop[market] = False
        context.market_risk[market] = 0

    schedule_function(
        get_prices,
        date_rules.every_day(),
        time_rules.market_open(),
        False
    )

    schedule_function(
        validate_prices,
        date_rules.every_day(),
        time_rules.market_open(),
        False
    )

    schedule_function(
        compute_highs,
        date_rules.every_day(),
        time_rules.market_open(),
        False
    )

    schedule_function(
        compute_lows,
        date_rules.every_day(),
        time_rules.market_open(),
        False
    )

    schedule_function(
        log_risks,
        date_rules.every_day(),
        time_rules.market_close(minutes=1),
        False
    )

    schedule_function(
        clear_stops,
        date_rules.every_day(),
        time_rules.market_close(minutes=1),
        False
    )

    schedule_function(
        turn_limit_to_market_orders,                #make sure the limit orders are filled
        date_rules.every_day(),
        time_rules.market_close(minute=10)
    )

    total_minutes = 6*60 + 30

    for i in range(30, total_minutes, 30):
        schedule_function(
            get_contracts,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            compute_average_true_ranges,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            compute_dollar_volatilities,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            compute_trade_sizes,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            update_risks,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            detect_entry_signals,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            scaling_signals,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            place_stop_orders,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )
        schedule_function(
            stop_trigger_cleanup,
            date_rules.every_day(),
            time_rules.market_open(minutes=i),
            False
        )

        if context.is_debug:
            schedule_function(
                log_context,
                date_rules.week_end(),
                time_rules.market_close()
            )

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def clear_stops(context, data):
    """
    Clear stops 1 minute before market close.
    """
    if context.is_timed:
        start_time = time()

    for market in context.markets:
        order_info = get_order(context.orders[market][-1])

        if order_info.stop is not None and order_info.status == 0:
            cancel_order(context.orders[market][-1])


    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def log_context(context, data):
    log.info('Porfolio cash: %.2f' % context.portfolio.cash)
    log.info('Capital: 		 %.2f' % context.capital)
    log.info('Prices: 		 %.2f' % context.price)
    log.info(context.open_orders)
    log.info(context.market_risk)

def log_risks(context, data):
    """
    Log long and short risk 1 minute before market close.
    """
    record(
        long_risk = context.long_risk,
        short_risk = context.short_risk
    )

def get_prices(context, data):
    """
    Get high, low, and close prices.
    """
    if context.is_timed:
        start_time = time()

    fields = ['high', 'low', 'close']
    bars = strat_two_breakout + 1
    frequency = '1d'

    # Retrieves a pandas panel with axes labelled as:
    # (Index: field, Major-axis: date, Minor-axis: market)
    context.prices = data.history(
        context.markets,
        fields,
        bars,
        frequency
    )

    if context.is_test:
        assert(context.prices.shape[0] == 3)


    # Tranpose/Reindex panel in axes with:
    # (Index: market, Major-axis: field, Minor-axis: date)
    context.prices = context.prices.transpose(2, 0, 1)
    context.prices = context.prices.reindex()

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 8192)

def validate_prices(context, data):
# data is not used
    """
    Drop markets with null prices.
    """
    if context.is_timed:
        start_time = time()

    context.prices.dropna(axis=0, inplace=True)

    validated_markets = map(
        lambda market: market.root_symbol,
        context.prices.axes[0]
    )

    markets = map(
        lambda market: market.root_symbol,
        context.markets
    )

    dropped_markets = list(
        set(markets) - set(validated_markets)
    )

    if context.is_debug and dropped_markets:
        log.debug(
            'Null prices for %s. Dropped.'
            % ', '.join(dropped_markets)
        )


    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def compute_highs(context, data):
# data is not used
    """
    Compute high for breakout and exits
    """
    if context.is_timed:
        start_time = time()

    for market in context.prices.axes[0]:
        context.strat_one_breakout_high[market] = context.prices\
            .loc[market, 'high']\
            [-context.strat_one_breakout-1:-1]\
            .max()
        context.strat_two_breakout_high[market] = context.prices\
            .loc[market, 'high']\
            [-context.strat_two_breakout-1:-1]\
            .max()
        context.strat_one_exit_high[market] = context.prices\
            .loc[market, 'high']\
            [-context.strat_one_exit-1:-1]\
            .max()
        context.strat_two_exit_high[market] = context.prices\
            .loc[market, 'high']\
            [-context.strat_two_exit-1:-1]\
            .max()

    if context.is_test:
        assert(len(context.strat_one_breakout_high) > 0)
        assert(len(context.strat_two_breakout_high) > 0)
        assert(len(context.strat_one_exit_high) > 0)
        assert(len(context.strat_two_exit_high) > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def compute_lows(context, data):
# data is not used
    """
    Compute 20 and 55 day low.
    """
    if context.is_timed:
        start_time = time()

    for market in context.prices.axes[0]:
        context.strat_one_breakout_low[market] = context.prices\
            .loc[market, 'low']\
            [-context.strat_one_breakout-1:-1]\
            .min()
        context.strat_two_breakout_low[market] = context.prices\
            .loc[market, 'low']\
            [-context.strat_two_breakout-1:-1]\
            .min()
        context.strat_one_exit_low[market] = context.prices\
            .loc[market, 'low']\
            [-context.strat_one_exit-1:-1]\
            .min()
        context.strat_two_exit_low[market] = context.prices\
            .loc[market, 'low']\
            [-context.strat_two_exit-1:-1]\
            .min()

    if context.is_test:
        assert(len(context.strat_one_breakout_low) > 0)
        assert(len(context.strat_two_breakout_low) > 0)
        assert(len(context.strat_one_exit_low) > 0)
        assert(len(context.strat_two_exit_low) > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def get_contracts(context, data):
    """
    Get futures contracts.
    """
    if context.is_timed:
        start_time = time()

    fields = 'contract'

    context.contracts = data.current(
        context.markets,
        fields
    )

    context.open_orders = get_open_orders()

    if context.is_test:
        assert(context.contracts.shape[0] > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def compute_average_true_ranges(context, data):
# data is not used
    """
    Compute average true ranges, or N.
    """
    if context.is_timed:
        start_time = time()

    rolling_window = 21
    moving_average = 20

    for market in context.prices.axes[0]:
        context.average_true_range[market] = ATR(
            context.prices.loc[market, 'high'][-rolling_window:],
            context.prices.loc[market, 'low'][-rolling_window:],
            context.prices.loc[market, 'close'][-rolling_window:],
            timeperiod=moving_average
        )[-1]

    if context.is_test:
        assert(len(context.average_true_range) > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def compute_dollar_volatilities(context, data):
# data is not used
    """
    Compute dollar volatilities, or dollars per point.
    """
    if context.is_timed:
        start_time = time()

    for market in context.prices.axes[0]:
        context.contract = context.contracts[market]
        context.dollar_volatility[market] = context.contract.multiplier\
            * context.average_true_range[market]

    if context.is_test:
        assert(len(context.dollar_volatility) > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def compute_trade_sizes(context, data):
# data is not used
    """
    Compute trade sizes, or amount per trade.
    """
    if context.is_timed:
        start_time = time()

    context.profit = context.portfolio.portfolio_value\
        - context.portfolio.starting_cash

    if context.profit < 0:
        context.capital = context.portfolio.starting_cash\
            + context.profit\
            * context.capital_multiplier

    if context.capital <= 0:
        for market in context.prices.axes[0]:
            context.trade_size[market] = 0
    else:
        for market in context.prices.axes[0]:
            context.trade_size[market] = int(context.capital\
                * context.capital_risk_per_trade\
                / context.dollar_volatility[market])

    if context.is_test:
        assert(len(context.trade_size) > 0)

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def update_risks(context, data):
# data is not used
    """
    Update long, short, and market risks.
    """
    context.long_risk = 0
    context.short_risk = 0

    for position in context.portfolio.positions:
        market = sid(position.sid)
        amount = position.amount
        context.market_risk[market] = amount / context.trade_size[market]

        if context.market_risk[market] > 0:
            context.long_risk += abs(context.market_risk[market])
        elif context.market_risk[market] < 0:
            context.short_risk += abs(context.market_risk[market])

def place_stop_orders(context, data):
# data is not used
    """
    Place stop orders at 2 times average true range or continue a stop order that is canceled when market close.
    """
    for position in context.portfolio.positions:
        market = sid(position.sid)
        amount = position.amount
        order_info = get_order(context.orders[sid(position.sid).root_symbol][-1])

        #If the previous order is a limit order that starts to be filled
        if (order_info.filled != 0 and order_info.limit is not None):

            current_highest_price = order_info.limit

            try:
                context.price = context.prices.loc[market, 'close'][-1]
            except KeyError:
                context.price = 0

            if amount > 0:
                context.stop[market] = current_highest_price\
                - context.average_true_range[market]\
                * context.stop_multiplier

                order_identifier = order_target(
                    position,
                    0,
                    style=StopOrder(context.stop[market])
                )
            elif amount < 0:
                context.stop[market] = current_highest_price\
                    + context.average_true_range[market]\
                    * context.stop_multiplier

                order_identifier = order_target(
                    position,
                    0,
                    style=StopOrder(context.stop[market])
                )
            else:
                order_identifier = None


            if order_identifier is not None:
                context.orders[market].append(order_identifier)

            if context.is_info:
                log.info(
                    'Stop %s %.2f'
                    % (
                        market.root_symbol,
                        context.stop[market]
                    )
                )

        elif (order_info.stop_reached == False and\
            order_info.stop is not None and order_info.status == 2):
            """
            If stop order is created but canceled due to end of day
            """

            context.stop[market] = order_info.stop

            try:
                context.price = context.prices.loc[market, 'close'][-1]
            except KeyError:
                context.price = 0

            if amount > 0:
                order_identifier = order_target(
                    position,
                    0,
                    style=StopOrder(context.stop[market])
                )
            elif amount < 0:
                order_identifier = order_target(
                    position,
                    0,
                    style=StopOrder(context.stop[market])
                )
            else:
                order_identifier = None


            if order_identifier is not None:
                context.orders[market].append(order_identifier)

            if context.is_info:
                log.info(
                    'Stop %s %.2f'
                    % (
                        market.root_symbol,
                        context.stop[market]
                    )
                )


def detect_entry_signals(context, data):
# data is not used
    """
      Place limit orders on 20 or 55 day breakout.
    """
    long_quota = context.direction_risk_limit - math.ceil(context.long_risk)
    short_quota = context.direction_risk_limit - math.ceil(context.short_risk)

    for market in context.prices.axes[0]:
        context.price = data.current(market, 'price')

        if context.portfolio.cash <= 0 or context.capital <= (context.trade_size[market]*context.price):
            continue

        if context.price > context.strat_one_breakout_high[market]:
            if context.market_risk[market] == 0 and long_quota > 0 and\
                get_order(context.orders[market][-1]).limit is None:
                """
                The third condition is to prevent the program to place another limit order after placing one during the last
                schedule function. If last order is limit order, it means a limit order is placed already. If last order is 
                stop order, either it has stopped (then we enter), or the limit order is getting filled (This will violate the
                first condition and order will not be placed).

                """
                order_identifier = order(
                    context.contract,
                    context.trade_size[market],
                    style=LimitOrder(context.price)
                )
                if order_identifier is not None:
                    context.orders[market].append(order_identifier)
                
                long_quota - 1
                context.is_strat_one[market] = True
                context.is_strat_two[market] = False

                if context.is_info:
                    log.info(
                        'Long %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )

        elif context.price > context.strat_two_breakout_high[market] and\
                get_order(context.orders[market][-1]).limit is None:

            if context.market_risk[market] == 0 and long_quota > 0:
                order_identifier = order(
                    context.contract,
                    context.trade_size[market],
                    style=LimitOrder(context.price)
                )
                if order_identifier is not None:
                    context.orders[market].append(order_identifier)

                long_quota - 1
                context.is_strat_one[market] = False
                context.is_strat_two[market] = True

                if context.is_info:
                    log.info(
                        'Long %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )

        elif context.price < context.strat_one_breakout_low[market] and\
                get_order(context.orders[market][-1]).limit is None:

            if context.market_risk[market] == 0 and short_quota > 0:
                order_identifier = order(
                    context.contract,
                    -context.trade_size[market],
                    style=LimitOrder(context.price)
                )
                if order_identifier is not None:
                    context.orders[market].append(order_identifier)

                short_quota - 1
                context.is_strat_one = True
                context.is_strat_two = False

                if context.is_info:
                    log.info(
                        'Short %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )


        elif   context.price < context.strat_two_breakout_low[market] and\
                get_order(context.orders[market][-1]).limit is None:

            if context.market_risk[market] == 0 and short_quota > 0:
                order_identifier = order(
                    context.contract,
                    -context.trade_size[market],
                    style=LimitOrder(context.price)
                )
                if order_identifier is not None:
                    context.orders[market].append(order_identifier)

                short_quota - 1
                context.is_strat_one = False
                context.is_strat_two = True

                if context.is_info:
                    log.info(
                        'Short %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )

#Exit Strategy
    for position in context.portfolio.positions:
        symbol = sid(position.sid).root_symbol

        if context.is_strat_one[symbol]:
            if position[market].amount > 0:
                if context.price == context.strat_one_exit_low[market]:
                    order_target_percent(context.portfolio, 0)
                    context.is_strat_one[symbol] = False

            elif position[market].amount< 0:
                if price == context.strat_one_exit_high[market]:
                    order_target_percent(context.portfolio, 0)
                    context.is_strat_one[symbol] = False


        elif context.is_strat_two[symbol]:
            if context.position[market].amount > 0:
                if context.price == context.strat_two_exit_low[market]:
                    order_target_percent(context.portfolio, 0)
                    context.is_strat_two[symbol] = False

            elif position[market].amount < 0:
                if context.price == context.strat_two_exit_high[market]:
                    order_target_percent(context.portfolio, 0)
                    context.is_strat_two[symbol] = False

def scaling_signals(context,data) 
"""
This function need to be placed after place_stop_order
"""
    for market in context.prices.axes[0]:
        if context.market_risk[market] != 0 and abs(round(context.market_risk[market])) < context.market_risk_limit:

            context.price = data.current(market, 'price')

            if context.market_risk[market] > 0:
                if context.price > get_order(context.orders[market][-1]).stop + (2.5)*(context.average_true_range[market]):

                    order_identifier = order(
                    context.contract,
                    context.trade_size[market],
                    style=LimitOrder(context.price)
                    )

                    if order_identifier is not None:
                        context.orders[market].append(order_identifier)
                    

            elif context.market_risk[market] < 0:
                if context.price < get_order(context.orders[market][-1]).stop - (2.5) * (context.average_true_range[market]): 
                
                    order_identifier = order(
                    context.contract,
                    -context.trade_size[market],
                    style=LimitOrder(context.price)
                    )

                    if order_identifier is not None:
                        context.orders[market].append(order_identifier)

def stop_trigger_cleanup(context,data)

    for market in context.markets:
        order_info = get_order(context.orders[market][-1])
        
        if order_info.stop_reached == True:
            current_open_orders = get_open_orders(context.orders[market][-1].sid)

            for order in current_open_orders:
                cancel_order(order)

def turn_limit_to_market_orders(context,data)

    open_orders = get_open_orders

    for order in open_orders:
        open_order = get_order(order)
        if open_order.limit is not None:
            order_identifier = order(sid(open_order.sid) , (open_order.amount - open_order.filled))

            if order_identifier is not None:
                context.orders[sid(open_order.sid).root_symbol].append(order_identifier)
         
            cancel_order(order)
            

