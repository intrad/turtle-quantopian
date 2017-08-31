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

    context.strat_two_breakout = 55
    context.strat_two_breakout_high = {}
    context.strat_two_breakout_low = {}
    context.strat_two_exit = 20
    context.strat_two_exit_high = {}
    context.strat_two_exit_low = {}

    # Risk
    context.price_threshold = 1
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
        clear_stops,
        date_rules.every_day(),
        time_rules.market_open(minutes=1),
        False
    )

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
            place_stop_orders,
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
    Clear stops 1 minute after market open.
    """
    if context.is_timed:
        start_time = time()

    for market in context.markets:
        context.has_stop[market] = False

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

def log_context(context, data):
    log.info('Porfolio cash: %.2f' % context.portfolio.cash)
    log.info('Capital: 		 %.2f' % context.capital)
    log.info('Prices: 		 %.2f' % context.price)
    log.info(context.price_threshold)
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

def is_trade_allowed(context, market, direction):
    """
    Check if allowed to trade.
    """
    if context.is_timed:
        start_time = time()

    is_trade_allowed = True

    if context.portfolio.cash <= 0:
        is_trade_allowed = False

    if context.capital <= 0:
        is_trade_allowed = False

    # @Todo check WTF is price_threshold
    if context.price < context.price_threshold:
        is_trade_allowed = False

    if context.open_orders:
        if context.contracts[market] in context.open_orders:
            is_trade_allowed = False

    if context.market_risk[market] >= context.market_risk_limit:
        is_trade_allowed = False

    if direction == context.long_direction\
            and context.long_risk >= context.direction_risk_limit:
        is_trade_allowed = False

    if direction == context.short_direction\
            and context.short_risk >= context.direction_risk_limit:
        is_trade_allowed = False

    if context.is_timed:
        time_taken = (time() - start_time) * 1000
        log.debug('Executed in %f ms.' % time_taken)
        assert(time_taken < 1024)

    return is_trade_allowed

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
    for market in context.orders:
        for order_identifier in context.orders[market]:
            a_order = get_order(order_identifier)

            if a_order.status == context.filled:
                if a_order.limit_reached:
                    context.market_risk[market] += 1

                    if a_order.amount > 0:
                        context.long_risk += 1
                    if a_order.amount < 0:
                        context.short_risk += 1

                if a_order.stop_reached:
                    context.market_risk[market] -= 1

                    if a_order.amount > 0:
                        context.long_risk -= 1
                    if a_order.amount < 0:
                        context.short_risk -= 1

                context.orders[market].remove(order_identifier)

            if a_order.status == context.canceled\
                    or a_order.status == context.rejected:
                context.orders[market].remove(order_identifier)

def place_stop_orders(context, data):
# data is not used
    """
    Place stop orders at 2 times average true range.
    """
    for position in context.portfolio.positions:
        market = continuous_future(position.root_symbol)
        amount = context.portfolio.positions[position].amount
        check_if_filled = get_order(context.orders[position.root_symbol][-1])

        if (check_if_filled.filled != 0 and check_if_filled.limit is not None)\
           or (check_if_filled.status == 2 and check_if_filled.stop is not None):     #This makes use the context.orders chain 

            if check_if_filled.limit is not None:
                current_highest_price = check_if_filled.limit
            elif check_if_filled.stop is not None:
                current_highest_price = check_if_filled.stop

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
                    amount,
                    style=StopOrder(context.stop[market])
                )
            elif amount < 0:
                context.stop[market] = current_highest_price\
                    + context.average_true_range[market]\
                    * context.stop_multiplier
            
                order_identifier = order_target(
                    position,
                    -amount,
                    style=StopOrder(context.stop[market])
                )

            
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
    for market in context.prices.axes[0]:
        context.price = data.current(market, 'price')

        if context.price > context.strat_one_breakout_high[market]\
        or context.price > context.strat_two_breakout_high[market]:
            if is_trade_allowed(context, market, context.long_direction):
                order_identifier = order(
                    context.contract,
                    context.trade_size[market],
                    style=LimitOrder(context.price)
                )

                if order_identifier is not None:
                    context.orders[market].append(order_identifier)

                if context.is_info:
                    log.info(
                        'Long %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )
        elif context.price < context.strat_one_breakout_low[market]\
        or context.price < context.strat_two_breakout_low[market]:
            if is_trade_allowed(context, market, context.short_direction):
                order_identifier = order(
                    context.contract,
                    -context.trade_size[market],
                    style=LimitOrder(context.price)
                )
                if order_identifier is not None:
                    context.orders[market].append(order_identifier)

                if context.is_info:
                    log.info(
                        'Short %s %i@%.2f'
                        % (
                            market.root_symbol,
                            context.trade_size[market],
                            context.price
                        )
                    )

   for position in context.portfolio.position:
        if position[market].amount !=0
                 