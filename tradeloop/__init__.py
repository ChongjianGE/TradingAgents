"""A trading loop around TradingAgents: analyse on a schedule, turn decisions into
target positions, check them against fixed rules, and place orders on a paper account.

The analysis is TradingAgents'; everything from the decision to the order is plain
code, so the same decision and the same book always give the same orders.
"""
