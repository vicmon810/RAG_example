# Task 001: Summarise Transactions

Given a list of transactions with user IDs and amounts, write Python code to produce a summary for each user.

For each user, calculate:
- total amount
- number of transactions
- average transaction amount

Example input:

transactions = [
    {"user": "A", "amount": 10},
    {"user": "B", "amount": 20},
    {"user": "A", "amount": 15}
]

Expected output:

{
    "A": {"total": 25, "count": 2, "average": 12.5},
    "B": {"total": 20, "count": 1, "average": 20.0}
}