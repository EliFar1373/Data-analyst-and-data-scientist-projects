create database finance;
CREATE TABLE stock_prices (
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume INT
);


use finance;
select * from stock_prices;