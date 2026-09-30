-- Dense SKU x week demand panel (zero-filled) for the top-N SKUs.
WITH sku_stats AS (
    SELECT stock_code,
           SUM(quantity)              AS total_qty,
           COUNT(DISTINCT week_start) AS active_weeks
    FROM raw_sales
    WHERE is_return = 0
    GROUP BY stock_code
),
top_skus AS (
    SELECT stock_code
    FROM sku_stats
    WHERE active_weeks >= :min_weeks
    ORDER BY total_qty DESC, stock_code
    LIMIT :top_n
),
weekly AS (
    SELECT stock_code,
           week_start,
           SUM(CASE WHEN is_return = 0 THEN quantity ELSE 0 END)        AS sales_qty,
           SUM(CASE WHEN is_return = 1 THEN -quantity ELSE 0 END)       AS returns_qty,
           AVG(CASE WHEN is_return = 0 THEN price END)                  AS avg_price,
           COUNT(DISTINCT CASE WHEN is_return = 0 THEN invoice END)     AS n_invoices,
           COUNT(DISTINCT CASE WHEN is_return = 0 THEN customer_id END) AS n_customers
    FROM raw_sales
    GROUP BY stock_code, week_start
)
SELECT t.stock_code                AS sku,
       c.week_start                AS week_start,
       COALESCE(w.sales_qty, 0)    AS sales_qty,
       COALESCE(w.returns_qty, 0)  AS returns_qty,
       w.avg_price                 AS avg_price,
       COALESCE(w.n_invoices, 0)   AS n_invoices,
       COALESCE(w.n_customers, 0)  AS n_customers
FROM top_skus t
CROSS JOIN calendar_weeks c
LEFT JOIN weekly w
       ON w.stock_code = t.stock_code AND w.week_start = c.week_start
ORDER BY t.stock_code, c.week_start
