SELECT
  SUM(num_items_sold) AS total_penjualan_barang
FROM wpkr_wc_order_stats
WHERE num_items_sold IS NOT NULL;
