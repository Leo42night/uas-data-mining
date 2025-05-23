SELECT COUNT(DISTINCT state) AS total_kota_yang_membeli 
FROM wpkr_wc_order_addresses
WHERE state IS NOT NULL AND state <> '';

